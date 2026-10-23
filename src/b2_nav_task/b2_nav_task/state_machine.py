"""
Inspection task state machine, independent of ROS.

States follow the experiment plan: task check -> localization init -> track
global subgoals -> search the area -> confirm the object -> execute the
observation pose -> stop and report. The machine only decides; it returns
actions that the ROS node carries out and reports back as events.
"""

from dataclasses import dataclass, field

# States
TASK_CHECK = 'task_check'
LOCALIZATION_INIT = 'localization_init'
TRACK_SUBGOAL = 'track_subgoal'
SEARCH_AREA = 'search_area'
OBSERVE = 'observe'
SUCCEEDED = 'succeeded'
FAILED = 'failed'
TERMINAL = (SUCCEEDED, FAILED)

# Failure reasons, grouped as in the plan's failure attribution.
FAIL_PREFLIGHT = 'preflight_failed'
FAIL_LOCALIZATION = 'localization_failed'
FAIL_SUBGOAL_BLOCKED = 'subgoal_unreachable'
FAIL_SUBGOAL_TIMEOUT = 'subgoal_timeout'
FAIL_NOT_FOUND = 'object_not_found'
FAIL_NOT_CONFIRMED = 'object_not_confirmed'
FAIL_LOOKALIKE = 'lookalike_rejected'
FAIL_OBSERVATION = 'observation_unreachable'
FAIL_CANCELLED = 'cancelled'
FAIL_INTERNAL = 'internal_error'

SEARCH_FAILURES = {
    'not_found': FAIL_NOT_FOUND,
    'not_confirmed': FAIL_NOT_CONFIRMED,
    'lookalike_rejected': FAIL_LOOKALIKE,
    'observation_unreachable': FAIL_OBSERVATION,
}


@dataclass
class Action:
    kind: str  # preflight | wait_localization | navigate | search | spin | stop
    args: dict = field(default_factory=dict)


class TaskMachine:

    def __init__(self, spec):
        self.spec = spec
        self.state = None
        self.failure = None
        self.detail = ''
        self.subgoal_index = 0
        self.subgoal_attempts = 0
        self.subgoals_reached = 0
        self.search_attempts = 0
        self.last_search = None
        self.entries = 0
        self.history = []  # (state, detail) in order

    # -- driving ----------------------------------------------------------

    def start(self):
        return self._enter(TASK_CHECK, [Action('preflight')])

    def timeout_sec(self):
        """Time allowed in the current state, or None for no limit."""
        timeouts = self.spec.timeouts_sec
        return {
            TASK_CHECK: timeouts['preflight'],
            LOCALIZATION_INIT: timeouts['localization'],
            TRACK_SUBGOAL: timeouts['subgoal'],
            # Each search window has its own timeout in the search node;
            # this bounds the whole phase including turns.
            SEARCH_AREA: (
                self.spec.search.max_attempts
                * (self.spec.search.attempt_timeout_sec + 30.0)
            ),
            OBSERVE: timeouts['observation'],
        }.get(self.state)

    def handle(self, event, **data):
        """Apply one event and return the actions to run next."""
        if self.state in TERMINAL:
            return []
        if event == 'cancel':
            return self._fail(FAIL_CANCELLED, data.get('detail', 'cancel requested'))
        if event == 'state_timeout':
            return self._on_timeout()
        handler = getattr(self, f'_in_{self.state}', None)
        return handler(event, data) if handler else []

    # -- per-state handlers -----------------------------------------------

    def _in_task_check(self, event, data):
        if event == 'preflight_ok':
            return self._enter(LOCALIZATION_INIT, [Action('wait_localization')])
        if event == 'preflight_failed':
            return self._fail(FAIL_PREFLIGHT, data.get('detail', ''))
        return []

    def _in_localization_init(self, event, data):
        if event == 'localized':
            return self._start_subgoal(0)
        return []

    def _in_track_subgoal(self, event, data):
        if event == 'nav_succeeded':
            self.subgoals_reached += 1
            if self.subgoal_index + 1 < len(self.spec.route):
                return self._start_subgoal(self.subgoal_index + 1)
            return self._start_search()
        if event == 'nav_failed':
            if self.subgoal_attempts <= self.spec.subgoal_retries:
                return self._navigate_subgoal(f'retry after {data.get("detail", "")}')
            return self._fail(
                FAIL_SUBGOAL_BLOCKED,
                f'subgoal {self.subgoal_index} unreachable: {data.get("detail", "")}',
            )
        return []

    def _in_search_area(self, event, data):
        if event == 'spin_done' or event == 'spin_failed':
            # A failed turn still leaves a new view direction to try.
            return self._search_window()
        if event != 'search_result':
            return []
        self.last_search = data
        status = data.get('status')
        if status == 'confirmed':
            goal = data.get('observation_goal')
            if not goal:
                return self._fail(FAIL_INTERNAL, 'confirmed result without goal')
            return self._enter(OBSERVE, [Action('navigate', {
                'purpose': 'observation',
                'pose': goal,
            })], f'device at {data.get("evidence", {}).get("device_xy")}')
        if status == 'observation_unreachable':
            return self._fail(FAIL_OBSERVATION, 'no free stand-off pose')
        if status in ('not_found', 'not_confirmed', 'lookalike_rejected'):
            if self.search_attempts < self.spec.search.max_attempts:
                return [Action('spin', {'angle': self.spec.search.spin_step_rad})]
            return self._fail(SEARCH_FAILURES[status], f'{self.search_attempts} windows')
        return self._fail(FAIL_INTERNAL, f'search status {status!r}')

    def _in_observe(self, event, data):
        if event == 'nav_succeeded':
            return self._finish()
        if event == 'nav_failed':
            return self._fail(FAIL_OBSERVATION, data.get('detail', ''))
        return []

    # -- helpers ----------------------------------------------------------

    def _start_subgoal(self, index):
        self.subgoal_index = index
        self.subgoal_attempts = 0
        self.state = None  # force a fresh entry record per subgoal
        return self._navigate_subgoal(f'subgoal {index + 1}/{len(self.spec.route)}')

    def _navigate_subgoal(self, detail):
        self.subgoal_attempts += 1
        pose = self.spec.route[self.subgoal_index]
        action = Action('navigate', {
            'purpose': 'subgoal',
            'index': self.subgoal_index,
            'pose': {'x': pose.x, 'y': pose.y, 'yaw': pose.yaw, 'frame': 'map'},
        })
        if self.state == TRACK_SUBGOAL:
            self.history.append((TRACK_SUBGOAL, detail))
            return [action]
        return self._enter(TRACK_SUBGOAL, [action], detail)

    def _start_search(self):
        self.search_attempts = 0
        return self._enter(SEARCH_AREA, self._search_window(), 'arrived at search area')

    def _search_window(self):
        self.search_attempts += 1
        return [Action('search', {
            'attempt': self.search_attempts,
            'device_id': self.spec.device_id,
            'timeout_sec': self.spec.search.attempt_timeout_sec,
        })]

    def _on_timeout(self):
        reason = {
            TASK_CHECK: FAIL_PREFLIGHT,
            LOCALIZATION_INIT: FAIL_LOCALIZATION,
            TRACK_SUBGOAL: FAIL_SUBGOAL_TIMEOUT,
            SEARCH_AREA: FAIL_NOT_FOUND,
            OBSERVE: FAIL_OBSERVATION,
        }.get(self.state, FAIL_INTERNAL)
        # Keep what the state was waiting on (e.g. missing sensors).
        detail = f'timeout in {self.state}'
        if self.detail:
            detail += f': {self.detail}'
        return self._fail(reason, detail)

    def _enter(self, state, actions, detail=''):
        self.state = state
        self.detail = detail
        # Lets the node restart the state timer even for TRACK -> TRACK.
        self.entries += 1
        self.history.append((state, detail))
        return actions

    def _fail(self, reason, detail):
        self.failure = reason
        self._enter(FAILED, [], detail)
        return [Action('stop', {'outcome': FAILED, 'reason': reason})]

    def _finish(self):
        self._enter(SUCCEEDED, [], 'observation pose reached')
        return [Action('stop', {'outcome': SUCCEEDED, 'reason': None})]
