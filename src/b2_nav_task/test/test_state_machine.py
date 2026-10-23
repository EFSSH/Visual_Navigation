from b2_nav_task.state_machine import (
    FAIL_CANCELLED,
    FAIL_LOCALIZATION,
    FAIL_LOOKALIKE,
    FAIL_PREFLIGHT,
    FAIL_SUBGOAL_BLOCKED,
    FAILED,
    LOCALIZATION_INIT,
    OBSERVE,
    SEARCH_AREA,
    SUCCEEDED,
    TaskMachine,
    TRACK_SUBGOAL,
)
from b2_nav_task.task_spec import load_task, TaskError
import pytest

TASK = """
task_id: t1
device_id: PUMP-07
route:
  - {x: 1.0, y: 2.0, yaw: 0.0}
  - {x: 3.0, y: 4.0}
search: {max_attempts: 2, attempt_timeout_sec: 5.0}
retries: {subgoal: 1}
"""

CONFIRMED = {
    'status': 'confirmed',
    'observation_goal': {'x': 5.0, 'y': 5.0, 'yaw': 0.0, 'frame': 'map'},
    'evidence': {'device_xy': [6.0, 5.0]},
}


def _kinds(actions):
    return [a.kind for a in actions]


def _to_search(machine):
    machine.start()
    machine.handle('preflight_ok')
    machine.handle('localized')
    machine.handle('nav_succeeded')
    return machine.handle('nav_succeeded')


def test_task_parsing_and_validation():
    spec = load_task(TASK)
    assert spec.route[1].yaw == 0.0
    assert spec.search.max_attempts == 2
    with pytest.raises(TaskError):
        load_task('task_id: t\ndevice_id: d\nroute: []\n')
    with pytest.raises(TaskError):
        load_task('task_id: t\ndevice_id: d\nroute: [{x: 1}]\n')
    with pytest.raises(TaskError):
        load_task(TASK + 'timeouts_sec: {bogus: 1}\n')


def test_happy_path_reaches_success():
    machine = TaskMachine(load_task(TASK))
    assert _kinds(machine.start()) == ['preflight']
    assert _kinds(machine.handle('preflight_ok')) == ['wait_localization']
    assert machine.state == LOCALIZATION_INIT
    actions = machine.handle('localized')
    assert machine.state == TRACK_SUBGOAL
    assert actions[0].args['pose']['x'] == 1.0
    actions = machine.handle('nav_succeeded')
    assert actions[0].args['index'] == 1
    actions = machine.handle('nav_succeeded')
    assert machine.state == SEARCH_AREA and _kinds(actions) == ['search']
    actions = machine.handle('search_result', **CONFIRMED)
    assert machine.state == OBSERVE
    assert actions[0].args['purpose'] == 'observation'
    actions = machine.handle('nav_succeeded')
    assert machine.state == SUCCEEDED
    assert actions[0].args == {'outcome': SUCCEEDED, 'reason': None}
    assert machine.subgoals_reached == 2


def test_subgoal_retried_then_reported_blocked():
    machine = TaskMachine(load_task(TASK))
    machine.start()
    machine.handle('preflight_ok')
    machine.handle('localized')
    assert _kinds(machine.handle('nav_failed', detail='x')) == ['navigate']
    actions = machine.handle('nav_failed', detail='x')
    assert machine.state == FAILED and machine.failure == FAIL_SUBGOAL_BLOCKED
    assert _kinds(actions) == ['stop']


def test_search_turns_between_windows_then_fails():
    machine = TaskMachine(load_task(TASK))
    _to_search(machine)
    assert _kinds(machine.handle('search_result', status='not_found')) == ['spin']
    assert _kinds(machine.handle('spin_done')) == ['search']
    machine.handle('search_result', status='lookalike_rejected')
    assert machine.state == FAILED and machine.failure == FAIL_LOOKALIKE


def test_failed_turn_still_tries_next_window():
    machine = TaskMachine(load_task(TASK))
    _to_search(machine)
    machine.handle('search_result', status='not_confirmed')
    assert _kinds(machine.handle('spin_failed')) == ['search']


def test_preflight_failure_and_timeouts():
    machine = TaskMachine(load_task(TASK))
    machine.start()
    machine.handle('preflight_failed', detail='no scan')
    assert machine.failure == FAIL_PREFLIGHT

    machine = TaskMachine(load_task(TASK))
    machine.start()
    machine.handle('preflight_ok')
    machine.handle('state_timeout')
    assert machine.failure == FAIL_LOCALIZATION


def test_timeout_keeps_what_the_state_was_waiting_for():
    machine = TaskMachine(load_task(TASK))
    machine.start()
    machine.detail = '/scan not received'
    machine.handle('state_timeout')
    assert machine.failure == FAIL_PREFLIGHT
    assert '/scan not received' in machine.detail


def test_cancel_stops_and_terminal_ignores_events():
    machine = TaskMachine(load_task(TASK))
    machine.start()
    assert _kinds(machine.handle('cancel')) == ['stop']
    assert machine.failure == FAIL_CANCELLED
    assert machine.handle('preflight_ok') == []


def test_each_subgoal_restarts_state_timer():
    machine = TaskMachine(load_task(TASK))
    machine.start()
    machine.handle('preflight_ok')
    machine.handle('localized')
    first = machine.entries
    machine.handle('nav_succeeded')
    assert machine.entries == first + 1
