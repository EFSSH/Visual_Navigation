"""
What the robot's own record misses, separated into the two questions of the
close-range case:

1. Was the closest approach recorded? If the detector misses the frames
   around the true minimum distance, the logged clearance is too large and
   naive calibration learns that the interaction was safer than it was
   (the H-A mechanism).
2. How much reaction time was left when the agent was first detected? Even
   a correctly recorded near miss may have been detected too late to yield.

Both are properties of the outcomes alone, so they are computed once on the
test set, per action and per decision-time stratum, independent of the
calibration method.
"""

import numpy as np

from .sim2d import ACTIONS


def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


def detection_diagnostics(scn, robots, agents, dets, hidden0, d_safe):
    """
    ``robots``, ``agents``: (n, A, H+1, 2); ``dets``: (n, A, H+1) detector
    output; ``hidden0``: (n,). Returns {action: {stratum: metrics}}.
    """
    d = np.linalg.norm(robots[..., 1:, :] - agents[..., 1:, :], axis=-1)   # (n, A, H)
    det = dets[..., 1:]
    k_min = d.argmin(-1)
    d_true = np.take_along_axis(d, k_min[..., None], -1)[..., 0]
    d_rec = np.where(det, d, np.inf).min(-1)
    closest_seen = np.take_along_axis(det, k_min[..., None], -1)[..., 0]
    ever = det.any(-1)
    first = det.argmax(-1)
    first_dist = np.where(ever, np.take_along_axis(d, first[..., None], -1)[..., 0], np.inf)
    lead = (k_min - first) * scn.dt
    stop_time = scn.robot_speed0 / scn.robot_accel

    out = {}
    for a, name in enumerate(ACTIONS):
        per = {}
        for label, mask in (('all', np.ones_like(hidden0)), ('hidden0', hidden0),
                            ('visible0', ~hidden0)):
            nm = mask & (d_true[:, a] < d_safe)
            seen = nm & ever[:, a]
            per[label] = {
                'episodes': int(mask.sum()),
                'near_misses': int(nm.sum()),
                # Question 1: is the near miss in the robot's own record?
                'closest_undetected': float((nm & ~closest_seen[:, a]).sum() / max(nm.sum(), 1)),
                'recorded_as_safe': float((nm & (d_rec[:, a] >= d_safe)).sum() / max(nm.sum(), 1)),
                'mean_overestimate_m': float(np.mean(d_rec[seen, a] - d_true[seen, a]))
                if seen.any() else None,
                # Question 2: how much time was left after the first detection?
                'first_detect_dist_p10': _pct(first_dist[seen, a], 10),
                'first_detect_dist_p50': _pct(first_dist[seen, a], 50),
                'lead_time_p10': _pct(lead[seen, a], 10),
                'lead_time_p50': _pct(lead[seen, a], 50),
                'lead_below_stop_time': float((seen & (lead[:, a] < stop_time)).sum()
                                              / max(nm.sum(), 1)),
                'never_detected': float((nm & ~ever[:, a]).sum() / max(nm.sum(), 1)),
            }
        out[name] = per
    return out


def format_diagnostics(diag, actions=('go', 'slow')):
    lines = ['near misses in the test set (true clearance < d_safe), per action and stratum:',
             f'{"":<14}{"n":>6}{"closest":>9}{"logged":>8}{"over-":>7}'
             f'{"1st det m":>11}{"lead s":>11}{"lead<stop":>10}',
             f'{"":<14}{"":>6}{"missed":>9}{"safe":>8}{"est m":>7}'
             f'{"p10/p50":>11}{"p10/p50":>11}{"":>10}']

    def fmt(x, spec='.2f'):
        return '-' if x is None else format(x, spec)

    for action in actions:
        for stratum, m in diag[action].items():
            lines.append(
                f'{action + " " + stratum:<14}{m["near_misses"]:>6}'
                f'{fmt(m["closest_undetected"]):>9}{fmt(m["recorded_as_safe"]):>8}'
                f'{fmt(m["mean_overestimate_m"]):>7}'
                f'{fmt(m["first_detect_dist_p10"], ".1f") + "/" + fmt(m["first_detect_dist_p50"], ".1f"):>11}'
                f'{fmt(m["lead_time_p10"], ".1f") + "/" + fmt(m["lead_time_p50"], ".1f"):>11}'
                f'{fmt(m["lead_below_stop_time"]):>10}')
    return '\n'.join(lines)
