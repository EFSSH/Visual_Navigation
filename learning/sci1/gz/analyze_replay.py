"""
Analyse Gazebo replay detections (replay_detect.py output).

Per camera:
- recognition curve: P(YOLO person conf >= c | geometric visible fraction bin);
- fitted linear ramp (lo, hi) in the same form as corner_pass (cam_area_lo/hi),
  so the 2D simulator can be rerun with the measured curve;
- episode level: share of encounters never recognised (single-frame and
  3-consecutive-frame confirmation) and recognition delay after the worker
  first becomes geometrically visible.

Run: python -m sci1.gz.analyze_replay /home/xlz/sci1_runs/gz/replay_go.npz [...]
"""

import argparse
import json

import numpy as np

COLS = ['episode', 'step', 't', 'hfov', 'visible', 'gap', 'speed', 'conf']


def ramp_fit(vis, det):
    """Least-squares fit of p = clip((v - lo) / (hi - lo), 0, 1) over a grid."""
    best = (np.inf, None)
    for lo in np.arange(0.0, 0.8, 0.01):
        for hi in np.arange(lo + 0.02, 1.01, 0.01):
            p = np.clip((vis - lo) / (hi - lo), 0, 1)
            err = float(((p - det) ** 2).mean())
            if err < best[0]:
                best = (err, (round(float(lo), 2), round(float(hi), 2)))
    return best[1]


def episode_stats(rows, conf, visible_min, dt_frame):
    stats = {'never_1': 0, 'never_3': 0, 'delay_1': [], 'delay_3': [], 'n': 0}
    for ep in np.unique(rows[:, 0]):
        r = rows[rows[:, 0] == ep]
        r = r[np.argsort(r[:, 1])]
        vis_idx = np.flatnonzero(r[:, 4] >= visible_min)
        if not len(vis_idx):
            continue
        stats['n'] += 1
        det = r[:, 7] >= conf
        first_vis = r[vis_idx[0], 2]
        hits = np.flatnonzero(det)
        if len(hits):
            stats['delay_1'].append(r[hits[0], 2] - first_vis)
        else:
            stats['never_1'] += 1
        run = np.convolve(det.astype(int), np.ones(3, int), 'valid') == 3
        if run.any():
            k = int(np.argmax(run)) + 2               # confirmed at the third hit
            stats['delay_3'].append(r[k, 2] - first_vis)
        else:
            stats['never_3'] += 1
    n = max(stats['n'], 1)
    return {
        'encounters_visible': stats['n'],
        'never_recognised_1frame': stats['never_1'] / n,
        'never_recognised_3frames': stats['never_3'] / n,
        'delay_1frame_median_s': float(np.median(stats['delay_1'])) if stats['delay_1'] else None,
        'delay_3frames_median_s': float(np.median(stats['delay_3'])) if stats['delay_3'] else None,
    }


def analyse(path, conf=0.25, visible_min=0.1):
    d = np.load(path, allow_pickle=True)
    rows = d['rows']
    out = {'file': path, 'mode': str(d['mode']), 'conf': conf, 'cameras': {}}
    for hfov in np.unique(rows[:, 3]):
        r = rows[rows[:, 3] == hfov]
        vis, det = r[:, 4], (r[:, 7] >= conf).astype(float)
        bins = np.linspace(0, 1, 11)
        curve = []
        for a, b in zip(bins[:-1], bins[1:]):
            m = (vis >= a) & (vis < b if b < 1 else vis <= b)
            curve.append({'bin': [round(float(a), 1), round(float(b), 1)], 'frames': int(m.sum()),
                          'p_detect': float(det[m].mean()) if m.any() else None})
        close = r[:, 5] < 0.5
        out['cameras'][f'{int(round(hfov))}deg'] = {
            'frames': int(len(r)),
            'curve': curve,
            'ramp_fit_lo_hi': ramp_fit(vis[vis > 0], det[vis > 0]),
            'p_detect_gap_lt_0.5m': float(det[close].mean()) if close.any() else None,
            'false_positive_rate_vis0': float(det[vis == 0].mean()) if (vis == 0).any() else None,
            **episode_stats(r, conf, visible_min, 0.2),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    res = [analyse(f, a.conf) for f in a.files]
    for r in res:
        print(f"\n== {r['file']} (mode {r['mode']}, conf >= {r['conf']})")
        for cam, c in r['cameras'].items():
            print(f"  {cam}: frames {c['frames']}, ramp fit {c['ramp_fit_lo_hi']}, "
                  f"never recognised {100*c['never_recognised_1frame']:.1f}% (1 frame) / "
                  f"{100*c['never_recognised_3frames']:.1f}% (3 frames), "
                  f"delay {c['delay_1frame_median_s']} / {c['delay_3frames_median_s']} s, "
                  f"P(det | gap<0.5 m) {c['p_detect_gap_lt_0.5m']}, FP {c['false_positive_rate_vis0']}")
            cells = []
            for x in c['curve']:
                pct = '-' if x['p_detect'] is None else f"{100 * x['p_detect']:.0f}%"
                cells.append(f"{x['bin'][0]:.1f}-{x['bin'][1]:.1f}:{pct}({x['frames']})")
            print('     curve: ' + '  '.join(cells))
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
