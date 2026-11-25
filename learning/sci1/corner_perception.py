"""
SCI-1 perception sensitivity: does 'late detection' stay usable as an onboard
monitoring signal when the detector is slower and less reliable than the
ideal one?

Varied: recognition latency, N-frame track confirmation, a stricter
visible-fraction threshold for recognising a partly visible person, and
detection loss while turning (motion blur / shake).

Per encounter:
- never: the person is never recognised (invisible onboard);
- late / late_log: true late-detection rate vs what the robot itself knows;
- miss: share of late events missing from the onboard record;
- near / near_log: moving close calls (gap <= 10 cm), true vs onboard;
- cover: P(onboard late flag | true moving close call);
- contact: moving contact rate (consequence of the slower detector).

Run: PYTHONPATH=learning python -m sci1.corner_perception --n 10000
"""

import argparse
import json

from .corner_pass import CAMERA_CONFIGS, CornerScenario, simulate

VARIANTS = {
    'ideal': {},
    'latency 0.2s': {'recog_latency': 0.2},
    'latency 0.4s': {'recog_latency': 0.4},
    'confirm 3 frames': {'confirm_frames': 3},
    'partial 40-80%': {'cam_area_lo': 0.4, 'cam_area_hi': 0.8},
    'turn blur 0.5': {'turn_blur': 0.5},
    'realistic': {'recog_latency': 0.3, 'confirm_frames': 3,
                  'cam_area_lo': 0.4, 'cam_area_hi': 0.8, 'turn_blur': 0.5},
    'latency 0.2s + extrap': {'recog_latency': 0.2, 'extrapolate': True},
    'latency 0.4s + extrap': {'recog_latency': 0.4, 'extrapolate': True},
    'ideal + extrap': {'extrapolate': True},
    'realistic + extrap': {'recog_latency': 0.3, 'confirm_frames': 3, 'cam_area_lo': 0.4,
                           'cam_area_hi': 0.8, 'turn_blur': 0.5, 'extrapolate': True},
    'harsh': {'recog_latency': 0.5, 'confirm_frames': 3,
              'cam_area_lo': 0.5, 'cam_area_hi': 0.9, 'turn_blur': 0.7},
}
CONFIGS = ('front_87_h0.5', 'front_120_h0.5', 'ring4_87_h0.5')
ACTIONS = ('go', 'medium', 'slow')


def measure(out):
    enc = out['encounter']
    m = lambda key: float(out[key][enc].mean())
    late, late_log = m('late'), m('late_logged')
    near = out['near_moving'] & enc
    return {
        'never': m('never_recognised'), 'late': late, 'late_log': late_log,
        'miss': 1 - late_log / late if late else float('nan'),
        'near': m('near_moving'), 'near_log': m('near_moving_logged'),
        'cover': float(out['late_logged'][near].mean()) if near.any() else float('nan'),
        'contact': m('contact_mv'),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--attentive-share', type=float, default=0.5)
    ap.add_argument('--variants', nargs='*', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    rows = []
    cols = ('never', 'late', 'late_log', 'miss', 'near', 'near_log', 'cover', 'contact')
    print(f'attentive share {a.attentive_share}, n {a.n} per row; rates per encounter')
    print(f'{"variant":>22} {"camera":>15} {"act":>6} ' + ' '.join(f'{c:>8}' for c in cols))
    for vname, over in VARIANTS.items():
        if a.variants and vname not in a.variants:
            continue
        for cam in CONFIGS:
            scn = CornerScenario(cameras=CAMERA_CONFIGS[cam],
                                 attentive_share=a.attentive_share, **over)
            for j, act in enumerate(ACTIONS):
                r = {'variant': vname, 'camera': cam, 'action': act,
                     **measure(simulate(scn, act, a.n, a.seed + 1000 * j))}
                rows.append(r)
                print(f'{vname:>22} {cam:>15} {act:>6} '
                      + ' '.join(f'{100 * r[c]:7.1f}%' for c in cols))
    if a.out:
        with open(a.out, 'w') as f:
            json.dump({'attentive_share': a.attentive_share, 'variants': VARIANTS,
                       'rows': rows}, f, indent=1)


if __name__ == '__main__':
    main()
