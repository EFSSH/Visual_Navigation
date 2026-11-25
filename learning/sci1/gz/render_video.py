"""
Render typical corner encounters as side-by-side MP4: robot camera with YOLO
boxes (left) and an overhead view (right), with gap / speed / log status.

Uses the video world (corner_world_video.sdf) in its own GZ_PARTITION so it
can run next to the calibration batch.

  GZ_PARTITION=video GZ_IP=127.0.0.1 python render_video.py --out /root/sci1/videos
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sci1.corner_pass import CAMERA_CONFIGS, CornerScenario, simulate  # noqa: E402

from replay_detect import PERSON_YAW_OFFSET  # noqa: E402
from snapshot import grab_once, set_pose  # noqa: E402
from gz.transport13 import Node  # noqa: E402

CLIPS = [
    ('gz_87_go_closecall', 'cam87', 'front_87_h0.5', 'go', 'missed',
     '87 deg camera, go 1.0 m/s: close call at the corner'),
    ('gz_87_slow_safe', 'cam87', 'front_87_h0.5', 'slow', 'safe',
     '87 deg camera, slow 0.5 m/s: safe pass'),
    ('gz_120_go_logged', 'cam120', 'front_120_h0.5', 'go', 'logged',
     '120x90 deg camera, go 1.0 m/s: close call at the corner'),
]


def pick(out, selector):
    near, log = out['near_moving'], out['near_moving_logged']
    if selector == 'missed':
        cand = np.flatnonzero(near & ~log)
    elif selector == 'logged':
        cand = np.flatnonzero(near & log)
    else:
        cand = np.flatnonzero(out['encounter'] & ~near & (out['gap_true'] < 0.35))
    return int(cand[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/root/sci1/videos')
    ap.add_argument('--weights', default='/root/yolov8s.pt')
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--stride', type=int, default=1, help='render every n-th 20 Hz step')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import cv2
    import imageio_ffmpeg
    from ultralytics import YOLO
    model = YOLO(a.weights)
    node = Node()
    for name, topic, camcfg, mode, sel, title in CLIPS:
        scn = CornerScenario(cameras=CAMERA_CONFIGS[camcfg], extrapolate=True)
        out = simulate(scn, mode, 300, a.seed, trajectories=True)
        i = pick(out, sel)
        dist = np.linalg.norm(out['robot'][i] - out['person'][i], axis=-1)
        near = np.flatnonzero(dist < 4.0)
        ks = range(max(0, near[0] - 10), min(len(dist), near[-1] + 15), a.stride)
        H = 540
        writer = None
        hazard_seen = False
        for k in ks:
            rp, pp, hd = out['robot'][i, k], out['person'][i, k], out['heading'][i, k]
            step = out['person'][i, min(k + 1, len(dist) - 1)] - out['person'][i, max(k - 1, 0)]
            set_pose(node, 'robot', float(rp[0]), float(rp[1]), 0.0, float(hd))
            set_pose(node, 'worker', float(pp[0]), float(pp[1]), 0.0,
                     math.atan2(step[1], step[0]) + PERSON_YAW_OFFSET)
            cam = grab_once(node, topic)
            top = grab_once(node, 'cam_top')
            if cam is None or top is None:
                continue
            res = model(cam[:, :, ::-1], verbose=False, device=0, conf=0.25, classes=[0])[0]
            left = res.plot()                                         # BGR with boxes
            left = cv2.resize(left, (int(left.shape[1] * H / left.shape[0]), H))
            right = cv2.resize(top[:, :, ::-1], (H, H))
            frame = np.hstack([left, right])
            gap, speed, logged = out['gap'][i, k], out['speed'][i, k], out['logged'][i, k]
            hazard = gap <= 0.10 and speed > 0.05
            hazard_seen |= hazard
            bar = np.full((70, frame.shape[1], 3), 245, np.uint8)
            cv2.putText(bar, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
            status = (f't={k*scn.dt:4.1f}s  gap={max(gap,0):.2f} m  speed={speed:.2f} m/s  '
                      f'YOLO={"person" if len(res.boxes) else "none"}  '
                      f'2D-model log={"sees worker" if logged else "no worker"}')
            cv2.putText(bar, status, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1)
            if hazard_seen:
                cv2.putText(bar, 'CLOSE CALL (gap <= 10 cm)', (frame.shape[1] - 330, 56),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 210), 2)
            frame = np.vstack([bar, frame])
            ph, pw = (-frame.shape[0]) % 16, (-frame.shape[1]) % 16
            frame = np.pad(frame, ((0, ph), (0, pw), (0, 0)), constant_values=245)
            if writer is None:
                writer = imageio_ffmpeg.write_frames(
                    os.path.join(a.out, name + '.mp4'), (frame.shape[1], frame.shape[0]),
                    fps=20 // a.stride, quality=7)
                writer.send(None)
            writer.send(np.ascontiguousarray(frame[:, :, ::-1]))
        if writer is not None:
            writer.close()
            print('wrote', os.path.join(a.out, name + '.mp4'), 'episode', i, flush=True)


if __name__ == '__main__':
    main()
