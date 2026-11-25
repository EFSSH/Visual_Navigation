"""
Replay 2D corner encounters in Gazebo and run YOLO on each rendered frame.

For every sampled frame (5 Hz) of each episode from sci1.corner_pass:
place robot and worker with set_pose, grab the next image from each camera,
run YOLO, and record the detection together with the 2D model's geometric
visible fraction for that camera. The output calibrates the 2D detection
model (visible fraction -> recognition probability) and gives episode-level
never-recognised rates and recognition delays with a real detector.

Run on the server (ROS env sourced, GZ_IP=127.0.0.1, gz sim running the corner world):
  python replay_detect.py --episodes 300 --mode go --out /root/sci1/replay_go.npz
"""

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sci1.corner_pass import CAMERA_CONFIGS, CornerScenario, simulate  # noqa: E402
from sci1.geometry import line_of_sight  # noqa: E402
from sci1.sensors import body_points  # noqa: E402

from snapshot import grab_once, set_pose  # noqa: E402
from gz.transport13 import Node  # noqa: E402

CAMS = {'cam87': CAMERA_CONFIGS['front_87_h0.5'][0],
        'cam120': CAMERA_CONFIGS['front_120_h0.5'][0]}
PERSON_YAW_OFFSET = math.pi / 2   # model faces local +y (checked: front view when walking at the robot)


def visible_fraction(scn, robot_p, heading, person_p, half_width, cam):
    """Geometric visible body fraction for one camera (same model as corner_pass)."""
    _, hfov, vfov, height = cam
    pts = body_points(robot_p[None], person_p[None], 1.0)
    centre = person_p[None, None, :]
    pts = centre + (pts - centre) * half_width
    seen = line_of_sight(robot_p[None, None, :], pts, scn.occluders, scn.sensor_range)
    rel = pts - robot_p[None, None, :]
    bearing = np.arctan2(rel[..., 1], rel[..., 0])
    ang = (bearing - heading + np.pi) % (2 * np.pi) - np.pi
    in_fov = np.abs(ang) <= np.deg2rad(hfov) / 2
    dist_pts = np.linalg.norm(rel, axis=-1)
    f_h = float((seen & in_fov & (dist_pts <= scn.cam_max_range)).mean())
    dist = float(np.linalg.norm(person_p - robot_p))
    reach = dist * math.tan(math.radians(vfov) / 2)
    f_v = max(0.0, min(scn.person_height, height + reach) - max(0.0, height - reach)) / scn.person_height
    return f_h * f_v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=300)
    ap.add_argument('--mode', default='go')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--weights', default='/root/yolov8s.pt')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--save-images', type=int, default=0, help='save frames of the first N episodes')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    from ultralytics import YOLO
    import cv2
    model = YOLO(a.weights)
    scn = CornerScenario(cameras=CAMERA_CONFIGS['front_87_h0.5'], extrapolate=True)
    out = simulate(scn, a.mode, a.episodes, a.seed, trajectories=True)
    period = int(round(1.0 / (scn.rate_hz * scn.dt)))

    node = Node()
    time.sleep(0.5)
    rows = []
    img_dir = os.path.splitext(a.out)[0] + '_frames'
    t0 = time.time()
    for i in range(a.episodes):
        if not out['encounter'][i]:
            continue
        dist = np.linalg.norm(out['robot'][i] - out['person'][i], axis=-1)
        ks = [k for k in range(0, len(dist), period) if dist[k] < 5.0]
        for k in ks:
            rp, pp, hd = out['robot'][i, k], out['person'][i, k], out['heading'][i, k]
            k2 = min(k + 1, len(dist) - 1)
            step = out['person'][i, k2] - out['person'][i, max(k - 1, 0)]
            pyaw = math.atan2(step[1], step[0]) + PERSON_YAW_OFFSET
            set_pose(node, 'robot', float(rp[0]), float(rp[1]), 0.0, float(hd))
            set_pose(node, 'worker', float(pp[0]), float(pp[1]), 0.0, pyaw)
            for name, cam in CAMS.items():
                img = grab_once(node, name)
                if img is None:
                    continue
                res = model(img[:, :, ::-1], verbose=False, device=0, conf=a.conf, classes=[0])[0]
                conf = float(res.boxes.conf.max()) if len(res.boxes) else 0.0
                vis = visible_fraction(scn, rp, hd, pp, out['radius'][i, k], cam)
                rows.append((i, k, k * scn.dt, CAMS[name][1], vis, float(out['gap'][i, k]),
                             float(out['speed'][i, k]), conf))
                if i < a.save_images:
                    os.makedirs(img_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(img_dir, f'ep{i:03d}_k{k:03d}_{name}.jpg'),
                                res.plot())
        if i % 20 == 0:
            print(f'episode {i}/{a.episodes}  frames {len(rows)}  {time.time()-t0:.0f}s', flush=True)
    arr = np.array(rows, dtype=float)
    np.savez(a.out, rows=arr, columns=np.array(['episode', 'step', 't', 'hfov', 'visible',
                                                 'gap', 'speed', 'conf']),
             mode=a.mode, seed=a.seed, near_moving=out['near_moving'], encounter=out['encounter'])
    print('saved', a.out, arr.shape)


if __name__ == '__main__':
    main()
