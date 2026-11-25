"""
Smoke test for headless Gazebo rendering: place robot and worker, grab one
frame from each camera, save PNGs.

Run (server, ROS env sourced, gz sim running corner_world.sdf):
  python3 snapshot.py --out /root/sci1/gz_out
"""

import argparse
import math
import os
import threading
import time

import numpy as np
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.image_pb2 import Image
from gz.msgs10.pose_pb2 import Pose
from gz.transport13 import Node

WORLD = 'corner'


class Grabber:
    def __init__(self, node, topic):
        self.lock = threading.Lock()
        self.frame, self.stamp = None, 0
        node.subscribe(Image, topic, self._cb)

    def _cb(self, msg):
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        with self.lock:
            self.frame, self.stamp = img.copy(), time.time()

    def next_after(self, t, timeout=3.0):
        end = time.time() + timeout
        while time.time() < end:
            with self.lock:
                if self.frame is not None and self.stamp > t:
                    return self.frame
            time.sleep(0.01)
        return None


def grab_once(node, topic, settle=0.12, timeout=3.0):
    """Subscribe, take the first frame produced after `settle` seconds, unsubscribe.

    Keeping image subscriptions off while service requests run avoids a GIL
    deadlock between blocking requests and Python subscription callbacks.
    """
    box = {}
    t0 = time.time() + settle

    def cb(msg):
        if 'img' not in box and time.time() > t0:
            box['img'] = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3).copy()

    node.subscribe(Image, topic, cb)
    end = time.time() + timeout
    while 'img' not in box and time.time() < end:
        time.sleep(0.005)
    node.unsubscribe(topic)
    return box.get('img')


def set_pose(node, name, x, y, z, yaw):
    req = Pose()
    req.name = name
    req.position.x, req.position.y, req.position.z = x, y, z
    req.orientation.z, req.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    ok, resp = node.request(f'/world/{WORLD}/set_pose', req, Pose, Boolean, 2000)
    return ok and resp.data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/root/sci1/gz_out')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    def save(path, img):
        try:
            import cv2
            cv2.imwrite(path, img[:, :, ::-1])
        except ImportError:   # plain PPM fallback
            path = path[:-4] + '.ppm'
            with open(path, 'wb') as f:
                f.write(b'P6 %d %d 255\n' % (img.shape[1], img.shape[0]) + img.tobytes())
        return path
    node = Node()
    cams = {t: Grabber(node, t) for t in ('cam87', 'cam120')}
    time.sleep(1.0)
    # Robot on the horizontal leg heading +x, worker just round the corner.
    for k, (rx, px, py) in enumerate([(-2.5, 0.0, 1.5), (-1.2, 0.0, 0.8), (-0.6, -0.1, 0.3)]):
        print('robot', set_pose(node, 'robot', rx, 0.23, 0.0, 0.0),
              'worker', set_pose(node, 'worker', px, py, 0.0, -math.pi / 2))
        t = time.time()
        for name, g in cams.items():
            img = g.next_after(t + 0.25)
            if img is None:
                print('no frame from', name)
                continue
            path = save(os.path.join(a.out, f'snap{k}_{name}.png'), img)
            print('wrote', path, img.shape)


if __name__ == '__main__':
    main()
