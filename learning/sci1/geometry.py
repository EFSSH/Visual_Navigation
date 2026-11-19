"""Line-of-sight checks against axis-aligned box occluders."""

import numpy as np


def segment_hits_box(p, q, box):
    """
    Whether segments p->q intersect an axis-aligned box (slab method).

    ``p`` and ``q`` broadcast to (..., 2); ``box`` is
    ``((x_min, x_max), (y_min, y_max))``. Touching the boundary counts as a
    hit, so a sight line grazing a wall corner is treated as blocked.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p, q = np.broadcast_arrays(p, q)
    d = q - p
    t_lo = np.zeros(p.shape[:-1])
    t_hi = np.ones(p.shape[:-1])
    inside = np.ones(p.shape[:-1], dtype=bool)
    for axis, (lo, hi) in enumerate(box):
        pa, da = p[..., axis], d[..., axis]
        flat = np.abs(da) < 1e-12
        inside &= ~flat | ((pa >= lo) & (pa <= hi))
        with np.errstate(divide='ignore', invalid='ignore'):
            t1 = (lo - pa) / da
            t2 = (hi - pa) / da
        near = np.where(flat, -np.inf, np.minimum(t1, t2))
        far = np.where(flat, np.inf, np.maximum(t1, t2))
        t_lo = np.maximum(t_lo, near)
        t_hi = np.minimum(t_hi, far)
    return inside & (t_lo <= t_hi)


def line_of_sight(p, q, boxes, max_range):
    """True where q is visible from p: no box blocks the segment and |pq| <= range."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    visible = np.linalg.norm(q - p, axis=-1) <= max_range
    for box in boxes:
        visible &= ~segment_hits_box(p, q, box)
    return visible
