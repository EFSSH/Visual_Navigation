"""
2D animation of SCI-1 corner encounters: true worker position vs what the
onboard camera log holds, camera field of view, and the true vs logged gap.

Writes MP4 when an ffmpeg binary is available (imageio-ffmpeg), else GIF.

Run: PYTHONPATH=learning python -m sci1.animate_corner --out /home/xlz/sci1_runs/videos
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.patches import Circle, Rectangle, Wedge  # noqa: E402

from .corner_pass import CAMERA_CONFIGS, CornerScenario, simulate  # noqa: E402
from .plots_v7 import CRITICAL, INK, INK2, MUTED, SERIES, SURFACE, style  # noqa: E402

ROBOT_C, PERSON_C, LOG_C = SERIES[0], SERIES[1], SERIES[2]
WALL_C = '#d9d7cf'
VIEW = (-4.0, 1.5, -1.5, 4.0)

CLIPS = [
    # name, camera, mode, perception overrides, selector, title
    ('87deg_go_missed', 'front_87_h0.5', 'go', {'extrapolate': True}, 'missed',
     '87° 前视、go 1.0 m/s：真实发生险情，机载日志没有记下'),
    ('87deg_slow_safe', 'front_87_h0.5', 'slow', {'extrapolate': True}, 'safe',
     '87° 前视、slow 0.5 m/s：提前减速，安全擦肩'),
    ('120deg_go_logged', 'front_120_h0.5', 'go', {'extrapolate': True}, 'logged',
     '120°×90° 前视、go 1.0 m/s：险情被机载日志记下'),
]


def pick(out, selector):
    near, near_log = out['near_moving'], out['near_moving_logged']
    if selector == 'missed':
        cand = np.flatnonzero(near & ~near_log)
    elif selector == 'logged':
        cand = np.flatnonzero(near & near_log)
    else:
        cand = np.flatnonzero(out['encounter'] & ~near & (out['gap_true'] < 0.35))
    if not len(cand):
        raise RuntimeError(f'no episode for {selector}')
    return int(cand[0])


def draw_walls(ax, scn):
    for (x0, x1), (y0, y1) in scn.occluders:
        x0, x1 = max(x0, VIEW[0] - 1), min(x1, VIEW[1] + 1)
        y0, y1 = max(y0, VIEW[2] - 1), min(y1, VIEW[3] + 1)
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=WALL_C, zorder=0))


def render(scn, out, i, title, path, fps=10, stride=2):
    gap, logged = out['gap'][i], out['logged'][i]
    steps = np.flatnonzero(np.linalg.norm(out['robot'][i] - out['person'][i], axis=-1) < 4.0)
    k0, k1 = max(0, steps[0] - 10), min(len(gap) - 1, steps[-1] + 10)
    frames = list(range(k0, k1 + 1, stride))
    t = np.arange(len(gap)) * scn.dt

    fig = plt.figure(figsize=(9.6, 4.6))
    ax = fig.add_axes([0.03, 0.08, 0.45, 0.8])
    bx = fig.add_axes([0.58, 0.55, 0.39, 0.33])
    cx = fig.add_axes([0.58, 0.12, 0.39, 0.28])
    fig.suptitle(title, color=INK, fontsize=11)
    ax.set_xlim(VIEW[0], VIEW[1]); ax.set_ylim(VIEW[2], VIEW[3]); ax.set_aspect('equal')
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    draw_walls(ax, scn)
    robot = Circle((0, 0), scn.robot_radius, color=ROBOT_C, zorder=3)
    person = Circle((0, 0), 0.25, color=PERSON_C, zorder=3)
    ghost = Circle((0, 0), 0.25, fill=False, ls='--', lw=1.5, ec=LOG_C, zorder=4)
    ax.add_patch(robot); ax.add_patch(person); ax.add_patch(ghost)
    wedges = []
    for _ in scn.cameras:
        w = Wedge((0, 0), 3.0, 0, 1, color=ROBOT_C, alpha=0.12, zorder=1)
        ax.add_patch(w); wedges.append(w)
    trail, = ax.plot([], [], color=ROBOT_C, lw=1, alpha=0.5)
    ptrail, = ax.plot([], [], color=PERSON_C, lw=1, alpha=0.5)
    status = ax.text(0.02, 0.97, '', transform=ax.transAxes, va='top', fontsize=9, color=INK2)
    ax.legend(handles=[Circle((0, 0), 1, color=ROBOT_C, label='机器人（扇形为相机视场）'),
                       Circle((0, 0), 1, color=PERSON_C, label='工人真实位置'),
                       Circle((0, 0), 1, fill=False, ls='--', ec=LOG_C, label='机载日志里的工人')],
              loc='lower left', fontsize=7.5)

    lo = max(0, int(k0)); hi = int(k1)
    true_finite = np.minimum(gap, 3.0)
    log_gap = np.where(logged, gap, np.nan)
    bx.plot(t[lo:hi], true_finite[lo:hi], color=PERSON_C, lw=1.2, alpha=0.35)
    bx.axhline(0.10, color=CRITICAL, lw=1, ls='--')
    bx.text(t[lo], 0.13, '险情线 10 cm', color=CRITICAL, fontsize=7)
    tg, = bx.plot([], [], color=PERSON_C, lw=2, label='真实间隙')
    lg, = bx.plot([], [], color=LOG_C, lw=0, marker='o', ms=3, label='机载日志里的间隙')
    bx.set_xlim(t[lo], t[hi]); bx.set_ylim(-0.05, 1.5)
    bx.set_ylabel('间隙（m）'); bx.legend(loc='upper right', fontsize=7.5)
    sp, = cx.plot([], [], color=ROBOT_C, lw=2)
    cx.set_xlim(t[lo], t[hi]); cx.set_ylim(0, 1.1)
    cx.set_ylabel('机器人速度（m/s）'); cx.set_xlabel('时间（s）')
    flag = bx.text(0.02, 0.92, '', transform=bx.transAxes, fontsize=9, color=CRITICAL,
                   va='top')

    def update(k):
        rp, pp, hd = out['robot'][i, k], out['person'][i, k], out['heading'][i, k]
        robot.center = rp; person.center = pp; person.set_radius(out['radius'][i, k])
        seen = logged[k]
        ghost.set_visible(bool(seen)); ghost.center = pp
        for w, (ch, hfov, _, _) in zip(wedges, scn.cameras):
            c = np.degrees(hd) + ch
            w.set_center(rp); w.set_theta1(c - hfov / 2); w.set_theta2(c + hfov / 2)
        trail.set_data(out['robot'][i, lo:k + 1, 0], out['robot'][i, lo:k + 1, 1])
        ptrail.set_data(out['person'][i, lo:k + 1, 0], out['person'][i, lo:k + 1, 1])
        tg.set_data(t[lo:k + 1], true_finite[lo:k + 1])
        lg.set_data(t[lo:k + 1], log_gap[lo:k + 1])
        sp.set_data(t[lo:k + 1], out['speed'][i, lo:k + 1])
        hazard = (gap[lo:k + 1] <= 0.10) & (out['speed'][i, lo:k + 1] > 0.05)
        brake = '刹车中' if out['braking'][i, k] else '行驶'
        status.set_text(f't = {t[k]:.1f} s   {brake}   日志：{"看到工人" if seen else "没有工人"}')
        if hazard.any():
            logged_hazard = (hazard & logged[lo:k + 1]).any()
            flag.set_text('险情！' + ('日志记下了' if logged_hazard else '日志没有记下'))
            person.set_color(CRITICAL)
        else:
            flag.set_text('')
            person.set_color(PERSON_C)
        return robot, person, ghost, trail, ptrail, tg, lg, sp, status, flag, *wedges

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    try:
        import imageio_ffmpeg
        plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
        anim.save(path + '.mp4', writer=animation.FFMpegWriter(fps=fps, bitrate=1800), dpi=120)
        written = path + '.mp4'
    except ImportError:
        anim.save(path + '.gif', writer=animation.PillowWriter(fps=fps), dpi=90)
        written = path + '.gif'
    plt.close(fig)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/home/xlz/sci1_runs/videos')
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    style()
    for name, cam, mode, over, sel, title in CLIPS:
        scn = CornerScenario(cameras=CAMERA_CONFIGS[cam], **over)
        out = simulate(scn, mode, 300, a.seed, trajectories=True)
        i = pick(out, sel)
        print('wrote', render(scn, out, i, title, os.path.join(a.out, name)))


if __name__ == '__main__':
    main()
