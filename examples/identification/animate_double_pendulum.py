"""
Animate a double pendulum trajectory.

Usage:
    python animate_double_pendulum.py                          # run default demo
    python animate_double_pendulum.py trajectory.npy           # load saved trajectory
    python animate_double_pendulum.py trajectory.npy --save    # save as gif

Trajectory format: numpy array of shape (N, 4) with columns
    [theta1, theta2, dtheta1, dtheta2]
where theta = 0 means hanging straight down.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import os


def forward_kinematics(theta1, theta2, l1=1.0, l2=1.0):
    """
    Compute (x, y) positions of the two pendulum bobs.
    theta = 0  =>  hanging straight down  =>  (0, -l)
    """
    x1 =  l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)
    return x1, y1, x2, y2


def animate_double_pendulum(trajectory, l1=1.0, l2=1.0, dt=0.001,
                             skip=20, trail_length=200,
                             save_path=None, fps=30):
    """
    Animate a double-pendulum trajectory.

    Parameters
    ----------
    trajectory : ndarray, shape (N, 4)
        State history [theta1, theta2, dtheta1, dtheta2].
    l1, l2 : float
        Link lengths (must match the simulation).
    dt : float
        Integration time-step used in the simulation.
    skip : int
        Show every `skip`-th frame (speeds up the animation).
    trail_length : int
        Number of past tip positions to draw as a fading trail.
    save_path : str or None
        If given, save the animation to this file (e.g. "pendulum.gif").
        Supports .gif (uses Pillow, no ffmpeg needed) and .mp4 (needs ffmpeg).
    fps : int
        Frames per second for the saved file.
    """
    theta1 = trajectory[:, 0]
    theta2 = trajectory[:, 1]

    x1, y1, x2, y2 = forward_kinematics(theta1, theta2, l1, l2)

    # down-sample
    idx = np.arange(0, len(theta1), skip)
    x1, y1, x2, y2 = x1[idx], y1[idx], x2[idx], y2[idx]

    # figure setup
    lim = (l1 + l2) * 1.2
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("Double Pendulum")

    # artists
    line, = ax.plot([], [], "o-", color="royalblue", lw=2,
                    markersize=8, markerfacecolor="navy")
    trail, = ax.plot([], [], "-", color="tomato", lw=0.8, alpha=0.6)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                        fontsize=10, verticalalignment="top")

    trail_x, trail_y = [], []

    def init():
        line.set_data([], [])
        trail.set_data([], [])
        time_text.set_text("")
        return line, trail, time_text

    def update(frame):
        # pendulum links: pivot -> bob1 -> bob2
        xs = [0.0, x1[frame], x2[frame]]
        ys = [0.0, y1[frame], y2[frame]]
        line.set_data(xs, ys)

        # tip trail
        trail_x.append(x2[frame])
        trail_y.append(y2[frame])
        start = max(0, len(trail_x) - trail_length)
        trail.set_data(trail_x[start:], trail_y[start:])

        t = idx[frame] * dt
        time_text.set_text(f"t = {t:.2f} s")
        return line, trail, time_text

    anim = animation.FuncAnimation(
        fig, update, frames=len(idx),
        init_func=init, blit=True,
        interval=max(1, int(skip * dt * 1000))  # real-time-ish
    )

    if save_path is not None:
        ext = os.path.splitext(save_path)[1].lower()
        if ext == ".gif":
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=150)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()

    return anim


# --------------- demo: generate a trajectory if none is provided ---------------

def _generate_demo_trajectory(num_steps=10000, dt=0.001):
    """Quick standalone double-pendulum simulation (no external deps)."""
    m1, m2 = 1.0, 1.0
    l1, l2 = 1.0, 1.0
    g = 9.81
    b1, b2 = 0.01, 0.01

    state = np.array([0.7, 0.2, 0.0, 0.0])
    trajectory = []

    for _ in range(num_steps):
        theta1, theta2, dtheta1, dtheta2 = state
        delta = theta1 - theta2
        cos_d, sin_d = np.cos(delta), np.sin(delta)

        m11 = (m1 + m2) * l1**2
        m12 = m2 * l1 * l2 * cos_d
        m22 = m2 * l2**2

        f1 = (-(m1 + m2) * g * l1 * np.sin(theta1)
              - m2 * l1 * l2 * dtheta2**2 * sin_d - b1 * dtheta1)
        f2 = (-m2 * g * l2 * np.sin(theta2)
              + m2 * l1 * l2 * dtheta1**2 * sin_d - b2 * dtheta2)

        det = m11 * m22 - m12**2
        ddtheta1 = (m22 * f1 - m12 * f2) / det
        ddtheta2 = (-m12 * f1 + m11 * f2) / det

        state[0] += dtheta1 * dt
        state[1] += dtheta2 * dt
        state[2] += ddtheta1 * dt
        state[3] += ddtheta2 * dt

        trajectory.append(state.copy())

    return np.array(trajectory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate a double pendulum trajectory.")
    parser.add_argument("trajectory_file", nargs="?", default=None,
                        help="Path to a .npy file with shape (N,4).")
    parser.add_argument("--save", action="store_true",
                        help="Save animation as mp4 instead of displaying.")
    parser.add_argument("--skip", type=int, default=20,
                        help="Show every N-th frame (default: 20).")
    parser.add_argument("--dt", type=float, default=0.001,
                        help="Simulation time step (default: 0.001).")
    parser.add_argument("--l1", type=float, default=1.0, help="Length of link 1.")
    parser.add_argument("--l2", type=float, default=1.0, help="Length of link 2.")
    args = parser.parse_args()

    if args.trajectory_file is not None:
        traj = np.load(args.trajectory_file)
        print(f"Loaded trajectory: {traj.shape}")
    else:
        print("No trajectory file given — generating demo trajectory …")
        traj = _generate_demo_trajectory()

    save_path = "double_pendulum.mp4" if args.save else None
    animate_double_pendulum(traj, l1=args.l1, l2=args.l2, dt=args.dt,
                            skip=args.skip, save_path=save_path)
