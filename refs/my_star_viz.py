import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class NPointedStar:
    def __init__(self, n_points=5, period=3, radius=1.0):
        self.n_points = n_points
        self.period = period
        self.radius = radius
        self.reset()

    def reset(self):
        # Compute the vertices of a regular polygon (outer points)
        theta = np.linspace(0, 2 * np.pi, self.n_points, endpoint=False)
        points = np.stack([self.radius * np.cos(theta), self.radius * np.sin(theta)], axis=1)

        # Shift so that the first vertex is at (0, 0)
        shift = points[0]
        points -= shift

        # Mirror (flip) along y-axis -> negate x-coordinates
        points[:, 0] *= -1

        # Define the star traversal order (skip-one connection)
        star_order = [(i * 2) % self.n_points for i in range(self.n_points)]
        self.star_points = points[star_order]

        # Close the loop
        self.star_points = np.vstack([self.star_points, self.star_points[0]])

        # Uniform segment timing
        self.n_segments = len(self.star_points) - 1
        self.T = np.linspace(0.0, self.period, self.n_segments + 1)

    def pos(self, t):
        """Vectorized position output (3 x N)."""
        t = np.atleast_1d(t)
        i = np.searchsorted(self.T, t, side="right") - 1
        i = np.clip(i, 0, self.n_segments - 1)

        pointA = self.star_points[i]
        pointB = self.star_points[i + 1]
        tA, tB = self.T[i], self.T[i + 1]
        s = (t - tA) / (tB - tA)

        x = pointA[:, 0] + (pointB[:, 0] - pointA[:, 0]) * s
        y = pointA[:, 1] + (pointB[:, 1] - pointA[:, 1]) * s
        z = np.zeros_like(x)
        return np.vstack([x, y, z])

    def vel(self, t):
        """Piecewise-constant velocity along each segment."""
        t = np.atleast_1d(t)
        i = np.searchsorted(self.T, t, side="right") - 1
        i = np.clip(i, 0, self.n_segments - 1)

        pointA = self.star_points[i]
        pointB = self.star_points[i + 1]
        dt = (self.T[i + 1] - self.T[i])

        vx = (pointB[:, 0] - pointA[:, 0]) / dt
        vy = (pointB[:, 1] - pointA[:, 1]) / dt
        vz = np.zeros_like(vx)
        return np.vstack([vx, vy, vz])


# === MAIN / demo ===
if __name__ == "__main__":
    ref = NPointedStar(n_points=5, period=6.0, radius=1.0)

    t = np.linspace(0.0, ref.T[-1], 400)
    x = ref.pos(t)
    v = ref.vel(t)

    # --- 3D static trajectory plot ---
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(131, projection="3d")
    ax.plot(x[0], x[1], x[2], "k-", lw=1.5, label="Trajectory")
    ax.scatter(ref.star_points[:, 0], ref.star_points[:, 1],
               np.zeros_like(ref.star_points[:, 0]), c="r", s=40, label="Vertices")
    ax.set_title("3D Star Trajectory (flipped along y-axis)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 0.2])
    ax.legend()

    # --- Position profiles ---
    ax2 = fig.add_subplot(132)
    ax2.plot(t, x[0], label="x(t)")
    ax2.plot(t, x[1], label="y(t)")
    ax2.plot(t, x[2], label="z(t)")
    ax2.set_xlabel("time [s]"); ax2.set_ylabel("position")
    ax2.set_title("Position Profiles")
    ax2.legend(); ax2.grid(True)

    # --- Velocity profiles ---
    ax3 = fig.add_subplot(133)
    ax3.plot(t, v[0], label="vx(t)")
    ax3.plot(t, v[1], label="vy(t)")
    ax3.plot(t, v[2], label="vz(t)")
    ax3.set_xlabel("time [s]"); ax3.set_ylabel("velocity")
    ax3.set_title("Velocity Profiles")
    ax3.legend(); ax3.grid(True)

    plt.tight_layout()
    plt.show()

    # --- Animation ---
    fig2 = plt.figure(figsize=(6, 6))
    ax = fig2.add_subplot(111, projection="3d")
    ax.set_xlim(np.min(ref.star_points[:, 0]) - 0.2, np.max(ref.star_points[:, 0]) + 0.2)
    ax.set_ylim(np.min(ref.star_points[:, 1]) - 0.2, np.max(ref.star_points[:, 1]) + 0.2)
    ax.set_zlim(-0.2, 0.2)
    ax.plot(x[0], x[1], x[2], color="gray", lw=1, alpha=0.5)
    ax.scatter(ref.star_points[:, 0], ref.star_points[:, 1],
               np.zeros(ref.star_points.shape[0]), c="r", s=40)
    point, = ax.plot([], [], [], "bo", markersize=8)

    def init():
        point.set_data([], [])
        point.set_3d_properties([])
        return point,

    def update(frame):
        point.set_data([x[0, frame]], [x[1, frame]])
        point.set_3d_properties([x[2, frame]])
        return point,

    ani = FuncAnimation(fig2, update, frames=len(t), init_func=init, blit=True, interval=30, repeat=True)
    plt.show()
