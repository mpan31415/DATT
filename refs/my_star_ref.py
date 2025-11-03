import numpy as np
from DATT.quadsim.flatref import StaticRef
from DATT.refs.base_ref import BaseRef

class MyStarRef(BaseRef):
    def __init__(self, n_points=5, radius=1.0, period=2*np.pi, **kwargs):
        offset_pos = kwargs.get('offset_pos', np.zeros(3))
        super().__init__(offset_pos)
        self.n_points = n_points
        self.radius = radius
        self.period = period

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
        # t = np.atleast_1d(t)
        t = t % self.period
        i = np.searchsorted(self.T, t, side="right") - 1
        i = np.clip(i, 0, self.n_segments - 1)

        pointA = self.star_points[i]
        pointB = self.star_points[i + 1]
        tA, tB = self.T[i], self.T[i + 1]
        s = (t - tA) / (tB - tA)

        if isinstance(t, np.ndarray):
            x = pointA[:, 0] + (pointB[:, 0] - pointA[:, 0]) * s
            y = pointA[:, 1] + (pointB[:, 1] - pointA[:, 1]) * s
        else:
            x = pointA[0] + (pointB[0] - pointA[0]) * s
            y = pointA[1] + (pointB[1] - pointA[1]) * s
        z = np.zeros_like(x)

        return np.array([x, y, z])

    def vel(self, t):
        t = np.atleast_1d(t)
        t = t % self.period
        i = np.searchsorted(self.T, t, side="right") - 1
        i = np.clip(i, 0, self.n_segments - 1)

        pointA = self.star_points[i]
        pointB = self.star_points[i + 1]
        dt = (self.T[i + 1] - self.T[i])

        if isinstance(t, np.ndarray):
            vx = (pointB[:, 0] - pointA[:, 0]) / dt
            vy = (pointB[:, 1] - pointA[:, 1]) / dt
        else:
            vx = (pointB[0] - pointA[0]) / dt
            vy = (pointB[1] - pointA[1]) / dt
        vz = np.zeros_like(vx)

        return np.array([vx, vy, vz])

    def acc(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])

    def jerk(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])

    def snap(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])

    def yaw(self, t):
        return 0

if __name__ == '__main__':

    ref = MyStarRef(n_points=5, radius=1.0, period=6.0)

    t = np.linspace(0, 7, 500)

    import matplotlib.pyplot as plt
    # plt.subplot(2, 1, 1)
    # plt.plot(t, ref.pos(t)[0, :], label='x')
    # plt.subplot(2, 1, 2)
    # plt.plot(t, ref.pos(t)[1, :], label='y')
    # plt.show()

    # plot 3d plot
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ref.pos(t)[0, :], ref.pos(t)[1, :], ref.pos(t)[2, :], label='Star Path')
    # show start and end points
    ax.scatter(ref.pos(0)[0], ref.pos(0)[1], ref.pos(0)[2], color='red', label='Start')
    ax.scatter(ref.pos(t[-1])[0], ref.pos(t[-1])[1], ref.pos(t[-1])[2], color='green', label='End')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('3D Star Reference Path')
    plt.legend()
    plt.show()