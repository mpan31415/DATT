import numpy as np
from enum import Enum


class HoveringTargets(Enum):
    CORNERS_1m = 0
    GRID_2x2x2_1m = 1
    GRID_3x3x3_1m = 2
    GRID_4x4x4_1m = 3
    RANDOM_50_1m = 4



def generate_hovering_targets(targets_type, offset=[0.0, 0.0, 0.0]):
    
    if targets_type == HoveringTargets.CORNERS_1m:
        # 4 corners of a square
        target_poses = [
            [0.5, 0.5, 1.5],
            [0.5, -0.5, 1.5],
            [-0.5, -0.5, 1.5],
            [-0.5, 0.5, 1.5],
        ]

    elif targets_type == HoveringTargets.GRID_2x2x2_1m:
        # 2x2x2 grid
        x_vals = [-0.5, 0.5]
        y_vals = [-0.5, 0.5]
        z_vals = [0.5, 1.5]
        poses_grid = np.array([[x, y, z] for x in x_vals for y in y_vals for z in z_vals])
        target_poses = poses_grid[np.lexsort((poses_grid[:, 0], poses_grid[:, 1], poses_grid[:, 2]))]

    elif targets_type == HoveringTargets.GRID_3x3x3_1m:
        # 3x3x3 grid
        x_vals = [-0.5, 0.0, 0.5]
        y_vals = [-0.5, 0.0, 0.5]
        z_vals = [0.5, 1.0, 1.5]
        poses_grid = np.array([[x, y, z] for x in x_vals for y in y_vals for z in z_vals])
        target_poses = poses_grid[np.lexsort((poses_grid[:, 0], poses_grid[:, 1], poses_grid[:, 2]))]

    elif targets_type == HoveringTargets.GRID_4x4x4_1m:
        # 4x4x4 grid
        x_vals = [-0.6, -0.2, 0.2, 0.6]
        y_vals = [-0.6, -0.2, 0.2, 0.6]
        z_vals = [0.4, 0.8, 1.2, 1.6]
        poses_grid = np.array([[x, y, z] for x in x_vals for y in y_vals for z in z_vals])
        target_poses = poses_grid[np.lexsort((poses_grid[:, 0], poses_grid[:, 1], poses_grid[:, 2]))]

    elif targets_type == HoveringTargets.RANDOM_50_1m:
        # 50 random poses within a box centered at the origin
        N = 50
        rng = np.random.default_rng(42)
        max_offset = np.sqrt(1/3, dtype=np.float32) - 0.05    # avoid exact edges (5cm buffer)

        x = rng.uniform(-max_offset, max_offset, size=(N, 1))
        y = rng.uniform(-max_offset, max_offset, size=(N, 1))
        z = rng.uniform(1.0-max_offset, 1.0+max_offset, size=(N, 1))
        target_poses = np.hstack((x, y, z))     # shape: [N, 3]

    return target_poses + np.array(offset)  # add offset to all poses