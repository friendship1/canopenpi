"""Voxel grid utilities for LIBERO dataset pipeline.

Generates RGB-colorized 3D voxel grids from multi-view RGB-D renders around the
gripper. Uses hemisphere-placed virtual cameras to capture the scene, then
back-projects to a point cloud, crops around the gripper pose, and voxelizes.
"""

from __future__ import annotations

import os

import numpy as np
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_real_depth_map,
)


# ---------------------------------------------------------------------------
# Geometry helpers (adapted from extract_pointcloud.py)
# ---------------------------------------------------------------------------

def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to MuJoCo quaternion (w, x, y, z)."""
    m = R
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert MuJoCo quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def _lookat_quat(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute MuJoCo quaternion (w,x,y,z) so the camera looks at *target*.

    MuJoCo camera convention: -Z is viewing direction, Y is up, X is right.
    """
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
    right = right / right_norm

    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # MuJoCo camera frame: X=right, Y=up, Z=-forward
    R = np.column_stack([right, up, -forward])
    return _rotmat_to_quat_wxyz(R)


# ---------------------------------------------------------------------------
# Hemisphere camera generation
# ---------------------------------------------------------------------------

def generate_hemisphere_cameras(
    n_cameras: int = 10,
    radius: float = 1.0,
    center: tuple[float, float, float] = (0.2, 0.0, 0.85),
    elev_range: tuple[float, float] = (15.0, 75.0),
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate fixed camera (position, quaternion) pairs on a hemisphere.

    Cameras are placed along a spiral on the hemisphere, all pointing toward
    *center*. Uses the same spiral approach as extract_pointcloud.py but with
    fewer points.

    Returns:
        List of (position, quat_wxyz) tuples.
    """
    center = np.asarray(center, dtype=np.float64)
    elev_min, elev_max = elev_range
    elev_span = elev_max - elev_min

    # Auto-compute revolutions for uniform coverage
    n_revolutions = max(1.0, np.sqrt(n_cameras * elev_span / 360.0))

    cameras = []
    for i in range(n_cameras):
        t = i / max(n_cameras - 1, 1)
        elev = np.radians(elev_max - t * elev_span)  # high → low
        azimuth = 2.0 * np.pi * n_revolutions * t

        cos_elev = np.cos(elev)
        x = radius * cos_elev * np.cos(azimuth) + center[0]
        y = radius * cos_elev * np.sin(azimuth) + center[1]
        z = radius * np.sin(elev) + center[2]

        pos = np.array([x, y, z])
        quat = _lookat_quat(pos, center)
        cameras.append((pos, quat))

    return cameras


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_rgbd(
    sim,
    camera_name: str,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    height: int,
    width: int,
    cam_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render RGB-D by writing camera pose directly to sim.data (no sim.forward()).

    Writes to sim.data.cam_xpos / cam_xmat so both the renderer and
    get_camera_extrinsic_matrix see the updated pose immediately.
    The caller is responsible for saving/restoring sim.data camera state.

    Args:
        sim: MuJoCo simulation object.
        camera_name: Name of the camera to reposition (e.g. "agentview").
        cam_pos: Desired camera position (3,).
        cam_quat: Desired camera quaternion wxyz (4,).
        height, width: Render resolution.
        cam_id: Pre-resolved camera ID (avoids repeated name lookup).

    Returns:
        rgb: (H, W, 3) uint8
        depth_real: (H, W) float32 in meters
    """
    if cam_id is None:
        cam_id = sim.model.camera_name2id(camera_name)

    # Write pose directly to sim.data — no sim.forward() needed
    sim.data.cam_xpos[cam_id] = cam_pos
    sim.data.cam_xmat[cam_id] = _quat_wxyz_to_rotmat(cam_quat).flatten()

    # Render
    rgb, depth_raw = sim.render(camera_name=camera_name, height=height, width=width, depth=True)
    rgb = rgb[::-1].copy()  # OpenGL vertical flip
    depth_raw = depth_raw[::-1].copy()
    depth_real = get_real_depth_map(sim, depth_raw)

    return rgb, depth_real


# ---------------------------------------------------------------------------
# Back-projection
# ---------------------------------------------------------------------------

def backproject_to_pointcloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    sim,
    camera_name: str,
    height: int,
    width: int,
    max_depth: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project an RGB-D frame into world-space 3D points.

    Uses robosuite camera utilities to get intrinsics and extrinsics.

    Returns:
        points_world: (N, 3) float32
        colors: (N, 3) float32 in [0, 1]
    """
    K = get_camera_intrinsic_matrix(sim, camera_name, height, width)
    cam2world = get_camera_extrinsic_matrix(sim, camera_name)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    valid = (depth > 0) & (depth < max_depth)
    z = depth[valid]
    u_v = u[valid]
    v_v = v[valid]

    # Camera-frame 3D (OpenCV convention)
    x_cam = (u_v - cx) * z / fx
    y_cam = (v_v - cy) * z / fy

    pts_cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=-1)  # (N, 4)
    pts_world = (cam2world @ pts_cam.T).T[:, :3]  # (N, 3)

    colors = rgb[valid].astype(np.float32) / 255.0  # (N, 3) normalized

    return pts_world.astype(np.float32), colors


# ---------------------------------------------------------------------------
# Multi-view scene point cloud
# ---------------------------------------------------------------------------

def build_scene_pointcloud(
    sim,
    camera_positions: list[tuple[np.ndarray, np.ndarray]],
    camera_name: str,
    height: int,
    width: int,
    max_depth: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Render from all hemisphere cameras and fuse into a single point cloud.

    Writes camera poses directly to sim.data (no sim.forward() calls).
    Saves and restores the original sim.data camera state once.

    Args:
        sim: MuJoCo simulation object.
        camera_positions: List of (position, quat_wxyz) from generate_hemisphere_cameras.
        camera_name: Camera to reposition (e.g. "agentview").
        height, width: Render resolution.
        max_depth: Discard pixels beyond this depth.

    Returns:
        points_world: (N, 3) float32
        colors: (N, 3) float32 in [0, 1]
    """
    cam_id = sim.model.camera_name2id(camera_name)

    # Save original sim.data camera state (once)
    orig_xpos = sim.data.cam_xpos[cam_id].copy()
    orig_xmat = sim.data.cam_xmat[cam_id].copy()

    all_points = []
    all_colors = []

    for cam_pos, cam_quat in camera_positions:
        rgb, depth = render_rgbd(sim, camera_name, cam_pos, cam_quat, height, width, cam_id=cam_id)
        pts, cols = backproject_to_pointcloud(rgb, depth, sim, camera_name, height, width, max_depth)
        all_points.append(pts)
        all_colors.append(cols)

    # Restore original sim.data camera state (once, no forward needed)
    sim.data.cam_xpos[cam_id] = orig_xpos
    sim.data.cam_xmat[cam_id] = orig_xmat

    if all_points:
        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
    else:
        points = np.zeros((0, 3), dtype=np.float32)
        colors = np.zeros((0, 3), dtype=np.float32)

    return points, colors


# ---------------------------------------------------------------------------
# Voxel downsampling
# ---------------------------------------------------------------------------

def voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float = 0.002,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample point cloud by averaging within voxels.

    Args:
        points: (N, 3) float32
        colors: (N, 3) float32 in [0, 1]
        voxel_size: voxel size in meters

    Returns:
        (points_ds, colors_ds) both float32
    """
    if len(points) == 0:
        return points, colors

    # Quantize to voxel indices
    indices = np.floor(points / voxel_size).astype(np.int64)

    # Shift to non-negative
    mins = indices.min(axis=0)
    indices -= mins
    maxs = indices.max(axis=0) + 1

    keys = indices[:, 0] * (maxs[1] * maxs[2]) + indices[:, 1] * maxs[2] + indices[:, 2]

    # Group by voxel
    unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    n_voxels = len(unique_keys)

    # Sum points and colors per voxel
    pts_sum = np.zeros((n_voxels, 3), dtype=np.float64)
    col_sum = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(pts_sum, inverse, points.astype(np.float64))
    np.add.at(col_sum, inverse, colors.astype(np.float64))

    counts_f = counts[:, None].astype(np.float64)
    pts_avg = (pts_sum / counts_f).astype(np.float32)
    col_avg = np.clip(col_sum / counts_f, 0.0, 1.0).astype(np.float32)

    return pts_avg, col_avg


# ---------------------------------------------------------------------------
# Save scene point cloud
# ---------------------------------------------------------------------------

def save_scene_pointcloud(
    sim,
    camera_positions: list[tuple[np.ndarray, np.ndarray]],
    camera_name: str,
    height: int,
    width: int,
    output_path: str,
    voxel_size: float = 0.002,
    max_depth: float = 3.0,
) -> None:
    """Build multi-view point cloud, downsample, save as .npz.

    Saves: points (N,3) float32, colors (N,3) uint8 [0-255]
    """
    points, colors = build_scene_pointcloud(
        sim, camera_positions, camera_name, height, width, max_depth
    )
    if len(points) > 0:
        points, colors = voxel_downsample(points, colors, voxel_size)
    # Convert colors to uint8 to save space
    colors_u8 = np.clip(colors * 255, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, points=points, colors=colors_u8)


# ---------------------------------------------------------------------------
# Coordinate transform
# ---------------------------------------------------------------------------

def world_to_gripper_frame(
    points: np.ndarray,
    eef_pos: np.ndarray,
    eef_quat: np.ndarray,
) -> np.ndarray:
    """Transform world-frame points into the gripper-local frame.

    Args:
        points: (N, 3) world coordinates.
        eef_pos: (3,) end-effector position in world frame.
        eef_quat: (4,) end-effector quaternion (x, y, z, w) — robosuite convention.

    Returns:
        points_local: (N, 3) in gripper frame.
    """
    import robosuite.utils.transform_utils as T

    R = T.quat2mat(eef_quat)  # (3, 3), robosuite uses xyzw convention
    # R^T * (p - t)
    points_local = (points - eef_pos[None, :]) @ R  # R^T via (p-t) @ R
    return points_local.astype(np.float32)


# ---------------------------------------------------------------------------
# Voxelization
# ---------------------------------------------------------------------------

def voxelize_points(
    points: np.ndarray,
    colors: np.ndarray,
    grid_size: int = 32,
    extent: float = 0.3,
) -> np.ndarray:
    """Voxelize a colored point cloud into an RGBO grid.

    Maps points in [-extent/2, extent/2]^3 to a (grid_size, grid_size, grid_size, 4)
    tensor. Channel 0-2 are average RGB, channel 3 is occupancy (1.0 if occupied).

    Args:
        points: (N, 3) local-frame coordinates.
        colors: (N, 3) in [0, 1].
        grid_size: Number of voxels per axis.
        extent: Spatial extent in meters (cube side length).

    Returns:
        grid: (grid_size, grid_size, grid_size, 4) float32 — RGBO.
    """
    half = extent / 2.0

    # Filter points within the extent cube
    mask = np.all((points >= -half) & (points < half), axis=1)
    pts = points[mask]
    cols = colors[mask]

    if len(pts) == 0:
        return np.zeros((grid_size, grid_size, grid_size, 4), dtype=np.float32)

    # Map to voxel indices
    voxel_size = extent / grid_size
    indices = np.floor((pts + half) / voxel_size).astype(np.int32)
    indices = np.clip(indices, 0, grid_size - 1)

    # Flatten 3D indices for accumulation
    flat_idx = indices[:, 0] * (grid_size * grid_size) + indices[:, 1] * grid_size + indices[:, 2]

    # Accumulate colors and counts
    color_sum = np.zeros((grid_size**3, 3), dtype=np.float64)
    counts = np.zeros(grid_size**3, dtype=np.float64)

    np.add.at(color_sum, flat_idx, cols.astype(np.float64))
    np.add.at(counts, flat_idx, 1.0)

    # Average colors where occupied
    occupied = counts > 0
    color_avg = np.zeros_like(color_sum)
    color_avg[occupied] = color_sum[occupied] / counts[occupied, None]

    # Build RGBO grid
    grid = np.zeros((grid_size**3, 4), dtype=np.float32)
    grid[:, :3] = color_avg.astype(np.float32)
    grid[occupied, 3] = 1.0

    return grid.reshape(grid_size, grid_size, grid_size, 4)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_voxel_grid(
    sim,
    obs: dict,
    camera_positions: list[tuple[np.ndarray, np.ndarray]],
    camera_name: str,
    height: int,
    width: int,
    grid_size: int = 32,
    extent: float = 0.3,
    max_depth: float = 3.0,
) -> np.ndarray:
    """Compute a gripper-local RGB-occupancy voxel grid from multi-view renders.

    This is the main entry point: renders from all hemisphere cameras, builds a
    fused point cloud, transforms to gripper-local frame, and voxelizes.

    Args:
        sim: MuJoCo simulation object.
        obs: Observation dict from env.step(), must contain robot0_eef_pos and robot0_eef_quat.
        camera_positions: From generate_hemisphere_cameras().
        camera_name: Camera to reposition for rendering.
        height, width: Render resolution.
        grid_size: Voxels per axis.
        extent: Spatial extent in meters.
        max_depth: Discard depth beyond this.

    Returns:
        voxel_grid: (grid_size, grid_size, grid_size, 4) float32 — RGBO.
    """
    # Build full scene point cloud from all cameras
    points_world, colors = build_scene_pointcloud(
        sim, camera_positions, camera_name, height, width, max_depth
    )

    if len(points_world) == 0:
        return np.zeros((grid_size, grid_size, grid_size, 4), dtype=np.float32)

    # Transform to gripper-local frame
    eef_pos = obs["robot0_eef_pos"]
    eef_quat = obs["robot0_eef_quat"]
    points_local = world_to_gripper_frame(points_world, eef_pos, eef_quat)

    # Voxelize
    return voxelize_points(points_local, colors, grid_size=grid_size, extent=extent)
