"""Fuse multi-view RGB + depth captures into a single colored point cloud.

Reads the output of extract_pointcloud.py (transforms.json + rgb/*.png +
depth/*.npy), back-projects each pixel into world coordinates using the stored
camera intrinsics and extrinsics, and writes a single PLY file.

Usage:
    python fuse_pointcloud.py --input_dir data/multiview --output data/multiview/scene.ply

    # With voxel downsampling and depth filtering
    python fuse_pointcloud.py \
        --input_dir data/multiview \
        --output data/multiview/scene.ply \
        --max_depth 3.0 \
        --voxel_size 0.002
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib

import numpy as np
from PIL import Image
import tyro


@dataclasses.dataclass
class Args:
    input_dir: str = "data/multiview"
    """Directory containing transforms.json, rgb/, and depth/ from extract_pointcloud.py."""
    output: str = "data/multiview/scene.ply"
    """Output PLY file path."""
    max_depth: float = 3.0
    """Discard pixels with depth beyond this value (meters)."""
    voxel_size: float = 0.002
    """Voxel size for downsampling (meters). 0 = no downsampling."""
    skip_frames: int = 1
    """Use every N-th frame (1 = all frames)."""


# ---------------------------------------------------------------------------
# Back-projection
# ---------------------------------------------------------------------------

def backproject_frame(
    rgb: np.ndarray,
    depth: np.ndarray,
    cam2world: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a single RGB-D frame into world-frame 3D points.

    Args:
        rgb: (H, W, 3) uint8 image.
        depth: (H, W) float32 depth in meters.
        cam2world: (4, 4) camera-to-world transform.
        fx, fy, cx, cy: camera intrinsics.
        max_depth: ignore pixels beyond this depth.

    Returns:
        points: (N, 3) float32 world coordinates.
        colors: (N, 3) uint8 RGB.
    """
    H, W = depth.shape

    # Pixel grid
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    # Depth mask
    valid = (depth > 0) & (depth < max_depth)

    z = depth[valid]
    u_v = u[valid]
    v_v = v[valid]

    # Camera-frame 3D (OpenCV convention: X-right, Y-down, Z-forward)
    x_cam = (u_v - cx) * z / fx
    y_cam = (v_v - cy) * z / fy

    # Homogeneous camera coords → world coords
    pts_cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=-1)  # (N, 4)
    pts_world = (cam2world @ pts_cam.T).T[:, :3]  # (N, 3)

    colors = rgb[valid]  # (N, 3)

    return pts_world.astype(np.float32), colors.astype(np.uint8)


# ---------------------------------------------------------------------------
# Voxel downsampling
# ---------------------------------------------------------------------------

def voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Average points and colors within each voxel."""
    # Quantize to voxel indices
    indices = np.floor(points / voxel_size).astype(np.int64)

    # Pack (ix, iy, iz) into a single key per point for grouping
    # Shift to non-negative to avoid issues with negative coords
    mins = indices.min(axis=0)
    indices -= mins
    maxs = indices.max(axis=0) + 1

    keys = indices[:, 0] * (maxs[1] * maxs[2]) + indices[:, 1] * maxs[2] + indices[:, 2]

    # Group by voxel: use pandas-free approach with np.unique + bincount
    unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    n_voxels = len(unique_keys)

    # Sum points and colors per voxel
    pts_sum = np.zeros((n_voxels, 3), dtype=np.float64)
    col_sum = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(pts_sum, inverse, points.astype(np.float64))
    np.add.at(col_sum, inverse, colors.astype(np.float64))

    counts_f = counts[:, None].astype(np.float64)
    pts_avg = (pts_sum / counts_f).astype(np.float32)
    col_avg = np.clip(col_sum / counts_f, 0, 255).astype(np.uint8)

    return pts_avg, col_avg


# ---------------------------------------------------------------------------
# PLY writer
# ---------------------------------------------------------------------------

def save_ply(filepath: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a binary PLY with XYZ + RGB."""
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )

    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    arr = np.empty(n, dtype=dtype)
    arr["x"] = points[:, 0]
    arr["y"] = points[:, 1]
    arr["z"] = points[:, 2]
    arr["r"] = colors[:, 0]
    arr["g"] = colors[:, 1]
    arr["b"] = colors[:, 2]

    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: Args) -> None:
    input_dir = pathlib.Path(args.input_dir)
    transforms_path = input_dir / "transforms.json"

    with open(transforms_path) as f:
        transforms = json.load(f)

    fx = transforms["fx"]
    fy = transforms["fy"]
    cx = transforms["cx"]
    cy = transforms["cy"]
    all_frames = transforms["frames"]

    # Apply frame skipping
    frames = all_frames[:: args.skip_frames]
    logging.info(f"Fusing {len(frames)}/{len(all_frames)} frames (skip={args.skip_frames})")

    all_points = []
    all_colors = []

    for i, frame in enumerate(frames):
        rgb_path = input_dir / frame["rgb_path"]
        depth_path = input_dir / frame["depth_path"]
        cam2world = np.array(frame["transform_matrix"], dtype=np.float64)

        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        depth = np.load(depth_path)

        pts, cols = backproject_frame(rgb, depth, cam2world, fx, fy, cx, cy, args.max_depth)
        all_points.append(pts)
        all_colors.append(cols)

        if (i + 1) % 20 == 0 or i == len(frames) - 1:
            total = sum(len(p) for p in all_points)
            logging.info(f"  [{i+1}/{len(frames)}] {total:,} points so far")

    points = np.concatenate(all_points)
    colors = np.concatenate(all_colors)
    logging.info(f"Total raw points: {len(points):,}")

    # Voxel downsampling
    if args.voxel_size > 0:
        logging.info(f"Voxel downsampling (size={args.voxel_size}m)...")
        points, colors = voxel_downsample(points, colors, args.voxel_size)
        logging.info(f"After downsampling: {len(points):,} points")

    # Save
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_ply(str(output_path), points, colors)
    logging.info(f"Saved {len(points):,} points to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(tyro.cli(Args))
