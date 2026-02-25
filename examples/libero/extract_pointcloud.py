"""Capture multi-view RGB + depth images from LIBERO scenes.

Positions the agentview camera along a spiral path on a hemisphere around the
scene center, rendering RGB + depth at each viewpoint. Consecutive frames are
always close together for good overlap, while the spiral covers the full range
of elevations and azimuths.

Outputs NeRF-style transforms.json with camera intrinsics/extrinsics for
downstream 3D reconstruction.

Usage:
    python extract_pointcloud.py \
        --task_suite_name libero_spatial \
        --task_id 0 \
        --output_dir data/multiview

    # More views, farther, wider elevation
    python extract_pointcloud.py \
        --n_frames 120 --radius 0.8 --elevation_range '(10.0, 80.0)'
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
from typing import Tuple

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from PIL import Image
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_real_depth_map,
)
import tyro

DUMMY_ACTION = [0.0] * 6 + [-1.0]
ROBOT_GEOM_PREFIXES = ("robot0_", "gripper0_")


@dataclasses.dataclass
class Args:
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    episode_idx: int = 0
    resolution: int = 256
    output_dir: str = "data/multiview"

    # Hemisphere spiral sampling
    n_frames: int = 400
    """Total number of viewpoints to capture."""
    radius: float = 1.00
    """Distance from lookat center to each camera position (meters)."""
    elevation_range: Tuple[float, float] = (15.0, 75.0)
    """(min, max) elevation in degrees above the horizontal plane."""
    n_revolutions: float = 0.0
    """Number of full azimuth revolutions in the spiral. 0 = auto-compute."""
    lookat: Tuple[float, float, float] = (0.2, 0.0, 0.85)
    """World-frame point that every camera looks at (scene center)."""

    num_steps_wait: int = 10
    seed: int = 7


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
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


def lookat_quat(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute MuJoCo quaternion (w,x,y,z) so the camera looks at target.

    MuJoCo camera convention: -Z is the viewing direction, Y is up, X is right.
    """
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0.0, 0.0, 1.0])

    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        # Camera looking straight down or up — pick an arbitrary right vector
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
    right = right / right_norm

    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # MuJoCo camera frame: X=right, Y=up, Z=-forward
    R = np.column_stack([right, up, -forward])
    return rotmat_to_quat_wxyz(R)


def spiral_hemisphere(
    n: int,
    radius: float,
    center: np.ndarray,
    elev_min_deg: float,
    elev_max_deg: float,
    n_revolutions: float = 0.0,
) -> list[np.ndarray]:
    """Generate *n* camera positions along a spiral on a spherical cap.

    The spiral descends from *elev_max_deg* to *elev_min_deg* while sweeping
    azimuth, so **consecutive frames are always close together** — ideal for
    stereo overlap and smooth video-style capture.

    When *n_revolutions* is 0 (default), the number of revolutions is chosen
    automatically so that the elevation gap between adjacent tracks roughly
    equals the azimuth step along the track (square-ish coverage).
    """
    elev_span = elev_max_deg - elev_min_deg
    if n_revolutions <= 0:
        # Auto: elevation_step_per_rev ≈ azimuth_step_per_frame
        #   elev_span / n_rev ≈ 360 * n_rev / n
        #   => n_rev = sqrt(n * elev_span / 360)
        n_revolutions = max(1.0, np.sqrt(n * elev_span / 360.0))

    positions = []
    for i in range(n):
        t = i / max(n - 1, 1)
        elev = np.radians(elev_max_deg - t * elev_span)  # high → low
        azimuth = 2.0 * np.pi * n_revolutions * t

        cos_elev = np.cos(elev)
        x = radius * cos_elev * np.cos(azimuth) + center[0]
        y = radius * cos_elev * np.sin(azimuth) + center[1]
        z = radius * np.sin(elev) + center[2]
        positions.append(np.array([x, y, z]))

    return positions


# ---------------------------------------------------------------------------
# Robot visibility
# ---------------------------------------------------------------------------

def hide_robot_geoms(sim) -> np.ndarray:
    """Set alpha=0 on all robot geoms so they don't appear in renders.

    Returns the original alpha values so they can be restored later.
    """
    model = sim.model
    saved_alpha = model.geom_rgba[:, 3].copy()
    n_hidden = 0
    for gid in range(model.ngeom):
        name = model.geom_id2name(gid) or ""
        if name.startswith(ROBOT_GEOM_PREFIXES):
            model.geom_rgba[gid, 3] = 0.0
            n_hidden += 1
    logging.info(f"Hidden {n_hidden} robot geoms (prefixes={ROBOT_GEOM_PREFIXES})")
    return saved_alpha


def restore_robot_geoms(sim, saved_alpha: np.ndarray) -> None:
    """Restore original geom alpha values."""
    sim.model.geom_rgba[:, 3] = saved_alpha


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: Args) -> None:
    np.random.seed(args.seed)

    # --- Initialize environment ---
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)

    task_description = task.language
    logging.info(f"Task: {task_description}")

    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=args.resolution,
        camera_widths=args.resolution,
    )
    env.seed(args.seed)

    env.reset()
    env.set_init_state(initial_states[args.episode_idx])
    logging.info(f"Waiting {args.num_steps_wait} steps for objects to settle...")
    for _ in range(args.num_steps_wait):
        env.step(DUMMY_ACTION)

    sim = env.sim
    H = W = args.resolution

    # --- Prepare output dirs ---
    out_dir = pathlib.Path(args.output_dir)
    rgb_dir = out_dir / "rgb"
    depth_dir = out_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # --- Save original camera pose ---
    cam_id = sim.model.camera_name2id("agentview")
    orig_pos = sim.model.cam_pos[cam_id].copy()
    orig_quat = sim.model.cam_quat[cam_id].copy()

    # --- Hide robot arm for clean renders ---
    saved_alpha = hide_robot_geoms(sim)

    # --- Build spiral viewpoints ---
    target = np.array(args.lookat)
    positions = spiral_hemisphere(
        n=args.n_frames,
        radius=args.radius,
        center=target,
        elev_min_deg=args.elevation_range[0],
        elev_max_deg=args.elevation_range[1],
        n_revolutions=args.n_revolutions,
    )
    n_frames = len(positions)

    # Compute auto-determined revolutions for logging
    elev_span = args.elevation_range[1] - args.elevation_range[0]
    n_rev = args.n_revolutions if args.n_revolutions > 0 else max(1.0, np.sqrt(n_frames * elev_span / 360.0))
    logging.info(
        f"Capturing {n_frames} views  "
        f"(radius={args.radius}m, elev={args.elevation_range[0]:.0f}-{args.elevation_range[1]:.0f}deg, "
        f"{n_rev:.1f} revolutions)"
    )

    # --- Get intrinsics (constant across all frames) ---
    K = get_camera_intrinsic_matrix(sim, "agentview", H, W)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    frames = []

    for i, pos in enumerate(positions):
        quat = lookat_quat(pos, target)

        # Move camera
        sim.model.cam_pos[cam_id] = pos
        sim.model.cam_quat[cam_id] = quat
        sim.forward()

        # Render
        rgb, depth_raw = sim.render(camera_name="agentview", height=H, width=W, depth=True)
        rgb = rgb[::-1].copy()  # OpenGL bottom-up flip
        depth_raw = depth_raw[::-1].copy()
        depth_real = get_real_depth_map(sim, depth_raw)

        # Extrinsic (cam2world 4x4)
        cam2world = get_camera_extrinsic_matrix(sim, "agentview")

        # Save RGB as PNG
        rgb_path = f"rgb/{i:02d}.png"
        Image.fromarray(rgb).save(str(out_dir / rgb_path))

        # Save depth as .npy (float32, meters)
        depth_path = f"depth/{i:02d}.npy"
        np.save(str(out_dir / depth_path), depth_real.astype(np.float32))

        frames.append({
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "transform_matrix": cam2world.tolist(),
        })

        elev = np.degrees(np.arcsin(np.clip((pos[2] - target[2]) / args.radius, -1.0, 1.0)))
        logging.info(
            f"  [{i+1:2d}/{n_frames}] "
            f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})  elev={elev:.1f}deg  "
            f"depth=[{depth_real.min():.3f}, {depth_real.max():.3f}]m"
        )

    # --- Restore original state ---
    restore_robot_geoms(sim, saved_alpha)
    sim.model.cam_pos[cam_id] = orig_pos
    sim.model.cam_quat[cam_id] = orig_quat
    sim.forward()

    # --- Write transforms.json ---
    transforms = {
        "w": W,
        "h": H,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "frames": frames,
    }
    transforms_path = out_dir / "transforms.json"
    with open(transforms_path, "w") as f:
        json.dump(transforms, f, indent=2)

    logging.info(f"Saved {n_frames} views to {out_dir}")
    logging.info(f"  transforms: {transforms_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(tyro.cli(Args))
