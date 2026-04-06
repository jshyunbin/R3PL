"""
visualize_act_trajectory.py
────────────────────────────
Rolls out a pretrained ACT model and produces a side-by-side video:

  LEFT  │  agentview camera feed (live rollout)
  RIGHT │  3-D plot of the predicted EEF trajectory chunk for the current step

The right panel shows:
  • The full predicted action chunk (horizon steps) as a coloured line, fading
    from blue (near future) to red (far future).
  • The current EEF position as a gold sphere.
  • A faint grey trail of past EEF positions to give motion context.
  • Axis labels and a fixed-world bounding box so the view stays stable.

Action layout (abs_action=True, rotation_6d):
    action[..., 0:3]  → XYZ end-effector position  (world frame, metres)
    action[..., 3:9]  → rotation_6d (not plotted)
    action[..., 9]    → gripper

Usage
─────
python -m src.nn.scripts.visualize_act_trajectory \
    --checkpoint  checkpoints/epoch_200.ckpt \
    --dataset     data/robomimic/can/ph/image_abs.hdf5 \
    --output      trajectory_rollout.mp4 \
    --seed        42 \
    --max_steps   400 \
    --fps         10 \
    --image_resolution 84 \
    --abs_action
"""

import argparse
import collections
import io
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")                          # headless – must be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import torch

from src.nn.common.pytorch_util import dict_apply
from src.nn.common.rotation_transformer import RotationTransformer
from src.nn.env.robomimic_image_wrapper import RobomimicImageWrapper
from src.nn.gym_util.multistep_wrapper import MultiStepWrapper
from src.nn.models.act import ACTModule


# ─────────────────────────────────────────────────────────────────────────────
# Environment helpers  (identical to robomimic_runner pattern)
# ─────────────────────────────────────────────────────────────────────────────

def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta["obs"].items():
        modality_mapping[attr.get("type", "low_dim")].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=enable_render,
        use_image_obs=enable_render,
    )
    return env


def undo_transform_action(action: np.ndarray, rotation_transformer) -> np.ndarray:
    """Convert rotation_6d absolute actions back to axis-angle for the env."""
    raw_shape = action.shape
    if raw_shape[-1] == 20:
        action = action.reshape(-1, 2, 10)
    d_rot   = action.shape[-1] - 4
    pos     = action[..., :3]
    rot     = action[..., 3: 3 + d_rot]
    gripper = action[..., [-1]]
    rot     = rotation_transformer.inverse(rot)
    uaction = np.concatenate([pos, rot, gripper], axis=-1)
    if raw_shape[-1] == 20:
        uaction = uaction.reshape(*raw_shape[:-1], 14)
    return uaction


# ─────────────────────────────────────────────────────────────────────────────
# 3-D trajectory panel renderer
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryRenderer:
    """
    Keeps a persistent Matplotlib 3-D axes and renders each frame to a
    numpy array (H, W, 3) uint8 without re-creating the figure.

    Parameters
    ----------
    panel_h, panel_w : int
        Pixel dimensions of the rendered panel.
    world_bounds : dict with keys 'x', 'y', 'z' each a (lo, hi) tuple.
        Fixed axis limits so the view never jumps between frames.
        If None, limits are auto-set from the first predicted trajectory.
    trail_len : int
        Number of past EEF positions to show as a grey trail.
    elev, azim : float
        Initial 3-D viewing angle (degrees).
    """

    def __init__(self,
                 panel_h: int = 480,
                 panel_w: int = 480,
                 world_bounds: dict | None = None,
                 trail_len: int = 60,
                 elev: float = 25.0,
                 azim: float = -60.0):

        self.panel_h = panel_h
        self.panel_w = panel_w
        self.trail_len = trail_len
        self.world_bounds = world_bounds        # set on first render if None
        self._eef_trail: list[np.ndarray] = [] # list of (3,) xyz

        dpi = 100
        self.fig = plt.figure(figsize=(panel_w / dpi, panel_h / dpi), dpi=dpi)
        self.ax  = self.fig.add_subplot(111, projection="3d")
        self.ax.view_init(elev=elev, azim=azim)

        self.ax.set_xlabel("X (m)", fontsize=7, labelpad=2)
        self.ax.set_ylabel("Y (m)", fontsize=7, labelpad=2)
        self.ax.set_zlabel("Z (m)", fontsize=7, labelpad=2)
        self.ax.tick_params(labelsize=6)
        self.fig.tight_layout(pad=0.5)

    def _auto_bounds(self, traj_xyz: np.ndarray):
        """Compute axis limits with a small margin from first trajectory."""
        margin = 0.05
        lo = traj_xyz.min(axis=0) - margin
        hi = traj_xyz.max(axis=0) + margin
        # Make each axis at least 0.15 m wide so it never collapses
        for i in range(3):
            if hi[i] - lo[i] < 0.15:
                mid = (hi[i] + lo[i]) / 2
                lo[i] = mid - 0.075
                hi[i] = mid + 0.075
        self.world_bounds = {
            "x": (float(lo[0]), float(hi[0])),
            "y": (float(lo[1]), float(hi[1])),
            "z": (float(lo[2]), float(hi[2])),
        }

    def render(self,
               pred_xyz: np.ndarray,    # (horizon, 3)  predicted EEF positions
               current_xyz: np.ndarray  # (3,)           current EEF position
               ) -> np.ndarray:
        """
        Draw and return an (H, W, 3) uint8 RGB frame.

        pred_xyz    – the full predicted action-chunk EEF positions
        current_xyz – the robot's actual current EEF position (from obs)
        """

        # ── Auto-set bounds once from the first trajectory ────────────────
        if self.world_bounds is None:
            combined = np.vstack([pred_xyz, current_xyz[None]])
            self._auto_bounds(combined)

        # ── Update trail ──────────────────────────────────────────────────
        self._eef_trail.append(current_xyz.copy())
        if len(self._eef_trail) > self.trail_len:
            self._eef_trail.pop(0)

        # ── Clear and redraw ──────────────────────────────────────────────
        self.ax.cla()
        bx = self.world_bounds["x"]
        by = self.world_bounds["y"]
        bz = self.world_bounds["z"]
        self.ax.set_xlim(*bx)
        self.ax.set_ylim(*by)
        self.ax.set_zlim(*bz)
        self.ax.set_xlabel("X (m)", fontsize=7, labelpad=2)
        self.ax.set_ylabel("Y (m)", fontsize=7, labelpad=2)
        self.ax.set_zlabel("Z (m)", fontsize=7, labelpad=2)
        self.ax.tick_params(labelsize=6)
        self.ax.set_title("Predicted EEF trajectory", fontsize=8, pad=4)

        # ── Past trail (grey, fading) ─────────────────────────────────────
        if len(self._eef_trail) > 1:
            trail = np.array(self._eef_trail)           # (T, 3)
            n = len(trail)
            for i in range(n - 1):
                alpha = 0.15 + 0.55 * (i / max(n - 2, 1))
                self.ax.plot(
                    trail[i:i+2, 0], trail[i:i+2, 1], trail[i:i+2, 2],
                    color=(0.55, 0.55, 0.55), alpha=alpha, linewidth=1.2
                )

        # ── Predicted trajectory (blue → red gradient) ────────────────────
        horizon = len(pred_xyz)
        colours = cm.coolwarm(np.linspace(0.0, 1.0, horizon))   # blue=near, red=far
        for i in range(horizon - 1):
            self.ax.plot(
                pred_xyz[i:i+2, 0], pred_xyz[i:i+2, 1], pred_xyz[i:i+2, 2],
                color=colours[i], linewidth=2.0, alpha=0.85
            )

        # Dot at each predicted waypoint
        self.ax.scatter(
            pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2],
            c=colours, s=8, zorder=5, depthshade=True
        )

        # ── Current EEF position (gold sphere) ────────────────────────────
        self.ax.scatter(
            [current_xyz[0]], [current_xyz[1]], [current_xyz[2]],
            c=[(1.0, 0.75, 0.0)], s=60, zorder=10, depthshade=False,
            edgecolors="black", linewidths=0.6, label="current EEF"
        )

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=cm.coolwarm(0.0), lw=2, label="near future"),
            Line2D([0], [0], color=cm.coolwarm(1.0), lw=2, label="far future"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=(1.0, 0.75, 0.0), markersize=7,
                   markeredgecolor="black", label="current EEF"),
        ]
        self.ax.legend(handles=legend_elements, fontsize=6,
                       loc="upper left", framealpha=0.6)

        # ── Render to numpy array ──────────────────────────────────────────
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=self.fig.dpi, bbox_inches="tight")
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        panel = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)         # BGR
        panel = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)          # → RGB
        panel = cv2.resize(panel, (self.panel_w, self.panel_h)) # ensure exact size
        return panel

    def close(self):
        plt.close(self.fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    model: ACTModule = ACTModule.load_from_checkpoint(
        args.checkpoint, map_location=device, weights_only=False
    )
    model.eval().to(device)
    shape_meta = model.hparams.shape_meta

    # ── Build environment ─────────────────────────────────────────────────────
    dataset_path = os.path.expanduser(args.dataset)
    env_meta     = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env_meta["env_kwargs"]["use_object_obs"] = False

    rotation_transformer = None
    if args.abs_action:
        env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
        rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")

    robomimic_env = create_env(env_meta=env_meta, shape_meta=shape_meta)
    robomimic_env.env.hard_reset = False

    agentview_key = "agentview_image"
    env = MultiStepWrapper(
        RobomimicImageWrapper(
            env=robomimic_env,
            shape_meta=shape_meta,
            init_state=None,
            render_obs_key=agentview_key,
        ),
        n_obs_steps=1,
        n_action_steps=model.horizon,
        max_episode_steps=args.max_steps,
    )

    # ── Video writer ──────────────────────────────────────────────────────────
    panel_h = args.image_resolution
    panel_w = args.image_resolution
    out_h   = panel_h
    out_w   = panel_w * 2

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (out_w, out_h),
    )

    renderer = TrajectoryRenderer(
        panel_h=panel_h,
        panel_w=panel_w,
        world_bounds=None,           # auto-set from first trajectory
        trail_len=args.trail_len,
        elev=args.elev,
        azim=args.azim,
    )

    # ── Rollout ───────────────────────────────────────────────────────────────
    obs, _ = env.reset(seed=args.seed)
    step   = 0
    done   = False
    print(f"Starting rollout (max {args.max_steps} steps)…")

    while not done and step < args.max_steps:

        # Transfer obs to device
        obs_dict = dict_apply(dict(obs), lambda x: torch.from_numpy(x).to(device=device))

        # Forward pass — returns unnormalized actions (batch=1, horizon, action_dim)
        with torch.no_grad():
            action_hat = model.predict_action({"obs": obs_dict})["pred_action"]

        action = action_hat.cpu().numpy()                    # (1, horizon, action_dim)

        # ── Extract predicted EEF XYZ (first 3 dims = world-frame position) ──
        # predict_action already calls normalizer.unnormalize, so these are
        # real-world metres.
        pred_xyz = action[0, :, :3]                          # (horizon, 3)

        # Current EEF position from obs (also real-world after unnorm)
        # robot0_eef_pos is a proprioceptive key, normalised to [-1,1] by the
        # dataset normalizer, so we read the raw last obs value and unnorm via
        # the model's own normalizer for consistency.
        eef_obs = obs["robot0_eef_pos"][-1]                  # (3,) in normalizer space
        eef_tensor = torch.from_numpy(eef_obs).unsqueeze(0).to(device)
        current_xyz = model.normalizer["robot0_eef_pos"].unnormalize(eef_tensor)
        current_xyz = current_xyz.squeeze(0).cpu().numpy()  # (3,)

        # ── Agentview frame (left panel) ──────────────────────────────────────
        agent_chw = obs[agentview_key][-1]                   # (C, H, W), [0,1]
        agent_img = (np.moveaxis(agent_chw, 0, -1) * 255).astype(np.uint8)
        agent_img = cv2.resize(agent_img, (panel_w, panel_h))

        # ── 3-D trajectory panel (right panel) ────────────────────────────────
        traj_panel = renderer.render(pred_xyz, current_xyz)  # (H, W, 3) RGB

        # ── Labels ────────────────────────────────────────────────────────────
        def put_label(img, text):
            return cv2.putText(
                img.copy(), text, (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
            )

        agent_img  = put_label(agent_img,  f"agentview  step={step}")
        traj_panel = put_label(traj_panel, "predicted trajectory")

        # ── Compose and write ─────────────────────────────────────────────────
        frame     = np.concatenate([agent_img, traj_panel], axis=1)  # (H, 2W, 3)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

        # ── Step env ──────────────────────────────────────────────────────────
        env_action = action[:, :model.horizon, :]
        if args.abs_action and rotation_transformer is not None:
            env_action = undo_transform_action(env_action, rotation_transformer)

        obs, reward, terminated, truncated, info = env.step(env_action)
        done  = bool(terminated or truncated)
        step += env_action.shape[1]

        if step % 20 == 0:
            print(f"  step {step:4d}  reward={reward:.3f}")

    writer.release()
    renderer.close()
    print(f"\nDone. Video saved to: {output_path}  ({step} steps, done={done})")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ACT predicted-trajectory rollout visualizer")
    p.add_argument("--checkpoint",       required=True,
                   help="Path to .ckpt file")
    p.add_argument("--dataset",          required=True,
                   help="Path to robomimic .hdf5 dataset (used only for env_meta)")
    p.add_argument("--output",           default="trajectory_rollout.mp4")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--max_steps",        type=int,   default=400)
    p.add_argument("--fps",              type=int,   default=10)
    p.add_argument("--image_resolution", type=int,   default=84,
                   help="Square image resolution used during training")
    p.add_argument("--abs_action",       action="store_true",
                   help="Model was trained with absolute actions (rotation_6d)")
    p.add_argument("--trail_len",        type=int,   default=60,
                   help="Number of past EEF positions shown as a grey trail")
    p.add_argument("--elev",             type=float, default=25.0,
                   help="3-D plot elevation angle (degrees)")
    p.add_argument("--azim",             type=float, default=-60.0,
                   help="3-D plot azimuth angle (degrees)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())