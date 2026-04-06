"""
evaluate_rollout.py
===================
Standalone rollout evaluation script that mirrors the behaviour of
``RobomimicRunner`` but with no W&B / Lightning logger dependency.

For each episode the script:
  1. Resets the environment with a deterministic seed (reproducible via the
     ``seed`` field in the config).
  2. Rolls out the loaded model for up to ``max_steps`` control steps.
  3. Records a video (MP4) under ``<output_dir>/media/``.
  4. Writes a plain-text statistical report to ``<output_dir>/report.txt``.

Report contents
---------------
  Per-episode : seed, max_reward, success (max_reward == 1.0)
  Aggregate   : mean / min / max of max_reward, success rate

Usage
-----
    python evaluate_rollout.py <config_name>

    # Extension is optional — both forms work:
    python evaluate_rollout.py eval_config
    python evaluate_rollout.py eval_config.yaml

    The config file is resolved relative to the current working directory.

Config file schema  (YAML)
--------------------------
    checkpoint:      checkpoints/epoch_250-0.00.ckpt
    dataset:         data/robomimic/can/ph/image_abs.hdf5
    output_dir:      eval_output
    n_episodes:      22
    seed:            10000
    max_steps:       400          # optional, default 400
    n_obs_steps:     1            # optional, default 1
    n_action_steps:  8            # optional, default 8
    render_obs_key:  agentview_image   # optional
    fps:             10           # optional, default 10
    abs_action:      false        # optional, default false
    temporal_agg:    false        # optional, default false
    tqdm_interval:   5.0          # optional, default 5.0

    model:
      _target_: src.nn.models.actrm.ACTRMModule

    shape_meta:
      action:
        shape: [10]
      obs:
        robot0_eef_pos:
          shape: [3]
        robot0_eef_quat:
          shape: [4]
        robot0_joint_pos:
          shape: [7]
        robot0_gripper_qpos:
          shape: [2]
        robot0_eye_in_hand_image:
          shape: [3, 84, 84]
          type: rgb
        agentview_image:
          shape: [3, 84, 84]
          type: rgb
"""

import collections
import math
import os
import pathlib
import sys
import logging

from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER
ROBOSUITE_DEFAULT_LOGGER.setLevel(logging.ERROR)

import cv2
import dill
import numpy as np
import torch
import tqdm
import hydra.utils
from omegaconf import OmegaConf

import src.robothink.envs
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils

from src.nn.common.pytorch_util import dict_apply
from src.nn.common.rotation_transformer import RotationTransformer
from src.nn.env.robomimic_image_wrapper import RobomimicImageWrapper
from src.nn.gym_util.async_vector_env import AsyncVectorEnv
from src.nn.gym_util.multistep_wrapper import MultiStepWrapper
from src.nn.gym_util.video_wrapper import VideoWrapper


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_name: str):
    """Load a YAML config from the current working directory.

    The ``.yaml`` extension is appended automatically if omitted.
    """
    if not config_name.endswith(".yaml") and not config_name.endswith(".yml"):
        config_name = config_name + ".yaml"

    config_path = pathlib.Path(__file__).parent / "configs" / config_name
    if not config_path.is_file():
        print(f"[ERROR] Config file not found: '{config_path.resolve()}'",
              file=sys.stderr)
        sys.exit(1)

    return OmegaConf.load(config_path)


# ---------------------------------------------------------------------------
# Environment factory  (identical to the original RobomimicRunner)
# ---------------------------------------------------------------------------

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


def build_env_fns(env_meta, shape_meta, render_obs_key, n_obs_steps,
                  n_action_steps, max_steps):
    """Return (env_fn, dummy_env_fn) closures."""
    steps_per_render = 2

    def env_fn():
        robomimic_env = create_env(env_meta=env_meta, shape_meta=shape_meta)
        robomimic_env.env.hard_reset = False
        return MultiStepWrapper(
            VideoWrapper(
                RobomimicImageWrapper(
                    env=robomimic_env,
                    shape_meta=shape_meta,
                    init_state=None,
                    render_obs_key=render_obs_key,
                ),
                steps_per_render=steps_per_render,
            ),
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps,
        )

    def dummy_env_fn():
        robomimic_env = create_env(
            env_meta=env_meta, shape_meta=shape_meta, enable_render=False
        )
        return MultiStepWrapper(
            VideoWrapper(
                RobomimicImageWrapper(
                    env=robomimic_env,
                    shape_meta=shape_meta,
                    init_state=None,
                    render_obs_key=render_obs_key,
                ),
                steps_per_render=steps_per_render,
            ),
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps,
        )

    return env_fn, dummy_env_fn


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def run_rollout(
    model,
    env,
    env_fns,
    env_seeds,
    env_init_fn_dills,
    max_steps,
    n_obs_steps,
    n_action_steps,
    past_action,
    abs_action,
    temporal_agg,
    action_dim,
    rotation_transformer,
    tqdm_interval_sec,
    device,
):
    """Execute all episodes and return (videos, all_rewards)."""
    n_envs   = len(env_fns)
    n_inits  = len(env_init_fn_dills)
    n_chunks = math.ceil(n_inits / n_envs)

    videos      = [None] * n_inits
    all_rewards = [None] * n_inits

    for chunk_idx in range(n_chunks):
        start = chunk_idx * n_envs
        end   = min(n_inits, start + n_envs)
        this_global_slice  = slice(start, end)
        this_n_active_envs = end - start
        this_local_slice   = slice(0, this_n_active_envs)

        this_init_fns = env_init_fn_dills[this_global_slice]
        n_diff = n_envs - len(this_init_fns)
        if n_diff > 0:
            this_init_fns.extend([env_init_fn_dills[0]] * n_diff)
        assert len(this_init_fns) == n_envs

        env.call_each("run_dill_function", args_list=[(x,) for x in this_init_fns])

        obs, _ = env.reset(seed=env_seeds)
        past_action_list = []

        pbar = tqdm.tqdm(
            total=max_steps,
            desc=f"Rollout chunk {chunk_idx + 1}/{n_chunks}",
            leave=False,
            mininterval=tqdm_interval_sec,
        )

        done = False
        t    = 0
        if temporal_agg:
            all_time_actions = np.zeros(
                [n_envs, max_steps, max_steps + n_action_steps, action_dim]
            )

        while not done:
            np_obs_dict = dict(obs)

            if past_action and len(past_action_list) > 1:
                np_obs_dict["past_action"] = np.concatenate(past_action_list, axis=1)

            obs_dict = dict_apply(
                np_obs_dict,
                lambda x: torch.from_numpy(x).to(device=device),
            )

            with torch.no_grad():
                action_hat = model.predict_action({"obs": obs_dict})['pred_action']

            action = action_hat.to("cpu").numpy()

            if not np.all(np.isfinite(action)):
                raise RuntimeError("NaN or Inf detected in model action output.")

            env_action = action
            if temporal_agg:
                all_time_actions[:, [t], t : t + n_action_steps] = env_action[:, None, ...]
                actions_for_curr_step = all_time_actions[:, :, t]
                result = []
                for act_curr in actions_for_curr_step:
                    actions_populated = np.all(act_curr != 0, axis=1)
                    act_curr = act_curr[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(act_curr)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = exp_weights[:, None, ...]
                    result.append((act_curr * exp_weights).sum(axis=0, keepdims=True))
                env_action = np.array(result)
            else:
                env_action = env_action[:, :n_action_steps, ...]

            if abs_action:
                env_action = undo_transform_action(env_action, rotation_transformer)

            obs, reward, terminated, truncated, info = env.step(env_action)
            done = np.all(terminated | truncated)

            past_action_list.append(env_action)
            if len(past_action_list) > 2:
                past_action_list.pop(0)

            pbar.update(env_action.shape[1])
            t += env_action.shape[1]

        pbar.close()

        videos[this_global_slice]      = env.render()[this_local_slice]
        all_rewards[this_global_slice] = env.call("get_attr", "reward")[this_local_slice]

    # clear video buffer
    _ = env.reset()

    return videos, all_rewards


# ---------------------------------------------------------------------------
# Action transform helper  (identical to the original)
# ---------------------------------------------------------------------------

def undo_transform_action(action, rotation_transformer):
    raw_shape = action.shape
    if raw_shape[-1] == 20:
        action = action.reshape(-1, 2, 10)

    d_rot   = action.shape[-1] - 4
    pos     = action[..., :3]
    rot     = action[..., 3 : 3 + d_rot]
    gripper = action[..., [-1]]
    rot     = rotation_transformer.inverse(rot)
    uaction = np.concatenate([pos, rot, gripper], axis=-1)

    if raw_shape[-1] == 20:
        uaction = uaction.reshape(*raw_shape[:-1], 14)

    return uaction


# ---------------------------------------------------------------------------
# Video saving
# ---------------------------------------------------------------------------

def save_videos(videos, env_seeds, output_dir, fps):
    """Write MP4 files under <output_dir>/media/; return list of saved paths."""
    media_dir = pathlib.Path(output_dir) / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    all_video_paths = []
    for idx, video in enumerate(videos):
        if video is None:
            all_video_paths.append(None)
            continue
        path = media_dir / f"seed_{env_seeds[idx]:05d}.mp4"
        out = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (video.shape[2], video.shape[1]),  # (width, height)
        )
        for frame in video:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        all_video_paths.append(path)

    return all_video_paths


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(all_rewards, env_seeds, output_dir):
    """Compute statistics and write a plain-text report.

    Statistics
    ----------
    Per-episode : seed, max_reward, success (max_reward == 1.0)
    Aggregate   : mean / min / max of max_reward, success rate
    """
    per_episode = []
    for seed, rewards in zip(env_seeds, all_rewards):
        max_r   = float(np.max(rewards)) if rewards is not None else float("nan")
        success = max_r == 1.0
        per_episode.append({"seed": seed, "max_reward": max_r, "success": success})

    valid_rewards = [e["max_reward"] for e in per_episode
                     if not math.isnan(e["max_reward"])]
    n_total   = len(per_episode)
    n_success = sum(e["success"] for e in per_episode)

    mean_r = float(np.mean(valid_rewards)) if valid_rewards else float("nan")
    min_r  = float(np.min(valid_rewards))  if valid_rewards else float("nan")
    max_r  = float(np.max(valid_rewards))  if valid_rewards else float("nan")
    sr     = n_success / n_total * 100.0   if n_total > 0   else float("nan")

    lines = []
    lines.append("=" * 60)
    lines.append("ROLLOUT EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Total episodes : {n_total}")
    lines.append(f"Successful     : {n_success}  ({sr:.1f} %)")
    lines.append(f"Mean max-reward: {mean_r:.4f}")
    lines.append(f"Min  max-reward: {min_r:.4f}")
    lines.append(f"Max  max-reward: {max_r:.4f}")
    lines.append("")
    lines.append("-" * 60)
    lines.append(f"{'Episode':>8}  {'Seed':>8}  {'MaxReward':>10}  {'Success':>8}")
    lines.append("-" * 60)
    for ep_idx, entry in enumerate(per_episode):
        lines.append(
            f"{ep_idx:>8}  {entry['seed']:>8}  "
            f"{entry['max_reward']:>10.4f}  "
            f"{'YES' if entry['success'] else 'NO':>8}"
        )
    lines.append("=" * 60)

    report_path = pathlib.Path(output_dir) / "report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join(lines) + "\n"
    report_path.write_text(report_text)

    print(report_text)
    print(f"Report saved to: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python evaluate_rollout.py <config_name>\n"
            "  e.g. python evaluate_rollout.py eval_config\n"
            "       python evaluate_rollout.py eval_config.yaml",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Load config ---
    cfg = load_config(sys.argv[1])

    # Required fields
    checkpoint   = cfg.checkpoint
    dataset_path = os.path.expanduser(cfg.dataset)
    output_dir   = cfg.output_dir
    n_episodes   = cfg.n_episodes
    seed         = cfg.seed

    # Optional fields with defaults
    max_steps      = cfg.get("max_steps",      400)
    n_obs_steps    = cfg.get("n_obs_steps",    1)
    n_action_steps = cfg.get("n_action_steps", 8)
    render_obs_key = cfg.get("render_obs_key", "agentview_image")
    fps            = cfg.get("fps",            10)
    abs_action     = cfg.get("abs_action",     False)
    temporal_agg   = cfg.get("temporal_agg",   False)
    tqdm_interval  = cfg.get("tqdm_interval",  5.0)

    # shape_meta and model are always required
    shape_meta = OmegaConf.to_container(cfg.shape_meta, resolve=True)

    # --- Load model ---
    # hydra.utils.get_class resolves cfg.model._target_ to the actual class,
    # then load_from_checkpoint is called on the class (not an instance) so
    # that saved hyper-parameters and weights are restored correctly.
    print(f"Resolving model class: {cfg.model._target_}")
    model_class = hydra.utils.get_class(cfg.model._target_)
    print(f"Loading checkpoint: {checkpoint}")
    model = model_class.load_from_checkpoint(checkpoint, weights_only=False)
    model.eval()
    device = model.device

    # --- Environment metadata ---
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env_meta["env_kwargs"]["use_object_obs"] = False

    rotation_transformer = None
    if abs_action:
        env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
        rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")

    # --- Build env fns ---
    env_fn, dummy_env_fn = build_env_fns(
        env_meta=env_meta,
        shape_meta=shape_meta,
        render_obs_key=render_obs_key,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_steps=max_steps,
    )

    n_envs  = n_episodes  # always default to n_episodes
    env_fns = [env_fn] * n_envs

    # Build per-episode init functions (test-style: random seed, no fixed state)
    env_seeds         = []
    env_init_fn_dills = []

    for i in range(n_episodes):
        episode_seed  = seed + i
        enable_render = True  # render every episode for the video

        def init_fn(env, enable_render=enable_render):
            assert isinstance(env.env, VideoWrapper)
            env.env.enabled = enable_render
            assert isinstance(env.env.env, RobomimicImageWrapper)
            env.env.env.init_state = None  # random seed reset

        env_seeds.append(episode_seed)
        env_init_fn_dills.append(dill.dumps(init_fn))

    # --- Async vector env ---
    env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn, shared_memory=False)

    # --- Run rollout ---
    print(f"\nRunning {n_episodes} episode(s) with base seed {seed} ...")
    videos, all_rewards = run_rollout(
        model=model,
        env=env,
        env_fns=env_fns,
        env_seeds=env_seeds,
        env_init_fn_dills=env_init_fn_dills,
        max_steps=max_steps,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        past_action=False,
        abs_action=abs_action,
        temporal_agg=temporal_agg,
        action_dim=shape_meta["action"]["shape"][0],
        rotation_transformer=rotation_transformer,
        tqdm_interval_sec=tqdm_interval,
        device=device,
    )

    # --- Save videos ---
    print("\nSaving videos ...")
    save_videos(videos, env_seeds, output_dir, fps)

    # --- Write report ---
    print("\nGenerating report ...")
    write_report(all_rewards, env_seeds, output_dir)


if __name__ == "__main__":
    main()