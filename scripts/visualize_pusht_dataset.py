"""
visualize_pusht_dataset.py
──────────────────────────
Replays a trajectory from the PushT multitask dataset in PushTImageEnv,
using the dataset's recorded actions and matching the initial state exactly.

Usage
─────
uv run python scripts/visualize_pusht_dataset.py
uv run python scripts/visualize_pusht_dataset.py --episode 5 --output demo/pusht_ep5.mp4
"""

import argparse
import pathlib

import cv2
import numpy as np

from src.nn.data import ReplayBuffer
from src.nn.env.pusht_image_env import PushTImageEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/pusht_multitask")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index to replay (0-based).")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output", type=str, default=None,
                        help="Output path. Defaults to demo/pusht_dataset_ep{N}.mp4")
    args = parser.parse_args()

    rb = ReplayBuffer.copy_from_path(
        args.data_dir, keys=["img", "state", "action"]
    )
    n_episodes = rb.n_episodes
    assert 0 <= args.episode < n_episodes, \
        f"Episode {args.episode} out of range [0, {n_episodes - 1}]"

    # Resolve episode slice from episode_ends
    ends = rb.episode_ends  # end index (exclusive) for each episode
    ep_end = ends[args.episode]
    ep_start = 0 if args.episode == 0 else ends[args.episode - 1]
    imgs = rb["img"][ep_start:ep_end]         # (T, H, W, 3) uint8 — has correct goal rendered
    states = rb["state"][ep_start:ep_end]     # (T, 5)
    actions = rb["action"][ep_start:ep_end]   # (T, 2)
    print(f"Episode {args.episode}: {len(actions)} steps")

    env = PushTImageEnv(fix_goal=True)
    obs, _ = env.reset(options={"reset_to_state": states[0].tolist()})

    out_path = pathlib.Path(args.output or f"demo/pusht_dataset_ep{args.episode}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use dataset images for rendering: they contain the correct goal overlay
    # from the original collection environment, which env.render() cannot reproduce.
    h, w = imgs.shape[1], imgs.shape[2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (w, h),
    )
    writer.write(cv2.cvtColor(imgs[0], cv2.COLOR_RGB2BGR))

    for step, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        writer.write(cv2.cvtColor(imgs[step + 1] if step + 1 < len(imgs) else imgs[-1], cv2.COLOR_RGB2BGR))

        if terminated or truncated:
            print(f"Success at step {step + 1} (coverage: {info['coverage']:.2%})")
            break
    else:
        print(f"Finished all {len(actions)} steps (final coverage: {info['coverage']:.2%})")

    writer.release()
    env.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
