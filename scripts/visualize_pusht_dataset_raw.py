"""
visualize_pusht_dataset_raw.py
───────────────────────────────
Writes a video directly from the image observations stored in the
PushT multitask dataset zarr, with no environment involved.

Usage
─────
uv run python scripts/visualize_pusht_dataset_raw.py
uv run python scripts/visualize_pusht_dataset_raw.py --episode 5 --output demo/pusht_raw_ep5.mp4
"""

import argparse
import pathlib

import cv2
import numpy as np

from src.nn.data import ReplayBuffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/pusht_multitask")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    rb = ReplayBuffer.copy_from_path(args.data_dir, keys=["img"])

    ends = rb.episode_ends
    assert 0 <= args.episode < rb.n_episodes, \
        f"Episode {args.episode} out of range [0, {rb.n_episodes - 1}]"

    ep_end = ends[args.episode]
    ep_start = 0 if args.episode == 0 else ends[args.episode - 1]
    imgs = rb["img"][ep_start:ep_end]  # (T, H, W, 3) uint8

    out_path = pathlib.Path(args.output or f"demo/pusht_raw_ep{args.episode}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = imgs.shape[1], imgs.shape[2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (w, h),
    )
    for frame in imgs:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    print(f"Episode {args.episode}: {len(imgs)} frames → {out_path}")


if __name__ == "__main__":
    main()
