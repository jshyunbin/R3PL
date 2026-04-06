"""
visualize_pusht_random.py
─────────────────────────
Rolls out the PushTImageEnv with random actions and saves a video to demo/.

Usage
─────
uv run python scripts/visualize_pusht_random.py
uv run python scripts/visualize_pusht_random.py --fix_goal --seed 7 --max_steps 200
"""

import argparse
import pathlib

import cv2
import numpy as np

from src.nn.env.pusht_image_env import PushTImageEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--fix_goal", action="store_true", default=False,
                        help="Use fixed goal (single-goal mode). Default: multigoal.")
    parser.add_argument("--output", type=str, default="demo/pusht_random.mp4")
    args = parser.parse_args()

    env = PushTImageEnv(fix_goal=args.fix_goal)
    obs, _ = env.reset(seed=args.seed)

    # Collect frames at visualization resolution
    frame = env.render()
    h, w = frame.shape[:2]

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (w, h),
    )
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    rng = np.random.default_rng(args.seed)
    for step in range(args.max_steps):
        action = rng.uniform(0, 512, size=(2,)).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        frame = env.render()
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if terminated or truncated:
            print(f"Done at step {step + 1} (coverage: {info['coverage']:.2%})")
            break
    else:
        print(f"Finished {args.max_steps} steps.")

    writer.release()
    env.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
