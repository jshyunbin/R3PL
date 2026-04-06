import collections
import math
import pathlib

import dill
import numpy as np
import torch
import tqdm
import wandb
from lightning import Callback, LightningModule, Trainer

from src.nn.common.pytorch_util import dict_apply
from src.nn.env.pusht import PushTImageEnv
from src.nn.gym_util.async_vector_env import AsyncVectorEnv
from src.nn.gym_util.multistep_wrapper import MultiStepWrapper
from src.nn.gym_util.video_recording_wrapper import VideoRecordingWrapper
from src.nn.gym_util.video_recorder import VideoRecorder


class PushTMultigoalRunner(Callback):
    """Evaluation callback for the multigoal PushT environment.

    Runs rollouts using PushTImageEnv with fix_goal=False so each episode
    samples a different goal pose. Logs per-seed max rewards and videos to wandb.
    Follows the same structure as RobomimicRunner.
    """

    def __init__(
        self,
        output_dir: str,
        every_n_epoch: int = 10,
        n_train: int = 6,
        n_train_vis: int = 2,
        train_start_seed: int = 0,
        n_test: int = 22,
        n_test_vis: int = 6,
        test_start_seed: int = 10000,
        max_steps: int = 300,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        render_size: int = 96,
        fps: int = 10,
        crf: int = 22,
        tqdm_interval_sec: float = 5.0,
        n_envs: int = None,
    ):
        super().__init__()

        n_inits = n_train + n_test
        if n_envs is None:
            n_envs = n_inits

        steps_per_render = 1

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTImageEnv(fix_goal=False, render_size=render_size),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1,
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
            )

        env_fns = [env_fn] * n_envs
        env_seeds = []
        env_prefixs = []
        env_render_flags = []

        for i in range(n_train):
            env_seeds.append(train_start_seed + i)
            env_prefixs.append("train_")
            env_render_flags.append(i < n_train_vis)

        for i in range(n_test):
            env_seeds.append(test_start_seed + i)
            env_prefixs.append("test_")
            env_render_flags.append(i < n_test_vis)

        env = AsyncVectorEnv(env_fns, shared_memory=False)

        self.output_dir = output_dir
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_render_flags = env_render_flags
        self.fps = fps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.every_n_epoch = every_n_epoch

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if trainer.current_epoch % self.every_n_epoch != 0:
            return

        device = pl_module.device
        epoch = trainer.current_epoch
        env = self.env

        n_envs = len(self.env_fns)
        n_inits = len(self.env_seeds)
        n_chunks = math.ceil(n_inits / n_envs)

        # Build init functions fresh each epoch so video paths encode the epoch number
        env_init_fn_dills = []
        for seed, enable_render in zip(self.env_seeds, self.env_render_flags):
            video_path = None
            if enable_render:
                video_path = str(
                    pathlib.Path(self.output_dir)
                    / f"media/epoch{epoch:04d}_{seed:05d}.mp4"
                )

            def init_fn(env, fp=video_path):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = fp
                if fp is not None:
                    pathlib.Path(fp).parent.mkdir(parents=True, exist_ok=True)

            env_init_fn_dills.append(dill.dumps(init_fn))

        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns = this_init_fns + [env_init_fn_dills[0]] * n_diff
            assert len(this_init_fns) == n_envs

            # Initialize environments
            env.call_each("run_dill_function", args_list=[(x,) for x in this_init_fns])

            # Reset with per-environment seeds
            chunk_seeds = self.env_seeds[this_global_slice]
            chunk_seeds_padded = list(chunk_seeds) + [self.env_seeds[0]] * n_diff
            obs, _ = env.reset(seed=chunk_seeds_padded)

            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval PushTMultigoal {chunk_idx+1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            done = False
            while not done:
                np_obs_dict = dict(obs)

                obs_dict = dict_apply(
                    np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device),
                )

                with torch.no_grad():
                    action = pl_module.predict_action({"obs": obs_dict})["pred_action"]

                action = action.detach().cpu().numpy()

                if not np.all(np.isfinite(action)):
                    raise RuntimeError(f"Nan or Inf action: {action}")

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated | truncated
                done = np.all(done)

                pbar.update(action.shape[1])

            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call("get_attr", "reward")[
                this_local_slice
            ]

        # Reset to clear video buffers
        _ = env.reset()

        # Log metrics
        max_rewards = collections.defaultdict(list)
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            pl_module.log(
                f"val/{prefix}sim_max_reward_{seed}",
                max_reward,
                prog_bar=False,
                on_epoch=True,
            )

        for prefix, values in max_rewards.items():
            name = prefix + "mean_score"
            pl_module.log(
                f"val/{name}", np.mean(values), on_epoch=True, prog_bar=True
            )

        for i, path in enumerate(all_video_paths):
            if path is None:
                continue
            wandb.log({f"val/video_{self.env_seeds[i]}": wandb.Video(path, format="mp4")})
