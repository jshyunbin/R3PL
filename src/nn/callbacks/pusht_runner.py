
import torch
import torch.nn as nn
from lightning import Callback, LightningModule, Trainer

import os
import wandb
import numpy as np
import collections
import tqdm
import gymnasium as gym
from src.nn.gym_util.async_vector_env import AsyncVectorEnv
from src.nn.gym_util.multistep_wrapper import MultiStepWrapper

from src.nn.gym_util.video_recording_wrapper import VideoRecordingWrapper
from src.nn.gym_util.video_recorder import VideoRecorder

from src.nn.common.pytorch_util import dict_apply


class PushTRunner(Callback):
    """
    Robomimic gym environment runner for PyTorch Lightning.
    
    Rollouts actions and logs rewards and videos to wandb.
    """
    
    def __init__(
        self, 
        output_dir,
        dataset_path,
        n_test=30,
        n_test_vis=5,
        test_start_seed=0,
        max_steps=400,
        n_obs_steps=1,
        n_action_steps=8,
        fps=10,
        crf=22,
        past_action=False,
        tqdm_interval_sec=5.0,
        n_envs=None,
    ):
        super().__init__()

        if n_envs is None:
            n_envs = n_test

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        def env_fn():
            env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos")
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    env,
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
        env_seeds = list()

        for i in range(n_test):
            env_seeds.append(test_start_seed + i)

        env = AsyncVectorEnv(env_fns, shared_memory=False)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
    
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        device = pl_module.device
        dtype = pl_module.dtype
        env = self.env
        epoch = pl_module.current_epoch

        # plan for rollout
        n_envs = len(self.env_fns)

        # allocate data
        all_video_paths = [None] * n_envs
        all_rewards = [None] * n_envs

        this_init_fns = self.env_init_fn_dills
        n_diff = n_envs - len(this_init_fns)
        if n_diff > 0:
            this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
        assert len(this_init_fns) == n_envs

        # start rollout
        obs, _ = env.reset(seed=self.env_seeds)
        # past_action = None
        past_action_list = []
        # pl_module.reset()

        pbar = tqdm.tqdm(
            total=self.max_steps,
            desc=f"Eval PushT Image",
            leave=False,
            mininterval=self.tqdm_interval_sec,
        )

        done = False

        while not done:
            # create obs dict
            np_obs_dict = dict(obs)

            if self.past_action:
                if len(past_action_list) > 1:  ## get 16 actions
                    np_obs_dict["past_action"] = np.concatenate(
                        past_action_list, axis=1
                    )

            # device transfer
            obs_dict = dict_apply(
                np_obs_dict, lambda x: torch.from_numpy(x).to(device=device)
            )

            # run policy
            with torch.no_grad():
                action_dict = pl_module.predict_action({'obs': obs_dict})

            # device_transfer
            np_action_dict = dict_apply(
                action_dict, lambda x: x.detach().to("cpu").numpy()
            )

            action = np_action_dict["action"]
            if not np.all(np.isfinite(action)):
                print(action)
                raise RuntimeError("Nan or Inf action")

            # step env
            env_action = action
            if self.abs_action:
                env_action = self.undo_transform_action(action)

            obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated | truncated
            done = np.all(done)

            # past_action = action
            past_action_list.append(action)
            if len(past_action_list) > 2:
                past_action_list.pop(0)

            # update pbar
            pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths = env.render()
            all_rewards = env.call("get_attr", "reward")

        # clear out video buffer
        _ = env.reset()

        # log
        max_rewards = collections.defaultdict(list)
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_envs):
            seed = self.env_seeds[i]
            max_reward = np.max(all_rewards[i])
            pl_module.log(f"val/sim_max_reward_{seed}", max_reward, prog_bar=True, on_epoch=True)


        # log aggregate metrics
        name = "mean_score"
        value = np.mean(value)
        pl_module.log(f"val/{name}", value, on_epoch=True, prog_bar=True)
        
        for idx, path in enumerate(all_video_paths):
            if path is None:
                continue
            wandb.log({f"val/video_{idx}": wandb.Video(path, format="mp4")})
