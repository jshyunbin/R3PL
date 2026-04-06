
import torch
import torch.nn as nn
from lightning import Callback, LightningModule, Trainer

import os
import wandb
import numpy as np
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import cv2
import wandb.sdk.data_types.video as wv
from src.nn.gym_util.async_vector_env import AsyncVectorEnv
from src.nn.gym_util.multistep_wrapper import MultiStepWrapper

from src.nn.gym_util.video_wrapper import VideoWrapper
from src.nn.common.rotation_transformer import RotationTransformer

from src.nn.common.pytorch_util import dict_apply
from src.nn.env.robomimic_image_wrapper import (
    RobomimicImageWrapper,
)
import src.robothink.envs
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


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


class RobomimicRunner(Callback):
    """
    Robomimic gym environment runner for PyTorch Lightning.
    
    Rollouts actions and logs rewards and videos to wandb.
    """
    
    def __init__(
        self, 
        output_dir,
        dataset_path,
        shape_meta: dict,
        every_n_epoch: int=1,
        n_train=10,
        n_train_vis=3,
        train_start_idx=0,
        n_test=22,
        n_test_vis=6,
        test_start_seed=10000,
        max_steps=400,
        n_obs_steps=1,
        n_action_steps=8,
        render_obs_key="agentview_image",
        fps=10,
        past_action=False,
        abs_action=False,
        tqdm_interval_sec=5.0,
        n_envs=None,
        temporal_agg=False
    ):
        super().__init__()

        if n_envs is None:
            n_envs = n_train + n_test

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = 1

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        # disable object state observation
        env_meta["env_kwargs"]["use_object_obs"] = False

        rotation_transformer = None
        if abs_action:
            env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
            rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")

        def env_fn():
            robomimic_env = create_env(env_meta=env_meta, shape_meta=shape_meta)
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
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

        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
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

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, "r") as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f"data/demo_{train_idx}/states"][0]

                def init_fn(env, init_state=init_state, enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoWrapper)
                    env.env.enabled = enable_render

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicImageWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append("train_")
                env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoWrapper)
                env.env.enabled = enable_render

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicImageWrapper)
                env.env.env.init_state = None

            env_seeds.append(seed)
            env_prefixs.append("test_")
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn, shared_memory=False)

        self.output_dir = output_dir
        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.temporal_agg = temporal_agg
        self.action_dim = shape_meta['action']['shape'][0]
        self.every_n_epoch = every_n_epoch
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if trainer.current_epoch % self.every_n_epoch != 0:
            return

        device = pl_module.device
        dtype = pl_module.dtype
        env = self.env
        epoch = pl_module.current_epoch

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        videos = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each("run_dill_function", args_list=[(x, ) for x in this_init_fns])

            # start rollout
            obs, _ = env.reset(seed=self.env_seeds)
            # past_action = None
            past_action_list = []
            # pl_module.reset()

            env_name = self.env_meta["env_name"]
            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            done = False
            t = 0
            if self.temporal_agg:
                all_time_actions = np.zeros([n_envs, self.max_steps, self.max_steps+self.n_action_steps, 10])

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
                    action_hat = pl_module.predict_action({'obs': obs_dict})['pred_action']

                # device_transfer
                action = action_hat.to("cpu").numpy()

                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")

                # step env
                env_action = action
                if self.temporal_agg:
                    all_time_actions[:, [t], t:t+self.n_action_steps] = env_action[:, None, ...] # (batch, 1, n_action_steps, 10)
                    actions_for_curr_step = all_time_actions[:, :, t] # (batch, time, action_dim)
                    result = list()
                    for act_curr in actions_for_curr_step:
                        actions_populated = np.all(act_curr != 0, axis=1) # (time)
                        act_curr = act_curr[actions_populated] # (selected_time, action_dim)
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(act_curr)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = exp_weights[:, None, ...]
                        result.append((act_curr * exp_weights).sum(axis=0, keepdims=True))
                    env_action = np.array(result) # (batch, 1, action_dim)
                else:
                    env_action = env_action[:, :self.n_action_steps, ...] # truncate
                if self.abs_action:
                    env_action = self.undo_transform_action(env_action)

                obs, reward, terminated, truncated, info = env.step(env_action)
                done = terminated | truncated
                done = np.all(done)

                # past_action = action
                past_action_list.append(env_action)
                if len(past_action_list) > 2:
                    past_action_list.pop(0)

                # update pbar
                pbar.update(env_action.shape[1])
                t += env_action.shape[1]
            pbar.close()

            # collect data for this round
            videos[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call("get_attr", "reward")[
                this_local_slice
            ]

        # clear out video buffer
        _ = env.reset()

        all_video_paths = list()
        # save videos
        for idx, video in enumerate(videos):
            if video is None:
                continue
            path = pathlib.Path(self.output_dir).joinpath(f'media/epoch{epoch:04d}_{self.env_seeds[idx]:05d}.mp4')
            path.parent.mkdir(exist_ok=True)
            all_video_paths.append(path)
            out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (video.shape[2], video.shape[1]))
            for image in video:
                out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            out.release()

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
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            pl_module.log(f"val/{prefix}sim_max_reward_{seed}", max_reward, prog_bar=True, on_epoch=True)
    

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix + "mean_score"
            value = np.mean(value)
            pl_module.log(f"val/{name}", value, on_epoch=True, prog_bar=True)
        
        for idx, path in enumerate(all_video_paths):
            if path is None:
                continue
            wandb.log({f"val/video_{idx}": wandb.Video(path, format="mp4")})

    
    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1, 2, 10)

        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3 : 3 + d_rot]
        gripper = action[..., [-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction

if __name__ == "__main__":
    from src.nn.models.actrm import ACTRMModule
    model = ACTRMModule.load_from_checkpoint("checkpoints/epoch_250-0.00.ckpt", weights_only=False)

    shape_meta = {
        "image_resolution": 84,
        "action": {
            "shape": [10],
        },
        "obs": {
            "robot0_eef_pos": {
                "shape": [3]
            },
            "robot0_eef_quat": {
                "shape": [4]
            },
            "robot0_joint_pos": {
                "shape": [7]
            },
            "robot0_eye_in_hand_image": {
                "shape": [3, 84, 84],
                "type": 'rgb'
            },
            "robot0_gripper_qpos": {
                "shape": [2]
            },
            "agentview_image": {
                "shape": [3, 84, 84],
                "type": 'rgb'
            },
        }
    }

    runner = RobomimicRunner(
        'demo/temp_agg',
        'data/robomimic/can/ph/image_abs.hdf5',
        shape_meta,
        n_train=10,
        n_train_vis=10,
        train_start_idx=0,
        n_test=0,
        n_test_vis=0,
        test_start_seed=10000,
        max_steps=400,
        n_obs_steps=1,
        n_action_steps=32,
        render_obs_key="agentview_image",
        fps=20,
        past_action=False,
        abs_action=True,
        tqdm_interval_sec=5.0,
        n_envs=None,
        temporal_agg=True
    )

    runner.on_train_epoch_end(None, model)