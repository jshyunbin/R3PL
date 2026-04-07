import os
from typing import Dict, Optional
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import gymnasium as gym
import gym_pusht

from lightning import LightningModule

from src.nn.modules import (
    ConditionalUnet1D,
    LowdimMaskGenerator,
    LinearNormalizer
)
from src.nn.modules.utils import compute_lr
from src.nn.vision.model_getter import get_resnet
from src.nn.common.pytorch_util import dict_apply, replace_submodules
from src.nn.utils import RankedLogger
import wandb

log = RankedLogger(__name__, rank_zero_only=True)

class DiffusionPolicyModule(LightningModule):
    def __init__(self, 
            shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            beta_start=0.0001,
            clip_sample=True,
            num_train_timesteps=100,
            prediction_type='epsilon',
            variance_type='fixed_small',
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            learning_rate=1e-4,
            weight_decay=1e-6,
            warmup_steps=500,
            lr_min_ratio=0.0,
            rollout_during_val=True,
            # parameters passed to step
            **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # CRITICAL: Manual optimization
        self.automatic_optimization = False

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = 512

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        # Use FiLM conditioning
        if obs_as_global_cond:
            global_cond_dim = input_dim * n_obs_steps
            input_dim = action_dim

        self.denoise_net = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.obs_encoder = get_resnet(name='resnet18')
        # replace batchnorm with groupnorm
        self.obs_encoder = replace_submodules(
            self.obs_encoder, 
            predicate=lambda m: isinstance(m, nn.BatchNorm2d),
            func=lambda m: nn.GroupNorm(num_groups=m.num_features//16, num_channels=m.num_features)
        )


        self.noise_scheduler = DDPMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            num_train_timesteps=num_train_timesteps,
            clip_sample=clip_sample,
            prediction_type=prediction_type,
            variance_type=variance_type,
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
            
        self.manual_step = 0

    def setup(self, stage: str):
        """Called by Lightning when setting up the model."""
        self.mask_generator = LowdimMaskGenerator(
            action_dim=self.action_dim,
            obs_dim=0 if self.obs_as_global_cond else self.obs_feature_dim,
            max_n_obs_steps=self.n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
            device=self.device
        )

        if stage == "fit":
            # Calculate steps from dataset and epochs
            dm = self.trainer.datamodule
        
            samples_per_epoch = len(dm.train_dataset)
            
            steps_per_epoch = samples_per_epoch // dm.batch_size
            
            if self.trainer.max_epochs > 0:
                self.total_steps = steps_per_epoch * self.trainer.max_epochs
            else:
                self.total_steps = float("inf")

            log.info("Training configuration:")
            log.info(f"  Episodes: {len(dm.train_dataset)}")
            log.info(f"  Steps per epoch: {steps_per_epoch}")
            log.info(f"  Total steps: {self.total_steps}")

    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = self.denoise_net(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                # **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict['obs'])
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs['image'])
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, To, -1)
            global_cond = torch.cat([global_cond, nobs['agent_pos'][:, :To,...]], dim=-1)
            global_cond = global_cond.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred,
            'pred_action': action,  # alias for compatibility with runner callbacks
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss_and_metrics(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            
            nobs_features = self.obs_encoder(this_nobs['image'])
            global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            global_cond = torch.cat([global_cond, nobs['agent_pos'][:, :self.n_obs_steps,...]], dim=-1)
            global_cond = global_cond.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the trajectory
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each trajectory
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean trajectory according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps
        )
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.denoise_net(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        metrics = {
            'loss': loss.detach()
        }
        return loss, metrics
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        Training step of the diffusion policy model
        """
        batch_size = batch['action'].shape[0]
        # Handle case when not attached to trainer (for testing)
        try:
            opt = self.optimizers()
        except RuntimeError:
            # For testing without trainer
            if not hasattr(self, "_optimizer"):
                raise RuntimeError("No optimizer available. Set model._optimizer for testing.")
            log.warning("No trainer attached, using manually set optimizer.")
            opt = self._optimizer
        

        loss, metrics = self.compute_loss_and_metrics(batch)
        self.manual_backward(loss)

        if batch_idx % 50 == 0:
            self.grad_monitoring()

        current_step = self.manual_step
        base_lr = self.hparams.learning_rate

        if current_step < self.hparams.warmup_steps:
            lr_this_step = compute_lr(
                base_lr=base_lr,
                lr_warmup_steps=self.hparams.warmup_steps,
                lr_min_ratio=self.hparams.lr_min_ratio,
                current_step=current_step,
                total_steps=self.total_steps,
            )
        else:
            lr_this_step = base_lr

        for param_group in opt.param_groups:
            param_group['lr'] = lr_this_step
        opt.step()
        opt.zero_grad()
        
        self.log_metrics(metrics, lr_this_step, batch_size=batch_size)

        self.manual_step += 1

        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        batch_size = batch['action'].shape[0]

        with torch.no_grad():
            loss, metrics = self.compute_loss_and_metrics(batch)

            for name, value in metrics.items():
                self.log(f"val/{name}", value, on_step=False, on_epoch=True)

            return metrics
 
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        with torch.no_grad():
            if self.hparams.rollout_during_val:
                log.info("Validating model by rollout...")

                env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos")
                obs, info = env.reset()

                images = np.array([obs['pixels'], obs['pixels']])
                agent_pos = np.array([obs['agent_pos'], obs['agent_pos']])

                imgs = [env.render()]
                # observation has keys 'pixels' and 'agent_pos'

                rewards = list()

                for _ in range(100):
                    observation = {
                        'obs': {
                            'image': rearrange(torch.Tensor(images[-2:] / 255.0), 't x y c -> 1 t c x y'),
                            'agent_pos': torch.Tensor(agent_pos[-2:]).unsqueeze(0)
                        }
                    }
                    result = self.predict_action(observation)
                    action = result['action'][0].detach().cpu()
                    for i in range(len(action)):
                        obs, reward, terminated, truncated, info = env.step(action[i])
                        images = np.concatenate((images, np.expand_dims(obs['pixels'], 0)), axis=0)
                        agent_pos = np.concatenate((agent_pos, np.expand_dims(obs['agent_pos'], 0)), axis=0)

                        imgs.append(env.render())
                        rewards.append(reward)
                
                    if terminated or truncated:
                        break
                
                self.log(f'val/reward', max(rewards), on_step=False, on_epoch=True, prog_bar=True)
                wandb.log({"val/vid": wandb.Video(rearrange(np.array(imgs), "t x y c -> t c x y"), format="mp4", fps=20)})
                env.close()


    def grad_monitoring(self):
        with torch.no_grad():

            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), 
                max_norm = float('inf')
            ).item()

            # grad_metrics = {}

            # if self.denoise_net is not None:
            #     grad_metrics['unet'] = self.denoise_net.weight.grad.norm().item()

            # if self.obs_encoder is not None:
            #     grad_metrics['vision_encoder'] = self.obs_encoder.weight.grad.norm().item() 
            self.log('grad/total_norm', total_grad_norm, on_step=True)

            # Optional: log individual components
            # for name, value in grad_metrics.items():
            #     self.log(f'grad/{name}', value, on_step=True)

            # Warning for problematic gradients
            if total_grad_norm < 1e-6 or total_grad_norm > 100:
                log.warning(f"Step {self.manual_step}: Gradient norm={total_grad_norm:.2e}")

    def log_metrics(self, metrics: dict, lr_this_step: float = None, batch_size: int = None):

        # Log learning rate (will log the last optimizer's LR)
        self.log("train/lr", lr_this_step, on_step=True)

        # Log metrics
        with torch.no_grad():
            self.log("train/loss", metrics.get("loss", 0), on_step=True)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


if __name__=='__main__':
    # test model
    # model = DiffusionPolicyModule(
    #     shape_meta={
    #         'action': {'shape': (2,)},
    #         'obs': {
    #             'agent_pos': {'shape': (2,)},
    #             'image': {
    #                 'shape': (3,94,94),
    #             }
    #         }
    #     },
    #     horizon=16,
    #     n_action_steps=8,
    #     n_obs_steps=2,
    #     prediction_type='epsilon',
    #     variance_type='fixed_small',
    #     obs_as_global_cond=True,
    #     diffusion_step_embed_dim=256,
    #     down_dims=[256,512,1024],
    #     kernel_size=5,
    #     n_groups=8,
    #     cond_predict_scale=True,
    #     learning_rate=1e-4,
    #     weight_decay=1e-6,
    # )

    # model._optimizer = model.configure_optimizers()

    from src.nn.data.pusht_datamodule import PushtDataModule

    dm = PushtDataModule(
        data_dir="data/pusht/pusht_cchi_v7_replay.zarr", 
        val_ratio=0.1, 
        test_ratio=0.1,
        pad_before=1,
        pad_after=7,
        batch_size=32, 
        horizon=16, 
        num_workers=0
    )
    batch = next(iter(dm.train_dataloader()))
    print(f"{batch['obs']['image'].shape=}, {batch['obs']['image'][0, 0, 0, :5, :5]}")
    print(f"{batch['obs']['agent_pos'].shape=}, {batch['obs']['agent_pos'][0, 0, :]}")
    # Outputs
    # batch['obs']['image'].shape=torch.Size([32, 16, 3, 96, 96]), tensor([[1.0000, 0.9725, 0.9725, 0.9725, 0.9725],
    #         [0.9725, 0.8706, 0.9137, 0.9137, 0.9137],
    #         [0.9686, 0.9137, 1.0000, 1.0000, 1.0000],
    #         [0.9686, 0.9137, 1.0000, 1.0000, 1.0000],
    #         [0.9725, 0.9137, 1.0000, 1.0000, 1.0000]])
    # batch['obs']['agent_pos'].shape=torch.Size([32, 16, 2]), tensor([252.2717, 152.1889])


    # with torch.no_grad():
    #     model.set_normalizer(dm.train_dataset.get_normalizer())
    #     dm.setup("fit")
    #     model.total_steps = 100*167
    #     batch = next(iter(dm.train_dataloader()))
    #     result = model.predict_action(batch)
    #     print(f"{result['action'].shape=}, {result['action_pred'].shape=}")

    #     loss, metric = model.compute_loss_and_metrics(batch)
    #     print(f"{loss=}, {metric=}")

    # loss = model.training_step(batch, 1)
    # print(f"Training step {loss=}")


    from tqdm import tqdm


    model = DiffusionPolicyModule.load_from_checkpoint('checkpoints/diffusion_policy.ckpt', weights_only=False)

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos")
    obs, info = env.reset()

    images = np.array([obs['pixels'], obs['pixels']])
    agent_pos = np.array([obs['agent_pos'], obs['agent_pos']])
    print(f"{images.shape=}, {rearrange(torch.Tensor(images[-2:] / 255.0), 't x y c -> t c x y')[0, 0, :5, :5]}")
    print(f"{agent_pos.shape=}, {agent_pos[0, :]}")
    # Outputs
    # images.shape=(2, 96, 96, 3), tensor([[1.0000, 0.9725, 0.9725, 0.9725, 0.9725],
    #         [0.9725, 0.8706, 0.9137, 0.9137, 0.9137],
    #         [0.9686, 0.9137, 1.0000, 1.0000, 1.0000],
    #         [0.9686, 0.9137, 1.0000, 1.0000, 1.0000],
    #         [0.9725, 0.9137, 1.0000, 1.0000, 1.0000]])
    # agent_pos.shape=(2, 2), [194. 242.]
    
    imgs = [env.render()]
    # observation has keys 'pixels' and 'agent_pos'

    with torch.no_grad():
        for _ in tqdm(range(30)):
            observation = {
                'obs': {
                    'image': rearrange(torch.Tensor(images[-2:] / 255.0), 't x y c -> 1 t c x y'),
                    'agent_pos': torch.Tensor(agent_pos[-2:]).unsqueeze(0)
                }
            }
            result = model.predict_action(observation)
            action = result['action'][0].cpu()
            for i in range(len(action)):
                obs, reward, terminated, truncated, info = env.step(action[i])
                images = np.concatenate((images, np.expand_dims(obs['pixels'], 0)), axis=0)
                agent_pos = np.concatenate((agent_pos, np.expand_dims(obs['agent_pos'], 0)), axis=0)

                imgs.append(env.render())
        
            if terminated or truncated:
                obs, info = env.reset()
                images = np.array([obs['pixels'], obs['pixels']])
                agent_pos = np.array([obs['agent_pos'], obs['agent_pos']])
                imgs.append(env.render())
    
        env.close()

    import cv2
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (len(imgs[0][0]), len(imgs[0][0])))

    for img in imgs:
        out.write(img)
    out.release()