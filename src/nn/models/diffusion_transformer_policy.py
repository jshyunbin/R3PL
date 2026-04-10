from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from lightning import LightningModule

from src.nn.modules import (
    TransformerForDiffusion,
    LowdimMaskGenerator,
    LinearNormalizer,
)
from src.nn.modules.utils import compute_lr
from src.nn.vision.model_getter import get_resnet
from src.nn.common.pytorch_util import dict_apply, replace_submodules
from src.nn.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class DiffusionTransformerPolicyModule(LightningModule):
    def __init__(
        self,
        shape_meta: dict,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        beta_start: float = 0.0001,
        clip_sample: bool = True,
        num_train_timesteps: int = 100,
        prediction_type: str = "epsilon",
        variance_type: str = "fixed_small",
        num_inference_steps: Optional[int] = None,
        # transformer architecture
        n_layer: int = 8,
        n_head: int = 4,
        n_emb: int = 256,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.3,
        causal_attn: bool = True,
        n_cond_layers: int = 0,
        # training
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        warmup_steps: int = 1000,
        lr_min_ratio: float = 0.0,
        rollout_during_val: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = 512  # ResNet18 output

        # agent_pos is 2D; cond_dim = image_features + agent_pos
        low_dim_obs_size = shape_meta["obs"]["agent_pos"]["shape"][0]
        cond_dim = obs_feature_dim + low_dim_obs_size  # 512 + 2 = 514

        self.denoise_net = TransformerForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            n_cond_layers=n_cond_layers,
        )

        self.obs_encoder = get_resnet(name="resnet18")
        self.obs_encoder = replace_submodules(
            self.obs_encoder,
            predicate=lambda m: isinstance(m, nn.BatchNorm2d),
            func=lambda m: nn.GroupNorm(
                num_groups=m.num_features // 16, num_channels=m.num_features
            ),
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
        self.cond_dim = cond_dim

        if num_inference_steps is None:
            num_inference_steps = num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.manual_step = 0

    def setup(self, stage: str):
        self.mask_generator = LowdimMaskGenerator(
            action_dim=self.action_dim,
            obs_dim=0,
            max_n_obs_steps=self.n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
            device=self.device,
        )

        if stage == "fit":
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

    # ========= inference ============
    def _encode_obs(self, nobs: dict, B: int) -> torch.Tensor:
        """Returns cond tensor (B, To, cond_dim)."""
        To = self.n_obs_steps
        this_nobs = dict_apply(
            nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs["image"])  # (B*To, 512)
        cond = nobs_features.reshape(B, To, -1)               # (B, To, 512)
        cond = torch.cat([cond, nobs["agent_pos"][:, :To]], dim=-1)  # (B, To, 514)
        return cond

    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        generator=None,
    ) -> torch.Tensor:
        scheduler = self.noise_scheduler
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = self.denoise_net(trajectory, t, cond=cond)
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict
        nobs = self.normalizer.normalize(obs_dict["obs"])
        value = next(iter(nobs.values()))
        B = value.shape[0]
        T = self.horizon
        Da = self.action_dim

        device = self.device
        dtype = self.dtype

        cond = self._encode_obs(nobs, B)
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        nsample = self.conditional_sample(cond_data, cond_mask, cond=cond)

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {
            "action": action,
            "action_pred": action_pred,
            "pred_action": action,
        }

    # ========= training ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss_and_metrics(self, batch: Dict[str, torch.Tensor]):
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]

        cond = self._encode_obs(nobs, batch_size)
        trajectory = nactions

        condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=trajectory.device,
        ).long()

        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        pred = self.denoise_net(noisy_trajectory, timesteps, cond=cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()

        return loss, {"loss": loss.detach()}

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        batch_size = batch["action"].shape[0]
        opt = self.optimizers()

        loss, metrics = self.compute_loss_and_metrics(batch)
        self.manual_backward(loss)

        if batch_idx % 50 == 0:
            self._grad_monitoring()

        current_step = self.manual_step
        base_lr = self.hparams.learning_rate

        lr_this_step = compute_lr(
            base_lr=base_lr,
            lr_warmup_steps=self.hparams.warmup_steps,
            lr_min_ratio=self.hparams.lr_min_ratio,
            current_step=current_step,
            total_steps=self.total_steps,
        )

        for param_group in opt.param_groups:
            param_group["lr"] = lr_this_step
        opt.step()
        opt.zero_grad()

        self.log("train/lr", lr_this_step, on_step=True)
        self.log("train/loss", metrics.get("loss", 0), on_step=True)
        self.manual_step += 1
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        with torch.no_grad():
            loss, metrics = self.compute_loss_and_metrics(batch)
            for name, value in metrics.items():
                self.log(f"val/{name}", value, on_step=False, on_epoch=True)
            return metrics

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def _grad_monitoring(self):
        with torch.no_grad():
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), max_norm=float("inf")
            ).item()
            self.log("grad/total_norm", total_grad_norm, on_step=True)
            if total_grad_norm < 1e-6 or total_grad_norm > 100:
                log.warning(f"Step {self.manual_step}: Gradient norm={total_grad_norm:.2e}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95),
        )
        return optimizer
