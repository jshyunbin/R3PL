from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
import math
from torchvision.models._utils import IntermediateLayerGetter
import torchvision.transforms as transforms

from lightning import LightningModule

from src.nn.modules import (
    LinearNormalizer,
    get_sinusoid_encoding_table,
    PositionEmbeddingSine
)
from src.nn.modules.trm_block import (
    Transformer,
    TransformerDecoder,
    CastedLinear,
    CastedEmbedding
)
from src.nn.vision.model_getter import get_resnet
from src.nn.common.pytorch_util import dict_apply, replace_submodules, reparametrize
from src.nn.modules.utils import compute_lr
from src.nn.utils import RankedLogger
from src.nn.losses.kl_divergence import kl_divergence

log = RankedLogger(__name__, rank_zero_only=True)

class ACTModule(LightningModule):
    def __init__(self, 
            shape_meta: dict,
            horizon, 
            n_heads=8,
            num_enc_layers=6,
            num_dec_enc_layers=4,
            num_dec_dec_layers=7,
            dropout=0.1,
            kl_weight: float = 10.0,
            hidden_dim: int = 512,
            latent_dim: int = 32,
            feedforward_dim: int = 3200,
            learning_rate=1e-4,
            weight_decay=1e-6,
            warmup_steps=500,
            lr_min_ratio=0.0,
            forward_dtype=None,
            pretrained_backbone=True,
            # parameters passed to step
            **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # CRITICAL: Manual optimization
        self.automatic_optimization = False
        
        if forward_dtype is not None:
            self.forward_dtype = forward_dtype
            print(f"Manually casting ACT Module to {self.forward_dtype}")
        else:
            # cast according to device
            if torch.backends.mps.is_available():
                log.info("MPS (Mac) detected. Forcing forward_dtype to float32 to avoid NaNs.")
                self.forward_dtype = torch.float32
            elif not torch.cuda.is_available():
                # Fallback for pure CPU testing if needed
                self.forward_dtype = torch.float32
            else:
                # Standard GPU behavior
                self.forward_dtype = torch.bfloat16
            print(f"Auto-casting ACT to {self.forward_dtype}")

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        vision_shape = []
        self.vision_key = []
        joint_shape = []
        self.joint_key = []

        for key, val in shape_meta['obs'].items():
            if key.endswith('image'):
                # image observation
                vision_shape.append(val['shape'])
                self.vision_key.append(key)
            else:
                joint_shape.append(val['shape'][0])
                self.joint_key.append(key)

        # log.info(f"{self.vision_key, self.joint_key}")
        # get feature dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Token embeddings
        self.embed_scale = math.sqrt(self.hidden_dim)
        embed_init_std = 1.0 / self.embed_scale
        self.cls_embedding = CastedEmbedding(
            1, hidden_dim, embed_init_std, self.forward_dtype
        )
        
        # create encoder model
        self.enc_action_project = CastedLinear(
            action_dim, hidden_dim, bias=False
        )
        self.enc_joint_project = CastedLinear(
            sum(joint_shape), hidden_dim, bias=False
        )
        self.encoder = Transformer(hidden_dim, n_heads, num_enc_layers, feedforward_dim, dropout)
        self.enc_latent_proj = CastedLinear(hidden_dim, latent_dim*2, bias=False)

        # create decoder model
        self.vision_encoder = get_resnet(name='resnet18', weights='IMAGENET1K_V1' if pretrained_backbone else None)
        return_layers = {"layer4": "0"}
        self.vision_encoder = IntermediateLayerGetter(self.vision_encoder, return_layers=return_layers)
        self.vision_encoder = replace_submodules(
            self.vision_encoder, 
            predicate=lambda m: isinstance(m, nn.BatchNorm2d),
            func=lambda m: nn.GroupNorm(num_groups=m.num_features//16, num_channels=m.num_features)
        )

        self.vision_proj = CastedLinear(512, hidden_dim, bias=False)
        self.joint_proj = CastedLinear(sum(joint_shape), hidden_dim, bias=False)
        self.latent_proj = CastedLinear(latent_dim, hidden_dim, bias=False)

        self.decoder_enc = Transformer(
            hidden_dim,
            n_heads,
            num_dec_enc_layers,
            feedforward_dim,
            dropout,
        )

        self.decoder_dec = TransformerDecoder(
            hidden_dim, 
            n_heads, 
            num_dec_dec_layers,
            feedforward_dim,
            dropout
            )
        self.action_proj = CastedLinear(hidden_dim, action_dim, bias=False)
        
        self.normalizer = LinearNormalizer()
        self.pos_enc = PositionEmbeddingSine(hidden_dim//2, normalize=True, dtype=self.forward_dtype)
        self.register_buffer('pos_table', get_sinusoid_encoding_table(2+horizon, hidden_dim)) # [CLS], qpos, actions
        self.additional_pos_emb = CastedEmbedding(2, hidden_dim, embed_init_std, self.forward_dtype)
        self.query_emb = CastedEmbedding(horizon, hidden_dim, embed_init_std, self.forward_dtype)

        self.horizon = horizon
        self.action_dim = action_dim
        self.kl_weight = kl_weight
        self.kwargs = kwargs
        
        self.manual_step = 0

    def setup(self, stage: str):
        """Called by Lightning when setting up the model."""

        if stage == "fit":
            # Calculate steps from dataset and epochs
            dm = self.trainer.datamodule
        
            samples_per_epoch = len(dm.train_dataset)
            
            steps_per_epoch = samples_per_epoch // dm.batch_size
            
            if self.trainer.max_epochs > 0:
                self.total_steps = steps_per_epoch * self.trainer.max_epochs
            else:
                self.total_steps = float("inf")

            episodes = set(np.where(dm.train_dataset.train_mask)[0])
            log.info("Training configuration:")
            log.info(f"  Episodes: {len(episodes)}")
            log.info(f"  Trajectories: {len(dm.train_dataset)}")
            log.info(f"  Steps per epoch: {steps_per_epoch}")
            log.info(f"  Total steps: {self.total_steps}")

    
    # ========= inference  ============
    def encode_action(self, input: Dict[str, torch.Tensor]):
        # Normalized input 
        inp_action = input['action'] # (batch, t, action_dim)
        joints = [input['obs'][k] for k in self.joint_key]
        joints = torch.cat(joints, dim=-1)
        joints = joints[:, :1, :] # (batch, 1, joint_dim)

        bs = inp_action.shape[0]
        
        inp_action = self.enc_action_project(inp_action) # (bs, t, hidden)
        joint_emb = self.enc_joint_project(joints) # (bs, 1, hidden)
        cls_embed = self.cls_embedding.embedding_weight
        cls_embed = torch.unsqueeze(cls_embed, 0).repeat(bs, 1, 1) # (bs, 1, hidden)

        feed = torch.cat((cls_embed, joint_emb, inp_action), axis=1) # (bs, 2 + horizon, hidden)
        pos_embed = self.pos_table.clone().detach().to(dtype=self.forward_dtype)
        out = self.encoder(cos_sin=None, input=feed, pos_enc=pos_embed)
        latent = self.enc_latent_proj(out[:, 0, ...]) # take the cls token
        mu, log_var = latent[:, :self.latent_dim], latent[:, self.latent_dim:]
        latent_sample = reparametrize(mu, log_var)
        
        return latent_sample, (mu, log_var)


    def decode_action(self, latent_z: torch.Tensor, 
                      input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        - latent_z: latent vector (batch, latent_dim)
        - input: batch input with 'obs' key

        returns:
        - action: actions to perform (batch, n_action_steps, action_dim)
        - pred_action: total predicted action (batch, horizon, action_dim)
        """
        # normalized input
        B = latent_z.shape[0]
        T = self.horizon
        Da = self.action_dim
        
        vision_images = []
        for key in self.vision_key:
            vision_images.append(input['obs'][key][:, 0, ...])
        vision_tokens = []
        pos_emb = []
        for image in vision_images:
            image = self.vision_encoder(image)['0']
            vision_tokens.append(rearrange(image, "b l W H -> b (W H) l"))
            pos_emb.append(rearrange(self.pos_enc(image), "b l W H -> b (W H) l"))
        
        vision_tokens = torch.cat(vision_tokens, dim=1)
        vision_tokens = self.vision_proj(vision_tokens)
        pos_emb = torch.cat(pos_emb, dim=1) # (1, seq_len, hidden)

        joints = [input['obs'][k] for k in self.joint_key]
        joints = torch.cat(joints, dim=-1)
        joints = joints[:, :1, :] # (batch, 1, joint_dim)
        joints_token = self.joint_proj(joints) # (batch, 1, hidden)

        latent_token = torch.unsqueeze(self.latent_proj(latent_z), dim=1) # (batch, 1, hidden)
        tokens = torch.cat((vision_tokens, joints_token, latent_token), dim=1).to(dtype=self.forward_dtype)
        pos_emb = torch.cat((pos_emb, self.additional_pos_emb.embedding_weight[None, ...]), dim=1).to(dtype=self.forward_dtype)
        pred_action = self.decoder_enc(cos_sin=None, input=tokens, pos_enc=pos_emb)

        target = torch.zeros(B, T, self.hidden_dim, dtype=self.forward_dtype, device=pred_action.device)
        pred_action = self.decoder_dec(source=pred_action, target=target, 
                                       pos_enc=self.query_emb.embedding_weight.to(self.forward_dtype)[None, ...], 
                                       key_pos=pos_emb)
        pred_action = self.action_proj(pred_action)

        return pred_action
    
    def predict_action(self, input: dict[str, torch.Tensor]):
        """
        - input: batch input with 'obs' key. 

        returns: Unnormalized actions
        - action: actions to perform (batch, n_action_steps, action_dim)
        - pred_action: total predicted action (batch, horizon, action_dim)
        """
        # normalize input
        nobs = self.normalizer.normalize(input['obs'])
        B = nobs[self.joint_key[0]].shape[0]
        T = self.horizon
        Da = self.action_dim
        
        vision_images = []
        for key in self.vision_key:
            vision_images.append(nobs[key][:, -1, ...])
        vision_tokens = []
        pos_emb = []
        for image in vision_images:
            image = self.vision_encoder(image)['0']
            vision_tokens.append(rearrange(image, "b l W H -> b (W H) l"))
            pos_emb.append(rearrange(self.pos_enc(image), "b l W H -> b (W H) l"))
        
        vision_tokens = torch.cat(vision_tokens, dim=1)
        vision_tokens = self.vision_proj(vision_tokens)
        pos_emb = torch.cat(pos_emb, dim=1) # (1, seq_len, hidden)

        joints = [nobs[k] for k in self.joint_key]
        joints = torch.cat(joints, dim=-1)
        joints = joints[:, -1:, :] # (batch, 1, joint_dim)
        joints_token = self.joint_proj(joints) # (batch, 1, hidden)
        latent_z = torch.zeros(B, self.latent_dim, dtype=self.forward_dtype, device=joints_token.device)
        latent_token = torch.unsqueeze(self.latent_proj(latent_z), dim=1) # (batch, 1, hidden)
        tokens = torch.cat((vision_tokens, joints_token, latent_token), dim=1).to(dtype=self.forward_dtype)
        pos_emb = torch.cat((pos_emb, self.additional_pos_emb.embedding_weight[None, ...]), dim=1).to(dtype=self.forward_dtype)
        pred_action = self.decoder_enc(cos_sin=None, input=tokens, pos_enc=pos_emb)

        target = torch.zeros(B, T, self.hidden_dim, dtype=self.forward_dtype, device=pred_action.device)
        pred_action = self.decoder_dec(source=pred_action, target=target, 
                                       pos_enc=self.query_emb.embedding_weight.to(self.forward_dtype)[None, ...], 
                                       key_pos=pos_emb)
        pred_action = self.action_proj(pred_action)
        pred_action = self.normalizer['action'].unnormalize(pred_action)
        
        output = {
            'pred_action': pred_action
        }

        return output
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
        
        assert horizon == self.horizon, "ACT dataset horizon does not match initialized settings"

        latent_z, (mu, log_var) = self.encode_action({'action': nactions, 'obs': nobs})
        
        pred = self.decode_action(latent_z, {'obs': nobs})

        reg_loss, _, _ = kl_divergence(mu, log_var)
        reg_loss = reg_loss[0]
        rec_loss = F.l1_loss(pred, nactions, reduction='none')
        rec_loss = reduce(rec_loss, 'b ... -> b (...)', 'mean') # mean over sequence
        rec_loss = rec_loss.mean() # mean over batch
        
        loss = rec_loss + self.kl_weight * reg_loss

        metrics = {
            'reconstruction_loss': rec_loss.detach(),
            'regulation_loss': reg_loss.detach(),
            'total_loss': loss.detach()
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

        # TODO: implement LR warmup by wiring compute_lr into configure_optimizers
        # (warmup_steps, lr_min_ratio are stored in hparams but unused)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        opt.step()
        opt.zero_grad()

        self.log_metrics(metrics, self.hparams.learning_rate, batch_size=batch_size)

        self.manual_step += 1

        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        batch_size = batch['action'].shape[0]
        # print(f"Validation step {batch_idx}, batch size {batch_size}")

        with torch.no_grad():
            loss, metrics = self.compute_loss_and_metrics(batch)

            for name, value in metrics.items():
                self.log(f"val/{name}", value, on_step=False, on_epoch=True)

            return metrics
 
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return self.validation_step(batch, batch_idx)

    def grad_monitoring(self):
        with torch.no_grad():

            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), 
                max_norm = float('inf')
            ).item()

            grad_metrics = {}

            # if self.denoise_net is not None:
            #     grad_metrics['unet'] = self.denoise_net.weight.grad.norm().item()

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
            for key, value in metrics.items():
                self.log(f"train/{key}", value, on_step=True)

    # TODO: Use Muon optimizer
    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "vision_encoder" not in n and p.requires_grad],
            },
            {
                "params": [p for n, p in self.named_parameters() if "vision_encoder" in n and p.requires_grad],
                "lr": self.hparams.learning_rate * 0.1,
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


if __name__=='__main__':
    vision_encoder = get_resnet(name='resnet18')
    vision_encoder = IntermediateLayerGetter(vision_encoder, {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"})
    vision_encoder = replace_submodules(
        vision_encoder, 
        predicate=lambda m: isinstance(m, nn.BatchNorm2d),
        func=lambda m: nn.GroupNorm(num_groups=m.num_features//16, num_channels=m.num_features)
    )

    print([name for name, _ in vision_encoder.named_parameters()])


    x = torch.rand((4, 3, 256, 256))
    y = vision_encoder(x)
    print(f"{y['0'].shape=}")
    print(f"{y['1'].shape=}")
    print(f"{y['2'].shape=}")
    print(f"{y['3'].shape=}")