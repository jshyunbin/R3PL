from typing import Optional

import torch
import torch.nn as nn

from src.nn.modules.positional_embeddings import SinusoidalPosEmb


class TransformerForDiffusion(nn.Module):
    """Transformer-based denoising network for diffusion policy.

    Encoder-decoder architecture: obs features + diffusion timestep are
    encoded into memory; noisy action trajectory is decoded via cross-attention.

    Based on Chi et al. "Diffusion Policy" (2023), transformer hybrid variant.
    Assumes obs_as_cond=True, time_as_cond=True.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int,
        cond_dim: int,
        n_layer: int = 8,
        n_head: int = 4,
        n_emb: int = 256,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.3,
        causal_attn: bool = True,
        n_cond_layers: int = 0,
    ):
        super().__init__()

        # Action input embedding
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # Diffusion timestep embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(n_emb),
            nn.Linear(n_emb, n_emb * 4),
            nn.Mish(),
            nn.Linear(n_emb * 4, n_emb),
        )

        # Obs conditioning embedding
        self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        # Conditioning sequence: [time_token, obs_tokens_0, ..., obs_tokens_{To-1}]
        T_cond = 1 + n_obs_steps
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))

        # Condition encoder (MLP when n_cond_layers=0, TransformerEncoder otherwise)
        if n_cond_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.cond_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_cond_layers)
        else:
            self.cond_encoder = nn.Sequential(
                nn.Linear(n_emb, 4 * n_emb),
                nn.Mish(),
                nn.Linear(4 * n_emb, n_emb),
            )

        # Action decoder (cross-attends to encoded conditioning)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layer)

        # Causal self-attention mask for decoder
        if causal_attn:
            mask = torch.tril(torch.ones(horizon, horizon)).bool()
            mask = mask.float().masked_fill(~mask, float("-inf")).masked_fill(mask, 0.0)
            self.register_buffer("causal_mask", mask)
        else:
            self.causal_mask = None

        # Output head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.cond_pos_emb, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            sample:   (B, T, input_dim)  — noisy action trajectory
            timestep: (B,) or scalar     — diffusion timestep
            cond:     (B, To, cond_dim)  — obs features (image + agent_pos)
        Returns:
            (B, T, output_dim)           — denoised action trajectory
        """
        B, T, _ = sample.shape

        # Embed action tokens
        x = self.input_emb(sample) + self.pos_emb  # (B, T, n_emb)
        x = self.drop(x)

        # Build conditioning sequence: [time_token | obs_tokens]
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.expand(B).to(sample.device)

        time_token = self.time_emb(timestep).unsqueeze(1)  # (B, 1, n_emb)

        if cond is not None:
            obs_tokens = self.cond_obs_emb(cond)           # (B, To, n_emb)
            cond_tokens = torch.cat([time_token, obs_tokens], dim=1)  # (B, 1+To, n_emb)
        else:
            cond_tokens = time_token

        cond_tokens = cond_tokens + self.cond_pos_emb[:, :cond_tokens.shape[1]]
        memory = self.cond_encoder(cond_tokens)             # (B, 1+To, n_emb)

        # Decode action tokens with cross-attention to memory
        out = self.decoder(x, memory, tgt_mask=self.causal_mask)  # (B, T, n_emb)
        out = self.ln_f(out)
        return self.head(out)                               # (B, T, output_dim)
