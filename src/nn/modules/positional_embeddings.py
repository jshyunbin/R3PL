import math
import einops
import torch
import torch.nn as nn
import numpy as np

def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.stack([emb.sin(), emb.cos()])
        emb = einops.rearrange(emb, "SC L D -> L (D SC)")
        return emb
    
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, dtype=torch.float32):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.dtype = dtype
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        x = tensor
        # mask = tensor_list.mask
        # assert mask is not None
        # not_mask = ~mask

        not_mask = torch.ones_like(x[0, [0]])
        y_embed = not_mask.cumsum(1, dtype=self.dtype)
        x_embed = not_mask.cumsum(2, dtype=self.dtype)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=self.dtype, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        dim_t = dim_t.to(dtype=self.dtype)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos
    
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()
        self.enabled = base > 0
        if not self.enabled:
            return
        # RoPE
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        if not self.enabled:
            return None, None
        return self.cos_cached, self.sin_cached

class RotaryEmbedding2D(nn.Module):
    """2D RoPE using same frequency basis as 1D, with prefix support."""
    
    def __init__(self, dim, prefix_len, max_grid_size, base=10000, device=None):
        super().__init__()
        self.prefix_len = prefix_len
        self.max_grid_size = max_grid_size
        self.dim = dim
        
        # SAME frequency formula as 1D RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        self._build_cache(device)
    
    def _build_cache(self, device=None):
        if device is None:
            device = self.inv_freq.device
        
        n_freq = self.inv_freq.shape[0]  # dim // 2 (32 for dim=64)
        quarter = n_freq // 2            # dim // 4 (16 for dim=64)
        
        # Prefix: standard 1D positions
        if self.prefix_len > 0:
            prefix_pos = torch.arange(self.prefix_len, dtype=torch.float32, device=device)
            prefix_freqs = torch.outer(prefix_pos, self.inv_freq)  # [prefix, 32]
            prefix_emb = torch.cat((prefix_freqs, prefix_freqs), dim=-1)  # [prefix, 64]
        
        # Grid: 2D positions
        grid_len = self.max_grid_size ** 2
        indices = torch.arange(grid_len, dtype=torch.float32, device=device)
        rows = indices // self.max_grid_size
        cols = indices % self.max_grid_size
        
        # Row and col BOTH use the same frequencies (first quarter of inv_freq)
        # This gives them equal expressiveness
        row_freqs = torch.outer(rows, self.inv_freq[:quarter])  # [grid, 16]
        col_freqs = torch.outer(cols, self.inv_freq[:quarter])  # [grid, 16]
        
        # Structure: [row, col, row, col] to match rotate_half pattern
        grid_emb = torch.cat([row_freqs, col_freqs, row_freqs, col_freqs], dim=-1)  # [grid, 64]
        
        # Combine prefix + grid
        if self.prefix_len > 0:
            full_emb = torch.cat([prefix_emb, grid_emb], dim=0)
        else:
            full_emb = grid_emb
        
        self.cos_cached = nn.Buffer(full_emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(full_emb.sin(), persistent=False)
    
    def forward(self):
        return self.cos_cached, self.sin_cached
    
def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


if __name__=="__main__":
    sin = PositionEmbeddingSine(256)
    x = torch.rand((64, 512, 82, 82))
    print(sin(x).shape)