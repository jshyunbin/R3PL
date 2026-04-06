import torch
import pytest

from nn.modules.positional_embeddings import (
    get_sinusoid_encoding_table,
    SinusoidalPosEmb,
    PositionEmbeddingSine,
    apply_rotary_pos_emb,
)


def test_get_sinusoid_encoding_table_shape():
    n, d = 20, 64
    table = get_sinusoid_encoding_table(n, d)
    assert table.shape == (1, n, d)
    assert table.dtype == torch.float32


def test_sinusoidal_pos_emb_shape():
    dim = 32
    emb = SinusoidalPosEmb(dim)
    x = torch.arange(10).float()
    out = emb(x)
    assert out.shape == (10, dim)


def test_position_embedding_sine_shape():
    # PositionEmbeddingSine uses x[0, [0]] to build the mask, so batch dim in
    # output is always 1 regardless of input batch size.
    B, C, H, W = 2, 3, 8, 8
    num_pos_feats = 32
    pos_emb = PositionEmbeddingSine(num_pos_feats=num_pos_feats)
    x = torch.randn(B, C, H, W)
    out = pos_emb(x)
    assert out.shape == (1, 2 * num_pos_feats, H, W)


def test_apply_rotary_pos_emb_shapes():
    B, S, H, D = 2, 6, 4, 16
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, S, H, D)
    cos = torch.randn(S, D)
    sin = torch.randn(S, D)
    q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_embed.shape == q.shape
    assert k_embed.shape == k.shape


def test_apply_rotary_pos_emb_dtype_preserved():
    B, S, H, D = 1, 4, 2, 8
    q = torch.randn(B, S, H, D, dtype=torch.float32)
    k = torch.randn(B, S, H, D, dtype=torch.float32)
    cos = torch.randn(S, D, dtype=torch.float32)
    sin = torch.randn(S, D, dtype=torch.float32)
    q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_embed.dtype == torch.float32
    assert k_embed.dtype == torch.float32
