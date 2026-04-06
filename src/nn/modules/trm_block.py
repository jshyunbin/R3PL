from typing import List, Tuple, Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention

from src.nn.modules.utils import trunc_normal_init_
from src.nn.modules.positional_embeddings import (
    SinusoidalPosEmb,
    RotaryEmbedding,
    RotaryEmbedding2D,
    apply_rotary_pos_emb
)
from src.nn.utils import RankedLogger

CosSin = Tuple[torch.Tensor, torch.Tensor]

log = RankedLogger(__name__, rank_zero_only=True)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMS normalization
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / norm


class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(
                torch.empty((out_features, in_features)), std=1.0 / (in_features**0.5)
            )
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features,)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(
            input,
            self.weight.to(input.dtype),
            bias=self.bias.to(input.dtype) if self.bias is not None else None,
        )


class CastedEmbedding(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, init_std: float, cast_to: torch.dtype
    ):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class CastedLayerNorm(nn.Module):
    """LayerNorm that casts its parameters to match the input dtype, consistent
    with the CastedLinear / CastedEmbedding convention used throughout this file."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(x.dtype) if self.weight is not None else None
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)


class LinearSwish(nn.Module):
    def __init__(self, hidden_size: int, reverse=False):
        super().__init__()

        self.linear = CastedLinear(hidden_size, hidden_size, bias=False)
        self.reverse = reverse

    def forward(self, x):
        if self.reverse:
            return F.silu(self.linear(x))
        else:
            return self.linear(F.silu(x))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

class CastedConv1d(nn.Conv1d):
    """Conv1d that automatically casts weights/bias to input dtype."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.constant_(self.bias, 0) if self.bias is not None else None
        trunc_normal_init_(self.weight, std=0.02)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.conv1d(
            input,
            self.weight.to(input.dtype),
            self.bias.to(input.dtype) if self.bias is not None else None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

class CastedConv2d(nn.Conv2d):
    """Conv2d that automatically casts weights/bias to input dtype."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.constant_(self.bias, 0) if self.bias is not None else None
        trunc_normal_init_(self.weight, std=0.02)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            input,
            self.weight.to(input.dtype),
            self.bias.to(input.dtype) if self.bias is not None else None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )
    
class ConvSwiGLU(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        expansion: float,
        conv_kernel: int = 3, # Changed default to 3 (Odd is better for alignment)
        intermediate_size: Optional[int] = None,
    ):
        super().__init__()

        inter = intermediate_size if intermediate_size is not None else _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)

        self.dwconv = CastedConv1d(
            in_channels=inter * 2,
            out_channels=inter * 2,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=inter * 2,
            bias=True,
        )

        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, timer: Optional[object] = None, prefix: str = ""):
        
        x_expanded = self.gate_up_proj(x)
        x_conv = self.dwconv(x_expanded.transpose(1, 2))
        if x_conv.size(-1) != x.size(1):
             x_conv = x_conv[..., :x.size(1)]
        x_conv = x_conv.transpose(1, 2)
        gate, up = x_conv.chunk(2, dim=-1)
        x_out = F.silu(gate) * up
        return self.down_proj(x_out)
    
     
class Attention(nn.Module):
    def __init__(self, 
                 hidden_size, 
                 head_dim, 
                 num_heads, 
                 num_key_value_heads, 
                 is_causal=False,
                 is_cross_attention=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.is_cross_attention = is_cross_attention
        self.causal = is_causal

        self.q_proj = CastedLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = CastedLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False
        )
        self.v_proj = CastedLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False
        )
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, 
                cos_sin: CosSin, 
                query: torch.Tensor, 
                key_value: Optional[torch.Tensor] = None, 
                pos_enc: Optional[torch.Tensor] = None,
                key_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape
        assert self.is_cross_attention == (key_value is not None), "Cross attention requires key value tensors."
        if key_value is not None:
            key_val_len = key_value.shape[1]
            key = key_value
            value = key_value

        # hidden_states: [bs, seq_len, hidden_size]
        if not self.is_cross_attention:
            q = query
            query = self.q_proj(q if pos_enc is None else q+pos_enc).view(batch_size, seq_len, self.num_heads, self.head_dim)
            key = self.k_proj(q if pos_enc is None else q+pos_enc).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            value = self.v_proj(q).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        else:
            query = self.q_proj(query if pos_enc is None else query+pos_enc).view(batch_size, seq_len, self.num_heads, self.head_dim)
            key = self.k_proj(key if key_pos is None else key+key_pos).view(batch_size, key_val_len, self.num_key_value_heads, self.head_dim)
            value = self.v_proj(value).view(batch_size, key_val_len, self.num_key_value_heads, self.head_dim)

        # RoPE only when self-attention
        if (cos_sin is not None) and not self.is_cross_attention:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # flash attn
        query, key, value = map(
            lambda t: einops.rearrange(t, "B S H D -> B H S D"), (query, key, value)
        )  # needed for scaled_dot_product_attention but not flash_attn_func
        
        attn_output = scaled_dot_product_attention(
            query=query, key=key, value=value, is_causal=self.causal
        )
        attn_output = einops.rearrange(attn_output, "B H S D -> B S H D")

        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.output_size)  # type: ignore
        return self.o_proj(attn_output)
    

class DecoderTransformerBlock(nn.Module):
    def __init__(self, 
                 hidden_size: int, 
                 head_dim: int, 
                 num_heads: int, 
                 dim_feedforward: int,
                 dropout: float = 0.1,
                 is_causal: bool = False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.is_causal = is_causal

        self.self_attention = Attention(
            hidden_size,
            head_dim,
            num_heads,
            num_heads,
            is_causal=is_causal
        )

        self.cross_attention = Attention(
            hidden_size,
            head_dim,
            num_heads,
            num_heads,
            is_cross_attention=True,
            is_causal=is_causal
        )

        self.linear1 = CastedLinear(hidden_size, dim_feedforward, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = CastedLinear(dim_feedforward, hidden_size, bias=False)

        self.norm1 = CastedLayerNorm(hidden_size)
        self.norm2 = CastedLayerNorm(hidden_size)
        self.norm3 = CastedLayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, input, memory, pos_enc=None, key_pos=None) -> torch.Tensor:
        out = self.self_attention(query=input, pos_enc=pos_enc, cos_sin=None) + self.dropout1(input)
        out = self.norm1(out)
        out2 = self.cross_attention(query=out, key_value=memory, pos_enc=pos_enc, key_pos=key_pos, cos_sin=None)
        out += self.dropout2(out2)
        out = self.norm2(out)
        out2 = self.linear2(self.dropout(F.relu(self.linear1(out))))
        out = out + self.dropout3(out2)
        out = self.norm3(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, 
                 hidden_size: int, 
                 head_dim: int, 
                 num_heads: int, 
                 dim_feedforward: int,
                 dropout: float = 0.1,
                 is_causal: bool = False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.causal = is_causal

        self.attention = Attention(hidden_size, head_dim, num_heads, num_heads, is_causal=is_causal)
        self.linear1 = CastedLinear(hidden_size, dim_feedforward, bias=True)
        self.linear2 = CastedLinear(dim_feedforward, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = CastedLayerNorm(hidden_size)
        self.norm2 = CastedLayerNorm(hidden_size)

    def forward(self, cos_sin: CosSin, input: torch.Tensor, pos_enc: torch.Tensor):
        x = input
        att = self.attention(cos_sin=cos_sin, query=x, pos_enc=pos_enc)
        x = att + self.dropout1(x)
        x = self.norm1(x)
        res = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x += self.dropout2(res)
        x = self.norm2(x)
        return x
    
    
class Transformer(nn.Module):
    def __init__(self,
                 hidden_size: int, 
                 n_heads: int, 
                 num_layers: int, 
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 causal: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.causal = causal
        self.num_layers = num_layers
        
        self.model = nn.ModuleList([
            TransformerBlock(hidden_size, 
                             hidden_size//n_heads, 
                             n_heads, 
                             dim_feedforward,
                             dropout, 
                             causal) for _ in range(num_layers)
        ])
        
    def forward(self, cos_sin: CosSin, input: torch.Tensor, pos_enc: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            input (torch.Tensor): tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            torch.Tensor: output tensor same shape as input
        """
        for i in range(self.num_layers):
            input = self.model[i](cos_sin=cos_sin, input=input, pos_enc=pos_enc)
        return input

class TransformerDecoder(nn.Module):
    def __init__(self, 
                 hidden_size: int, 
                 n_heads: int, 
                 num_layers: int, 
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 causal: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.causal = causal
        self.num_layers = num_layers
        
        self.decoder = nn.ModuleList([
            DecoderTransformerBlock(hidden_size,
                             hidden_size//n_heads,
                             n_heads,
                             dim_feedforward,
                             dropout,
                             causal) for _ in range(num_layers)
        ])
        
    def forward(self, source: torch.Tensor, target: torch.Tensor, pos_enc: torch.Tensor, key_pos: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            input (torch.Tensor): tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            torch.Tensor: output tensor same shape as input
        """
        for i in range(self.num_layers):
            target = self.decoder[i](input=target, memory=source, pos_enc=pos_enc, key_pos=key_pos)

        return target

    
class ActionReasoningModule(nn.Module):
    def __init__(self,
                 hidden_size: int, 
                 n_heads: int, 
                 num_layers: int, 
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 causal: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.causal = causal
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, 
                             hidden_size//n_heads, 
                             n_heads, 
                             dim_feedforward,
                             dropout, 
                             causal) for _ in range(num_layers)
        ])

    def forward(
        self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(input=hidden_states, **kwargs)
        return hidden_states


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


def _find_multiple(a, b):
    return (-(a // -b)) * b


if __name__=='__main__':
    # test transformers
    encoder = Transformer(
        512, 8, 2
    )

    decoder = TransformerDecoder(
        512, 8, 2, 2
    )

    x = torch.rand((1, 5, 512))
    print(f"{x.shape=}")
    x = encoder(cos_sin=None, input=x)
    print(f"{x.shape=}")
    y = torch.rand((1, 7, 512))
    x = decoder(cos_sin=None, source=x, target=y)
    print(f"{x.shape=}")