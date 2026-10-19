import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from einops import rearrange
from torch import nn, einsum
from typing import Tuple, Optional

from .graph import GraphConv, Block
from .utils import (
    SequentialWithAdj,
    Identity,
    masked_softmax_inplace,
    build_softmax_mask,
    Residual,
    RMSNorm,
)
import time
import inspect


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        time_emb_dim=None,
        groups=8,
        linear_layer_type: nn.Module = GraphConv,
        **kwargs
    ):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.Tanh(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim is not None
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups, **kwargs)
        self.block2 = Block(dim_out, dim_out, groups=groups, **kwargs)
        kwargs["bias"] = False
        self.res_linear = (
            SequentialWithAdj(linear_layer_type(dim, dim_out, **kwargs))
            if dim != dim_out
            else Identity()
        )

    def forward(
        self,
        x,
        time_emb=None,
        adj: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ):

        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b 1 c")
            scale_shift = time_emb.chunk(2, dim=-1)

        # # Step 1: block1

        # # Step 2: block2

        # # Step 3: res_linear

        h = self.block1(
            x, scale_shift=scale_shift, adj=adj, mask=mask, attn_bias=attn_bias
        )

        h = self.block2(h, adj=adj, mask=mask, attn_bias=attn_bias)

        return h + self.res_linear(x, adj=adj, mask=mask)


    # )

    # )


# We need default num_heads: int = 8,
class MaskedAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        num_heads=4,
        dim_head=32,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        layer_type: nn.Module = GraphConv,
        proj_dropout: float = 0.0,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.Identity,
        if_somophormer_attention_bias: bool = False,
        **kwargs
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.num_heads = num_heads
        self.if_somophormer_attention_bias = if_somophormer_attention_bias
        hidden_dim = dim_head * num_heads
        dim_out = dim_out if dim_out is not None else dim

        self.to_qkv = layer_type(dim, hidden_dim * 3, bias=qkv_bias, **kwargs)
        self.to_out = layer_type(hidden_dim, dim_out, bias=False, **kwargs)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_dropout = nn.Dropout(proj_dropout)

        self.q_norm = norm_layer(dim_head) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(dim_head) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        adj: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ):
        *b, n, c = x.shape
        qkv = self.to_qkv(x, adj=adj, mask=mask).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "... b n (h c)  -> ... b h n c", h=self.num_heads),
            qkv,
        )
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale  # [b, num_heads, nodes, dim]
        sim = q @ k.mT
        if mask is None:
            mask = (adj.sum(-1) != 0.0).unsqueeze(-1)
        mask = build_softmax_mask(mask).unsqueeze(-3)
        if self.if_somophormer_attention_bias:
            assert attn_bias is not None
            sim = sim + attn_bias.unsqueeze(-3)  # [b, num_heads, nodes, nodes]
        attn = masked_softmax_inplace(sim, mask, dim=-1)  # [b, num_heads, nodes, nodes]
        attn = self.attn_dropout(attn)

        out = attn @ v  # [b, num_heads, nodes, dim]

        out = rearrange(
            out, "... b h n d -> ... b n (h d)"
        )  # [b, nodes, num_heads*dim]
        return self.out_dropout(self.to_out(out, adj=adj, mask=mask))


def Attention(*args, if_pi_denoiser=False, **kwargs):
    return MaskedAttention(*args, **kwargs)


