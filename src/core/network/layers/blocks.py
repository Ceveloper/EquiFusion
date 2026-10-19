import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from einops import rearrange
from torch import nn, einsum
from typing import Tuple, Optional
from functools import partial

from .graph import GraphConv, Block
from .utils import (
    get_n_params,
    SequentialWithAdj,
    Identity,
    masked_softmax_inplace,
    build_softmax_mask,
    Residual,
    RMSNorm,
    get_norm_layer,
)
from .linear import MaskedLinear2
from .transformer import TransformerBlock
from .attention import (
    ResnetBlock,
    Attention,
)
import time
import inspect


class ParallelAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        num_nodes,
        act_type="tanh",
        *args,
        transformer_kwargs={
            "num_heads": 4,
        },
        **graph_kwargs,
    ):
        super().__init__()

        if act_type == "tanh":
            self.act = nn.Tanh()
        else:
            assert 0, f"Activation type {act_type} not implemented!"
        self.branch_graphconv = SequentialWithAdj(
            ResnetBlock(dim, dim_out, num_nodes=num_nodes, bias=True, **graph_kwargs),
            get_norm_layer(transformer_kwargs["norm_layer"], dim_out),
            self.act,
            TransformerBlock(
                dim_in=dim_out,
                num_nodes=num_nodes,
                qkv_bias=False,
                **transformer_kwargs,
                **graph_kwargs,
            ),
            get_norm_layer(transformer_kwargs["norm_layer"], dim_out),
            self.act,
            ResnetBlock(
                dim_out, dim_out, num_nodes=num_nodes, bias=True, **graph_kwargs
            ),
        )
        self.branch_linear = SequentialWithAdj(
            MaskedLinear2(in_features=dim, out_features=dim_out, bias=True),
            get_norm_layer(transformer_kwargs["norm_layer"], dim_out),
            self.act,
            TransformerBlock(
                dim_in=dim_out,
                num_nodes=num_nodes,
                qkv_bias=False,
                **transformer_kwargs,
                **graph_kwargs,
            ),
            get_norm_layer(transformer_kwargs["norm_layer"], dim_out),
            self.act,
            ResnetBlock(
                dim_out, dim_out, num_nodes=num_nodes, bias=True, **graph_kwargs
            ),
        )

    def forward(
        self,
        x,
        adj: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ):
        x1 = self.branch_graphconv(x, adj=adj, mask=mask, attn_bias=attn_bias)
        x2 = self.branch_linear(x, adj=adj, mask=mask, attn_bias=attn_bias)
        x = x1 + x2
        return x


class ParallelBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_nodes,
        act_type="tanh",
        *args,
        transformer_kwargs={
            "num_heads": 4,
        },
        **graph_kwargs,
    ):
        super().__init__()

        if act_type == "tanh":
            self.act = nn.Tanh()
        else:
            assert 0, f"Activation type {act_type} not implemented!"
        self.branch_graphconv = SequentialWithAdj(
            TransformerBlock(
                dim_in=dim,
                num_nodes=num_nodes,
                qkv_bias=False,
                **transformer_kwargs,
                **graph_kwargs,
            ),
            get_norm_layer(transformer_kwargs["norm_layer"], dim),
            self.act,
            ResnetBlock(dim, dim, num_nodes=num_nodes, bias=True, **graph_kwargs),
        )

        self.branch_linear = SequentialWithAdj(
            MaskedLinear2(in_features=dim, out_features=dim, bias=True),
            get_norm_layer(transformer_kwargs["norm_layer"], dim),
            self.act,
            ResnetBlock(dim, dim, num_nodes=num_nodes, bias=True, **graph_kwargs),
        )

    def forward(
        self,
        x,
        adj: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ):
        x1 = self.branch_graphconv(x, adj=adj, mask=mask, attn_bias=attn_bias)
        x2 = self.branch_linear(x, adj=adj, mask=mask, attn_bias=attn_bias)
        x = x1 + x2
        return x


