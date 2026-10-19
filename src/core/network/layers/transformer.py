import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Union

from .graph import GraphConv
from .utils import Identity, get_norm_layer

from .attention import Attention, RMSNorm, Block
from .utils import SequentialWithAdj

# Adapted From https://github.com/huggingface/pytorch-image-models/tree/e3242a52584bbc69f848f762d254e8a23932832c/timm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int = None,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        # keep very small in magnitude if not None.
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: str = "tanh",
        norm_layer: str = "identity",
        attention_layer: nn.Module = Attention,
        fc_layer: nn.Module = Block,
        **graph_kwargs,
    ) -> None:
        super().__init__()
        self.norm1 = get_norm_layer(norm_layer, dim_in)

        dim_out = dim_out if dim_out is not None else dim_in
        self.attn = SequentialWithAdj(
            attention_layer(
                dim=dim_in,
                dim_out=dim_out,  # or leave
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                norm_layer=norm_layer,
                **graph_kwargs,
            )
        )
        self.ls1 = nn.Identity()
        self.drop_path1 = nn.Identity()
        self.norm2 = get_norm_layer(norm_layer, dim_out)

        self.fc = SequentialWithAdj(
            fc_layer(
                dim=dim_out,
                dim_out=dim_out,  # TO CHECK
                act_type=act_layer,
                **graph_kwargs,
            )
        )
        self.ls2 = nn.Identity()
        self.drop_path2 = nn.Identity()
        self.residual = (
            Identity()
            if dim_in == dim_out
            else fc_layer(
                dim=dim_in, dim_out=dim_out, act_type=act_layer, **graph_kwargs
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.residual(x, adj=adj, mask=mask) + self.drop_path1(
            self.ls1(self.attn(self.norm1(x), adj=adj, mask=mask, attn_bias=attn_bias))
        )
        x = x + self.drop_path2(self.ls2(self.fc(self.norm2(x), adj=adj, mask=mask)))
        return x


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------


