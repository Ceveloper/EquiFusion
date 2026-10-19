import torch
from torch import nn
from typing import Optional

from .utils import SinusoidalPosEmb
from ..layers import GraphConv, Attention, ResnetBlock, Residual, PreNorm, Identity, SequentialWithAdj


class DenoisingGraphTransformer(nn.Module):
    """Denoiser of the latent diffusion (differs from SkeletonDiffusion's Denoiser).

    Operates on one latent vector per graph node [S, B, N, D], conditioned on the latents of the observed motion.
    Alternates time-conditioned graph ResNet blocks with attention over the (masked) graph nodes; all layers are
    shared across nodes, so the network is equivariant to permutations of the nodes.
    Extra keyword arguments (e.g. num_heads, dim_head, add_bias_kv, norm_type, num_nodes) are forwarded to the layers.
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        cond_dim: int = 0,
        depth: int = 1,
        resnet_block_groups: int = 8,
        sinusoidal_pos_emb_theta: int = 10000,
        graph_arch: dict = {},
        **kwargs,
    ):
        super().__init__()
        self.channels = kwargs["num_nodes"]
        self.self_condition = False
        diffusion_size = dim
        layer_kwargs = {**graph_arch, **kwargs}

        self.init_lin = SequentialWithAdj(GraphConv(dim + cond_dim, diffusion_size, bias=True, **layer_kwargs))
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(diffusion_size, theta=sinusoidal_pos_emb_theta),
            nn.Linear(diffusion_size, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        def resnet_block():
            return ResnetBlock(diffusion_size, diffusion_size, time_emb_dim=time_dim, groups=resnet_block_groups,
                               **layer_kwargs)

        def attention_block():
            return Residual(PreNorm(diffusion_size, SequentialWithAdj(Attention(diffusion_size, **layer_kwargs))))

        layers = []
        for _ in range(depth):  # "down" half
            layers += [resnet_block(), attention_block()]
        for i in range(depth):  # "up" half (no attention after the last block)
            layers += [resnet_block(), attention_block() if i != depth - 1 else Identity()]
        self.layers = SequentialWithAdj(*layers)
        self.final_res_block = ResnetBlock(diffusion_size * 2, diffusion_size, groups=resnet_block_groups, **layer_kwargs)
        self.final_glin = SequentialWithAdj(GraphConv(diffusion_size, out_dim, bias=True, **layer_kwargs))

    def get_cond(self, z_past_motion, z_past_bone=None, adj=None, mask=None):
        """Conditioning: the per-node latents of the observed motion."""
        return z_past_motion

    def forward(self, x, time, x_self_cond=None, x_cond=None, adj: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None):
        if x_cond is not None:
            x = torch.cat([x_cond, x], dim=-1)
        x = self.init_lin(x, adj=adj, mask=mask)
        r = x
        t = self.time_mlp(time)
        x = self.layers(x, time_emb=t, adj=adj, mask=mask, attn_bias=None)
        x = torch.cat((x, r), dim=-1)
        x = self.final_res_block(x, adj=adj, mask=mask)
        return self.final_glin(x, adj=adj, mask=mask)
