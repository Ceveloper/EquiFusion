from typing import Optional

import torch
import torch.nn as nn

from ..layers import (
    TransformerBlock,
    ResnetBlock,
    GraphConv,
    SequentialWithAdj,
    ParallelAttentionBlock,
    ParallelBlock,
    get_n_params,
)
from .losses import loss as loss_function


class MotionGraphTransformer(nn.Module):
    """Permutation equivariant graph transformer that encodes each node (joint or bone) trajectory
    into a per-node latent vector and decodes it back to a trajectory.

    All layers act on the node dimension only through the (masked) adjacency matrix, so permuting
    the nodes together with the adjacency permutes latents and outputs accordingly.
    Sequences shorter than `num_timesteps` are zero-padded in time.
    """

    def __init__(
        self,
        num_timesteps: int = 120,
        num_nodes: int = 22,
        input_size: int = 3,
        embed_dim: int = 96,
        encoder_hidden_size: int = 128,
        graph_arch: dict = {},
        transformer_arch_enc: dict = {},
        transformer_arch_dec: dict = {},
    ):
        super().__init__()
        self.num_1d_patches = num_timesteps
        self.num_nodes = num_nodes
        enc_heads, enc_norm = transformer_arch_enc["num_heads"], transformer_arch_enc["norm_layer"]
        dec_heads, dec_norm = transformer_arch_dec["num_heads"], transformer_arch_dec["norm_layer"]

        # Encoder: [B, N, T*3] -> [B, N, embed_dim]
        init_size = num_timesteps * input_size
        smaller_size = init_size // 2
        self.encoder_parallel_attention_branch = SequentialWithAdj(
            ParallelAttentionBlock(
                dim=init_size, dim_out=smaller_size, num_nodes=num_nodes, act_type="tanh",
                transformer_kwargs={"num_heads": enc_heads, "norm_layer": enc_norm},
                graph_influence=None, **graph_arch,
            ),
        )
        self.enc_projection = SequentialWithAdj(
            TransformerBlock(dim_in=smaller_size, num_heads=enc_heads, qkv_bias=False, norm_layer=enc_norm, **graph_arch),
            ResnetBlock(smaller_size, encoder_hidden_size, num_nodes=num_nodes, bias=True, graph_influence=None, **graph_arch),
        )
        self.encoder_parallel_latent_branch = ParallelBlock(
            dim=encoder_hidden_size, act_type="tanh", num_nodes=num_nodes,
            transformer_kwargs={"num_heads": enc_heads, "norm_layer": enc_norm},
            graph_influence=None, **graph_arch,
        )
        self.fc = SequentialWithAdj(
            TransformerBlock(dim_in=encoder_hidden_size, num_heads=enc_heads, qkv_bias=False, norm_layer=enc_norm),
            GraphConv(encoder_hidden_size, embed_dim, num_nodes=num_nodes, graph_influence=None, bias=True, **graph_arch),
        )

        # Decoder: [B, N, embed_dim] -> [B, N, T*3]
        final_dim = num_timesteps * input_size
        self.decoder_pred = ResnetBlock(
            embed_dim, final_dim, num_nodes=num_nodes, bias=True, graph_influence=None, **graph_arch, norm_layer=dec_norm,
        )
        node_kwargs = dict(num_nodes=num_nodes, graph_influence=None, **graph_arch)
        self.decoder_blocks = SequentialWithAdj(
            TransformerBlock(dim_in=final_dim, num_heads=dec_heads, qkv_bias=False, norm_layer=dec_norm, **node_kwargs),
            ResnetBlock(final_dim, final_dim, bias=True, norm_layer=dec_norm, **node_kwargs),
            ResnetBlock(final_dim, final_dim, bias=True, norm_layer=dec_norm, **node_kwargs),
            TransformerBlock(dim_in=final_dim, num_heads=dec_heads, qkv_bias=False, norm_layer=dec_norm, **node_kwargs),
            ResnetBlock(final_dim, final_dim, bias=True, norm_layer=dec_norm, **node_kwargs),
            GraphConv(final_dim, final_dim, bias=True, **node_kwargs),
        )

        n_enc = sum(get_n_params(m) for m in [self.encoder_parallel_attention_branch, self.enc_projection,
                                              self.encoder_parallel_latent_branch, self.fc])
        n_dec = get_n_params(self.decoder_pred) + get_n_params(self.decoder_blocks)
        print(f"Created encoder with {n_enc:.3f}M and decoder with {n_dec:.3f}M trainable parameters.")

    def pad_with_zero(self, x):
        T = x.shape[-3]
        if self.num_1d_patches > T:
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, self.num_1d_patches - T))
        return x

    def encode(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        """x: [B, T, N, 3] -> latent [B, N, embed_dim]"""
        x = self.pad_with_zero(x)
        B, T, N, _ = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, N, -1)  # [B, N, T*3]
        x = self.encoder_parallel_attention_branch(x, adj=adj, mask=mask, attn_bias=None)
        x = self.enc_projection(x, adj=adj, mask=mask, attn_bias=None)
        x = self.encoder_parallel_latent_branch(x, adj=adj, mask=mask, attn_bias=None)
        return self.fc(x, adj=adj, mask=mask, attn_bias=None)

    def decode(self, h: torch.Tensor, timesteps: int, adj: Optional[torch.Tensor] = None,
               mask: Optional[torch.Tensor] = None):
        """h: [B, N, embed_dim] -> trajectory [B, timesteps, N, 3]"""
        h = self.decoder_pred(h, adj=adj, mask=mask)
        h = self.decoder_blocks(h, adj=adj, mask=mask, attn_bias=None)
        *B, N, _ = h.shape
        h = h.view(*B, N, -1, 3).transpose(-2, -3)  # [B, T, N, 3]
        return h[..., :timesteps, :, :]


class EquivariantAutoEncoder(nn.Module):
    """EquiFusion autoencoder (replaces SkeletonDiffusion's recurrent AutoEncoder).

    Maps a motion [B, T, N, 3] of N graph nodes (joints for the 3D representation, bones for the
    bone direction representation) to one latent vector per node.
    Unused node slots (padding of skeletons with fewer nodes) are masked out via `mask` and `adj`.
    """

    def __init__(
        self,
        num_nodes: int,
        latent_size: int = 96,
        encoder_hidden_size: int = 128,
        prediction_horizon_train: int = 120,
        input_size: int = 3,
        loss_pose_type: str = "l1",
        graph_arch: dict = {},
        transformer_arch_enc: dict = {},
        transformer_arch_dec: dict = {},
        **kwargs,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.latent_size = latent_size
        self.loss_pose_type = loss_pose_type
        self.mae = MotionGraphTransformer(
            num_timesteps=prediction_horizon_train,
            num_nodes=num_nodes,
            input_size=input_size,
            embed_dim=latent_size,
            encoder_hidden_size=encoder_hidden_size,
            graph_arch=graph_arch,
            transformer_arch_enc=transformer_arch_enc,
            transformer_arch_dec=transformer_arch_dec,
        )

    @staticmethod
    def node_norms(x: torch.Tensor) -> torch.Tensor:
        """Mean norm of each node vector over time [B, N, 1] (the bone length for the direction repr.)."""
        return torch.linalg.norm(x, dim=-1).mean(-2).unsqueeze(-1)

    def encode(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        """Returns the latent z [B, N, latent_size] and the tuple (z, node norms)."""
        z = self.mae.encode(x, adj=adj, mask=mask)
        return z, (z, self.node_norms(x))

    def forward(self, x, adj=None, mask=None, **kwargs):
        return self.encode(x, adj=adj, mask=mask)

    def get_embedding(self, y, adj=None, mask=None, **kwargs):
        return self.encode(y, adj=adj, mask=mask)

    def get_past_embedding(self, past, adj=None, mask=None, **kwargs):
        with torch.no_grad():
            z_past, z_past_vars = self.encode(past, adj=adj, mask=mask)
        return z_past, z_past_vars, (None, None)

    def get_train_embeddings(self, y, past, adj=None, mask=None, **kwargs):
        z_past, z_past_vars, _ = self.get_past_embedding(past, adj=adj, mask=mask)
        z, z_vars = self.get_embedding(y, adj=adj, mask=mask)
        return z_past, z, (z_past_vars, z_vars)

    def aggregate_latent(self, z_motion: torch.Tensor, z_bone: torch.Tensor = None) -> torch.Tensor:
        return z_motion

    def decode(self, x: torch.Tensor, h: torch.Tensor, z=None, ph: int = 1, adj=None, mask=None, **kwargs):
        """Decodes the latent h [B, N, latent_size] into ph frames. The past x is not used by the decoder;
        the argument is kept so that callers do not need to distinguish."""
        return self.mae.decode(h, timesteps=ph, adj=adj, mask=mask), None

    def autoencode(self, y, past, ph=1, adj=None, mask=None, **kwargs):
        z, (z_motion, z_bone) = self.encode(y, adj=adj, mask=mask)
        out, feats = self.decode(past, z, None, ph, adj=adj, mask=mask)
        return out, (z_motion, z_bone), z, feats, (None, None)

    def loss(self, y_pred, y, x: torch.Tensor = None, type=None, **kwargs):
        return loss_function(y_pred, y, type=self.loss_pose_type if type is None else type, **kwargs)
