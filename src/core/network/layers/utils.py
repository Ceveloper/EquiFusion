import torch
from torch import nn, einsum, Tensor
import time
import torch.nn.functional as F
from typing import Optional
import inspect


# @torch.compile
def masked_softmax_inplace(scores, mask, dim=-1):
    """
    scores: [..., N] attention logits
    mask: same shape (broadcastable) boolean, True = keep, False = mask out
    Memory-friendly: uses in-place masked_fill_ before softmax.
    """
    # Fill invalid positions with -inf in-place to avoid extra allocations
    scores = torch.where(mask, scores, float("-inf"))
    scores = F.softmax(scores, dim=dim)
    scores = torch.where(mask, scores, 0.0)  # scores.masked_fill(~mask, 0)
    return scores


# @torch.compile


def build_softmax_mask(mask):
    # [..., N, 1], True where valid
    mask = mask & mask.squeeze(-1).unsqueeze(-2)  # outer product
    return mask


class Identity(nn.Module):
    """Returns the identity, but is able to take more than one input. This enables better construction of Sequential operations."""

    def forward(self, x, *args, **kwargs):
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def get_norm_layer(norm_type, dim):
    if norm_type in ["none", "identity"]:
        norm = nn.Identity()  # nn.GroupNorm(groups, dim_out)
    elif norm_type.lower() == "rmsnorm":
        norm = nn.RMSNorm(dim)
    else:
        assert 0, f"Norm type {norm_type} not implemented!"
    return norm


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))

    def forward(self, x):
        # normalize divides by maximum norm element. Different from original in which we take the max norma nd not the sum of square elem.
        return F.normalize(x, dim=-1) * self.g * (x.shape[-1] ** 0.5)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


def get_forward_signature(module, depth: int = 0, max_depth: int = 10):
    """
    Recursively get the signature of the innermost module's forward method.
    Traverses through typical wrapper attributes like .fn, .module, .net, .layer.
    Stops at max_depth to prevent infinite recursion.
    """
    if depth > max_depth:
        return inspect.signature(module.forward)

    sig = inspect.signature(module.forward)
    params = sig.parameters

    # If the forward has *args/**kwargs, try to dig into inner modules
    if any(
        p.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        for p in params.values()
    ):
        # Common wrapping attributes in PyTorch wrappers
        for attr_name in ["fn", "module", "net", "layer"]:
            if hasattr(module, attr_name):
                inner = getattr(module, attr_name)
                if hasattr(inner, "forward"):
                    return get_forward_signature(inner, depth + 1, max_depth)

    return sig


class SequentialWithAdj(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)
        self._accepts_adj = []
        self._accepts_mask = []
        self._accepts_attn_bias = []
        self._accepts_time_emb = []

        for module in self:
            sig = get_forward_signature(module)
            params = sig.parameters
            self._accepts_adj.append("adj" in params)
            self._accepts_mask.append("mask" in params)
            self._accepts_attn_bias.append("attn_bias" in params)
            self._accepts_time_emb.append("time_emb" in params)

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        time_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for (
            module,
            accepts_adj,
            accepts_mask,
            accepts_attn_bias,
            accepts_time_emb,
        ) in zip(
            self,
            self._accepts_adj,
            self._accepts_mask,
            self._accepts_attn_bias,
            self._accepts_time_emb,
        ):
            kwargs = {}
            if accepts_adj:
                kwargs["adj"] = adj
            if accepts_mask:
                kwargs["mask"] = mask
            if accepts_attn_bias:
                kwargs["attn_bias"] = attn_bias
            if accepts_time_emb:
                kwargs["time_emb"] = time_emb
            x = module(x, **kwargs)
        return x


def get_n_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad) / 1000000


def model_size_mb(module):
    return sum(p.numel() * p.element_size() for p in module.parameters()) / (1024**2)


# Function to measure memory


