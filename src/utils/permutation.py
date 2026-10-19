import torch


def random_node_permutation(mask: torch.Tensor, generator: torch.Generator = None) -> torch.Tensor:
    """Permutation matrices [B, N, N] permuting the real nodes of each sample (padded nodes stay at the end).
    mask: [B, N] real nodes. With a (CPU) generator, the global random stream is not consumed."""
    B, N = mask.shape
    perm = torch.zeros(B, N, N, device=mask.device)
    for b in range(B):
        n = int(mask[b].sum())
        idx = torch.randperm(n, generator=generator) if generator is not None else torch.randperm(n, device=mask.device)
        perm[b, torch.arange(n), idx.to(mask.device)] = 1
        perm[b, torch.arange(n, N), torch.arange(n, N)] = 1
    return perm


def permute_nodes(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """x: [B, ..., N, D] -> P x (permutes the node dimension of each sample); P^T (perm.mT) undoes it."""
    idx = perm.argmax(-1)  # [B, N]: row i of P selects node idx[i]
    idx = idx.view(idx.shape[0], *([1] * (x.dim() - 3)), idx.shape[1], 1).expand(*x.shape[:-1], x.shape[-1])
    return torch.gather(x, -2, idx)
