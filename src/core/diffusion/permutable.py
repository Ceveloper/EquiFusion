from typing import Dict

import torch
from torch.amp import autocast

from .base import LatentDiffusion, extract, default


def extract_node_scale(scale: torch.Tensor, t: torch.Tensor, perm_idxs: torch.Tensor = None):
    """scale: [T, N] (all samples from one skeleton) or [B, T, N] (per-sample skeletons) -> [1, B, N, 1]
    at timesteps t, with the nodes permuted like the input sample if perm_idxs [B, N] is given."""
    if scale.dim() == 2:
        out = scale[t]  # [B, N]
    else:
        out = scale[torch.arange(len(t), device=t.device), t]  # [B, N]
    if perm_idxs is not None:
        out = torch.gather(out, -1, perm_idxs)
    return out.unsqueeze(0).unsqueeze(-1)


def apply_permutation_on_vec(vec, perm):
    return (vec.transpose(-1, -2) @ perm.mT).transpose(-2, -1)


class PermutableGaussianDiffusion(LatentDiffusion):
    """Isotropic Gaussian latent diffusion over the nodes of skeleton graphs of different sizes.

    Differs from SkeletonDiffusion's NonisotropicGaussianDiffusion: the noise is isotropic (i.i.d. per node),
    so the diffusion process is equivariant to permutations of the graph nodes. Skeletons with fewer nodes
    than `num_nodes` are zero-padded. All coefficients are per-node scales [T, N]; they are equal to the
    diagonals of the (identity-covariance) matrices of the released models' implementation, and nodes
    beyond a dataset's skeleton size keep the padding value of that implementation (they carry zeros).
    """

    def __init__(self, num_nodes_per_dataset: Dict[str, int], num_nodes: int, loss_reduction_type: str = "l1",
                 **kwargs):
        super().__init__(**kwargs)
        assert self.objective == "pred_x0", "only the pred_x0 objective is implemented"
        self.enable_masking = True
        self.loss_reduction_type = loss_reduction_type
        self.dataset_names = sorted(num_nodes_per_dataset.keys())
        self.data_name2idx = {name: i for i, name in enumerate(self.dataset_names)}

        # Variance schedule of one node (float32 arithmetic of the original implementation)
        alphas = 1.0 - self.betas
        var_t = 1 - alphas                                   # variance added at step t
        var_bar_t = 1 - self.alphas_cumprod                  # variance of x_t given x_0
        var_bar_t_prev = torch.cat([torch.zeros(1), var_bar_t[:-1]])
        posterior_var = var_t * var_bar_t_prev * (1 / var_bar_t)
        per_step = {  # name: (scale per timestep, value on padded nodes)
            "sqrt_var_bar": (torch.sqrt(var_bar_t), 1.0),                        # x_t = sqrt(ac) x0 + sqrt(var_bar) eps
            "recip_sqrt_var_bar": (1 / torch.sqrt(var_bar_t), 1.0),              # eps from (x_t, x0)
            "sqrt_ac_recip_sqrt_var_bar": ((1 / torch.sqrt(var_bar_t)) * self.sqrt_alphas_cumprod, 1.0),
            "mahalanobis": (torch.sqrt(1.0 / var_bar_t), 1.0),                   # loss weighting of the difference
            "posterior_coef1": (torch.sqrt(self.alphas_cumprod_prev) * ((1 / var_bar_t) * var_t), 1.0),
            "posterior_coef2": (torch.sqrt(alphas) * ((1 / var_bar_t) * var_bar_t_prev), 1.0),
            "posterior_log_variance_clipped": (torch.log(posterior_var.clamp(min=1e-20)), 0.0),
        }
        for name, (per_t, pad_value) in per_step.items():
            scales = []
            for dname in self.dataset_names:
                s = torch.full((len(per_t), num_nodes), pad_value, dtype=torch.float32)
                s[:, : num_nodes_per_dataset[dname]] = per_t.unsqueeze(-1)
                scales.append(s)
            self.register_buffer(name, torch.stack(scales), persistent=False)  # [D, T, N]
        self.register_buffer("loss_weight", self.alphas_cumprod.clone(), persistent=False)  # objective pred_x0
        self.eval_idx = 0

    def set_eval_dataidx(self, name: str, device="cuda"):
        """At inference, all samples of a batch come from the same dataset (skeleton)."""
        self.eval_idx = self.data_name2idx[name]

    def _scale(self, name, t, dataidx=None, perm=None):
        scale = getattr(self, name)
        if dataidx is None:
            scale = scale[self.eval_idx]  # [T, N]
        else:
            scale = scale[torch.tensor([self.data_name2idx[d] for d in dataidx], device=t.device)]  # [B, T, N]
        perm_idxs = None if perm is None else perm.argmax(-1)  # permutation matrix -> index vector
        return extract_node_scale(scale, t, perm_idxs)

    # --- noise -------------------------------------------------------------------------------------
    def get_white_noise(self, x_start, *args, num_samples=None, perm=None, **kwargs):
        mask = x_start.sum(-1) == 0  # padded nodes stay zero
        if num_samples is not None:
            noise = self.get_noise((num_samples, *x_start.shape), *args, device=x_start.device)
            noise[:, mask] = 0
        else:
            noise = self.get_noise(x_start.shape, *args, device=x_start.device)
            noise[mask] = 0
        if perm is not None:
            noise = apply_permutation_on_vec(noise, perm)
        return noise

    def get_start_noise(self, shape, mask=None, *args, perm=None, **kwargs):
        noise = self.get_noise(shape, *args, **kwargs)
        if perm is not None:
            noise = apply_permutation_on_vec(noise, perm)
        if mask is not None:
            noise = noise * mask
        return noise

    # --- forward process ---------------------------------------------------------------------------
    @autocast("cuda", enabled=False)
    def q_sample(self, x_start, t, noise=None, perm=None, dataidx=None):
        noise = default(noise, lambda: self.get_white_noise(x_start))
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + \
            self._scale("sqrt_var_bar", t, dataidx, perm) * noise

    def predict_noise_from_start(self, x_t, t, x0):
        return self._scale("recip_sqrt_var_bar", t) * x_t - self._scale("sqrt_ac_recip_sqrt_var_bar", t) * x0

    def loss_funct(self, model_out, target, t, difftrain_kwargs={}):
        loss = (self._scale("mahalanobis", t, **difftrain_kwargs) * (model_out - target)).abs()
        if self.loss_reduction_type == "mse":
            loss = loss ** 2
        return loss

    # --- reverse process ---------------------------------------------------------------------------
    def q_posterior(self, x_start, x_t, t):
        mean = self._scale("posterior_coef1", t) * x_start + self._scale("posterior_coef2", t) * x_t
        log_variance = self._scale("posterior_log_variance_clipped", t)
        return mean, log_variance.exp(), log_variance

    def p_combine_mean_var_noise(self, model_mean, posterior_log_variance, noise):
        return model_mean + (0.5 * posterior_log_variance).exp() * noise

    @torch.no_grad()
    def sample(self, batch_size=16, num_samples=50, *args, mask=None, perm=None, **kwargs):
        shape = (num_samples, batch_size, self.channels, self.seq_length)
        start_noise = self.get_start_noise(shape, mask=mask, device=self.betas.device, perm=perm)
        return self.p_sample_loop(shape, start_noise=start_noise, *args, mask=mask, perm=perm, **kwargs)
