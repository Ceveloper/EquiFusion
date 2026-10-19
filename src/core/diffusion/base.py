import math
from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


def identity(t, *args, **kwargs):
    return t


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(1, b, *((1,) * (len(x_shape) - 2)))


def cosine_beta_schedule(timesteps, s=0.008):
    """cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class LatentDiffusion(nn.Module):
    """Gaussian diffusion in the latent space of the autoencoder (one latent vector per graph node), predicting x_0.
    Subclasses define the forward process (q_sample, q_posterior, predict_noise_from_start, loss_funct)."""

    def __init__(self, model: torch.nn.Module, latent_size=96, diffusion_timesteps=10, diffusion_objective="pred_x0",
                 sampling_timesteps=None, diffusion_activation="identity", diffusion_conditioning=False,
                 beta_schedule="cosine", ddim_sampling_eta=0.0, **kwargs):
        super().__init__()
        assert diffusion_objective == "pred_x0", "only the pred_x0 objective is implemented"
        assert beta_schedule == "cosine", "only the cosine beta schedule is implemented"
        self.enable_masking = kwargs.get("enable_masking", False)
        self.activation = torch.nn.Tanh() if diffusion_activation == "tanh" else torch.nn.Identity()
        self.silent = True
        self.condition = diffusion_conditioning
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.seq_length = latent_size
        self.objective = diffusion_objective

        betas = cosine_beta_schedule(diffusion_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.num_timesteps = int(betas.shape[0])
        self.sampling_timesteps = default(sampling_timesteps, self.num_timesteps)
        assert self.sampling_timesteps <= self.num_timesteps
        self.is_ddim_sampling = self.sampling_timesteps < self.num_timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        print("Diffusion is_ddim_sampling: ", self.is_ddim_sampling)

    def set_eval_dataidx(self, *args, **kwargs):
        pass

    # --- noise ---------------------------------------------------------------------------------------
    def get_white_noise(self, x, *args, **kwargs):
        return self.get_noise(x, *args, **kwargs)

    def get_start_noise(self, x, *args, **kwargs):
        return self.get_white_noise(x, *args, **kwargs)

    def get_noise(self, x, *args, **kwargs):
        """x is either a tensor or a shape"""
        if torch.is_tensor(x):
            return torch.randn_like(x)
        return torch.randn(*x, *args, **kwargs)

    # --- to be defined by the diffusion process ---------------------------------------------------------
    def q_sample(self, x_start, t, noise=None):
        raise NotImplementedError

    def q_posterior(self, x_start, x_t, t):
        raise NotImplementedError

    def predict_noise_from_start(self, x_t, t, x0):
        raise NotImplementedError

    def p_combine_mean_var_noise(self, model_mean, model_log_variance, noise):
        raise NotImplementedError

    def loss_funct(self, model_out, target, *args, **kwargs):
        raise NotImplementedError

    # --- denoiser -----------------------------------------------------------------------------------
    def feed_model(self, x, t, x_self_cond=None, x_cond=None, **kwargs):
        if self.condition:
            assert x_cond is not None
            assert len(x_cond.shape) == len(x.shape), f"x_cond must have same number of dimensions as x, but got {x_cond.shape} vs {x.shape}"
            if x.shape[0] > x_cond.shape[0]:
                x_cond = x_cond.expand(x.shape[0], -1, -1, -1)
        model_output = self.model(x, t, x_self_cond, x_cond=x_cond, **kwargs)
        return self.activation(model_output)

    def model_predictions(self, x, t, x_self_cond=None, x_cond=None, clip_x_start=False, rederive_pred_noise=False,
                          **network_kwargs):
        x_start = self.feed_model(x, t, x_self_cond=x_self_cond, x_cond=x_cond, **network_kwargs)
        if clip_x_start:
            x_start = torch.clamp(x_start, min=-1.0, max=1.0)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return ModelPrediction(pred_noise, x_start)

    # --- training -----------------------------------------------------------------------------------
    def p_losses(self, x_start, t, noise=None, x_cond=None, num_samples=1, difftrain_kwargs={}, **network_kwargs):
        """Loss of num_samples noisy versions of x_start [B, N, D] -> ([num_samples, B], loss weights, model output)."""
        b, c, n = x_start.shape
        if x_cond is not None:
            x_cond = x_cond.unsqueeze(0)
        noise = default(noise, self.get_white_noise(x_start, num_samples=num_samples))
        x_start = x_start.unsqueeze(0)
        x = self.q_sample(x_start=x_start, t=t, noise=noise, **difftrain_kwargs)
        model_out = self.feed_model(x, t, x_self_cond=None, x_cond=x_cond, **network_kwargs)
        loss = self.loss_funct(model_out, x_start, t, difftrain_kwargs=difftrain_kwargs)
        if self.enable_masking:
            mask = network_kwargs["mask"].squeeze(-1)
            num_joints = mask.sum(-1)
            loss = loss.mean(-1).sum(-1) / num_joints  # mean over the real (non-padded) nodes
            assert len(loss.shape) == 2, f"loss shape {loss.shape} != 2"
        else:
            loss = loss.mean(dim=tuple(range(2, loss.dim())))  # [num_samples, B]
        return loss, extract(self.loss_weight, t.view(b, -1)[:, 0], loss.shape[0:1]), model_out

    def forward(self, x, *args, x_cond=None, **kwargs):
        b, c, n, device, seq_length, = *x.shape, x.device, self.seq_length
        assert n == seq_length, f"seq length must be {seq_length}"
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, x_cond=x_cond, **kwargs)

    # --- sampling -----------------------------------------------------------------------------------
    def p_mean_variance(self, x, t, x_self_cond=None, x_cond=None, clip_denoised=True, **network_kwargs):
        preds = self.model_predictions(x, t, x_self_cond, x_cond=x_cond, **network_kwargs)
        x_start = preds.pred_x_start
        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond=None, clip_denoised=True, *args, perm=None, **kwargs):
        s, b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, x_self_cond=x_self_cond, clip_denoised=clip_denoised, *args, **kwargs)
        noise = self.get_white_noise(x, perm=perm) if t > 0 else 0.0  # no noise at t == 0
        pred_img = self.p_combine_mean_var_noise(model_mean, model_log_variance, noise)
        return pred_img, x_start, noise, model_mean

    @torch.no_grad()
    def p_sample_loop(self, shape, x_cond=None, start_noise=None, perm=None, **kwargs):
        device = self.betas.device
        if start_noise is not None:
            assert start_noise.shape == shape, f"Shape mismatch: {start_noise.shape} != {shape}"
            img = start_noise
        else:
            img = self.get_start_noise(shape, device=device, perm=perm)
        noise = img.clone()
        x_start = None
        for t in reversed(range(0, self.num_timesteps)):
            self_cond = x_start if self.self_condition else None
            img, x_start, _, _ = self.p_sample(img, t, self_cond, x_cond=x_cond, perm=perm, **kwargs)
        return img, noise

    def ddim_sample(self, shape, clip_denoised = True, x_cond=None, start_noise=None, **network_kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(0, total_timesteps-1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        
        if start_noise is not None:
            assert start_noise.shape == shape
            img = start_noise
            noise = start_noise.clone()
        else:
            img = torch.randn(shape, device=device)
            noise = img.clone()

        x_start = None
        if not self.silent:
            print(f"Evaluation with {len(time_pairs)} diffusion steps")
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, x_cond=x_cond, clip_x_start = clip_denoised, **network_kwargs)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sqrt_alpha_next = self.sqrt_alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt() #eta*beta_t_tilde
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * sqrt_alpha_next + \
                  c * pred_noise + \
                  sigma * noise
        return img, noise

    @torch.no_grad()
    def sample(self, batch_size=16, num_samples=50, *args, **kwargs):
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        shape = (num_samples, batch_size, self.channels, self.seq_length)
        return sample_fn(shape, *args, **kwargs)
