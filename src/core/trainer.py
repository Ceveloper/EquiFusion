import torch
from ignite.engine import Engine, Events
from typing import Sequence, Optional
import numpy as np

import math
from ema_pytorch import EMA
from torch.optim import Adam
from .utils.scheduler import (
    LRScheduler,
    ExponentialLRSchedulerWarmup,
    setup_scheduler_step,
)



def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def LRScheduler(lr_scheduler_type: str, **kwargs):
    return globals()[lr_scheduler_type](**kwargs)


def setup_scheduler_step(model_trainer, train_engine: Engine):
    if model_trainer.lr_scheduler is not None:
        if isinstance(model_trainer.lr_scheduler, ExponentialLRSchedulerWarmup):
            train_engine.add_event_handler(
                Events.EPOCH_STARTED, model_trainer.lr_scheduler.step
            )
        else:
            assert False, f"Scheduler {model_trainer.lr_scheduler} not implemented yet"


class AutoEncoderTrainer(object):
    """Trains the autoencoder with the reconstruction loss of the (masked) graph nodes."""

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float,
        clip_grad_norm: float = 1.0,
        use_lr_scheduler: bool = False,
        prediction_horizon_train: int = 120,
        prediction_horizon_eval: int = 120,
        **config,
    ):
        self.model = model
        self.clip_grad_norm = clip_grad_norm
        self.prediction_horizon_train = prediction_horizon_train
        self.prediction_horizon_eval = prediction_horizon_eval

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True)
        self.lr_scheduler = (
            LRScheduler(
                lr=lr, optimizer=self.optimizer, **config["lr_scheduler_kwargs"]
            )
            if use_lr_scheduler
            else None
        )

    def objects_to_checkpoint(self):
        objects_to_checkpoint = {"optimizer": self.optimizer}
        if self.lr_scheduler is not None:
            objects_to_checkpoint["lr_scheduler"] = self.lr_scheduler.lr_scheduler
        return objects_to_checkpoint

    def train_step(self, engine: Engine, batch: Sequence[torch.Tensor]):
        self.model.train()
        self.optimizer.zero_grad()
        x, y, a = batch
        mask = (a.sum(-1) != 0).unsqueeze(-1)
        ph = self.prediction_horizon_train

        pred, *_ = self.model.autoencode(y, past=x, ph=ph, adj=a, mask=mask)
        loss = self.model.loss(pred, y[:, :ph], x=x, mask=mask)

        # The released models were trained with the gradient of 2 * loss (the original code
        # accumulated one extra backward pass while logging gradient norms). With gradient clipping
        # this is not equivalent to a learning-rate change, so we keep it for exact reproducibility.
        (2 * loss).backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

        losses = {"loss_rec": loss, "loss": loss}
        return (
            loss,
            (losses, {"loss_rec": 1.0}, {"loss_rec": loss.item()}, {}),
            pred,
            y,
            None,
        )

    def validation_step(self, engine: Engine, batch: Sequence[torch.Tensor]):
        self.model.eval()
        with torch.no_grad():
            x, y, a = batch
            mask = (a.sum(-1) != 0).unsqueeze(-1)
            model_out, _, z, _, _ = self.model.autoencode(
                y, past=x, ph=self.prediction_horizon_eval, adj=a, mask=mask
            )
        return model_out, y, x, z


class DiffusionTrainer(object):
    """Trains the latent diffusion model on the (frozen) autoencoder latents.

    For each training sample, `train_pick_best_sample_among_k` noisy versions are denoised and only the
    one closest to the ground-truth latent contributes to the loss (k-relaxation, as in SkeletonDiffusion).
    """

    def __init__(
        self,
        diffusion_model: torch.nn.Module,
        *,
        device: "cpu",
        lr=1e-4,
        weight_decay=0.0,
        train_pick_best_sample_among_k=1,
        if_use_ema=True,
        ema_update_every=10,
        ema_decay=0.995,
        ema_power=2 / 3,
        ema_min_value=0.0,
        step_start_ema: 100,
        adam_betas=(0.9, 0.99),
        use_lr_scheduler: bool = False,
        num_samples=25,
        max_grad_norm=1.0,
        skeleton=None,
        autoencoder=None,
        num_prob_samples: int = 50,
        prediction_horizon_eval: int = 120,
        **config,
    ):
        self.device = device
        self.model = diffusion_model
        assert self.model.condition
        self.train_pick_best_sample_among_k = train_pick_best_sample_among_k
        self.skeleton = skeleton
        self.autoencoder = autoencoder
        self.num_prob_samples = num_prob_samples
        self.prediction_horizon_eval = prediction_horizon_eval
        assert has_int_squareroot(num_samples), "number of samples must have an integer square root"
        self.num_samples = num_samples
        self.max_grad_norm = max_grad_norm
        self.opt = Adam(diffusion_model.parameters(), lr=lr, betas=adam_betas, weight_decay=weight_decay)
        self.if_use_ema = if_use_ema
        if if_use_ema:
            self.ema = EMA(
                diffusion_model,
                beta=ema_decay,
                update_every=ema_update_every,
                update_after_step=step_start_ema,
                power=ema_power,
                min_value=ema_min_value,
            )
            self.ema.to(self.device)
        self.lr_scheduler = (
            LRScheduler(lr=lr, optimizer=self.opt, **config["lr_scheduler_kwargs"]) if use_lr_scheduler else None
        )
        self.step = 0

    def set_eval_on_data(self, name: str):
        self.model.set_eval_dataidx(name)
        self.ema.ema_model.set_eval_dataidx(name)

    def get_checkpoint_object(self):
        data = {"model": self.model, "opt": self.opt}
        if self.if_use_ema:
            data["ema"] = self.ema
        if self.lr_scheduler is not None:
            data["lr_scheduler"] = self.lr_scheduler.lr_scheduler
        return data

    def loss(self, data, x_cond=None, adj=None, mask=None, difftrain_kwargs={}):
        """Diffusion loss of the best among k samples (per training example)."""
        loss, diff_weights, samples = self.model(
            data,
            x_cond=x_cond,
            num_samples=self.train_pick_best_sample_among_k,
            adj=adj,
            mask=mask,
            difftrain_kwargs=difftrain_kwargs,
        )
        if self.train_pick_best_sample_among_k > 1:
            b = data.shape[0]
            with torch.no_grad():
                closest2gt_idx = loss.min(axis=0).indices
            loss = torch.gather(loss, dim=0, index=closest2gt_idx.unsqueeze(0)).squeeze(0)
            assert len(loss.shape) == 1 and loss.shape[0] == b
        return (loss * diff_weights).mean()

    def train_step(self, engine: Engine, data: Sequence[torch.Tensor]):
        autoencoder = self.autoencoder
        if self.if_use_ema:
            self.ema.ema_model.train()
        self.model.train()
        self.opt.zero_grad()

        with torch.no_grad():
            autoencoder.eval()
            x, y, a = data
            if isinstance(a, (tuple, list)):
                # permutation of the sample and dataset index, used by the permutable diffusion process
                difftrain_kwargs, a = a
            else:
                difftrain_kwargs = {}
            mask = (a.sum(-1) != 0).unsqueeze(-1)
            _, _, (z_past_vars, z_vars) = autoencoder.get_train_embeddings(y, past=x, adj=a, mask=mask)
            z_past_motion, z_past_bone = z_past_vars
            z_motion, _ = z_vars
        x_cond = self.model.model.get_cond(
            z_past_motion=z_past_motion, z_past_bone=z_past_bone, adj=a, mask=mask
        )
        loss = self.loss(z_motion, x_cond=x_cond, adj=a, mask=mask, difftrain_kwargs=difftrain_kwargs)

        # The released models were trained with the gradient of 2 * loss (the original code accumulated one
        # extra backward pass while logging gradient norms). With gradient clipping this is not equivalent to a
        # learning-rate change, so we keep it for exact reproducibility.
        (2 * loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.opt.step()
        self.opt.zero_grad()
        self.step += 1
        if self.if_use_ema:
            self.ema.update()
        losses = {"loss_diff": loss, "loss": loss}
        return loss, (losses, {"loss_diff": 1.0}, {"loss_diff": loss.item()}, {})

    def decode_diffusion_sample(self, samples, z_b, obs, autoencoder, x_cond=None, adj=None, **kwargs):
        """Decodes latent samples [S, B, N, D] into motions [S, B, T, N, 3]."""
        x_t = obs.expand(samples.shape[0], -1, -1, -1, -1)
        z_b = z_b.expand(samples.shape[0], -1, -1, -1) if len(z_b.shape) == 3 else z_b
        if x_cond is not None:
            if isinstance(x_cond, (tuple, list)):
                x_cond = tuple(xc.expand(samples.shape[0], -1, -1, -1) for xc in x_cond)
            else:
                x_cond = x_cond.expand(samples.shape[0], -1, -1, -1)
        z = autoencoder.aggregate_latent(z_motion=samples, z_bone=z_b)
        out, _ = autoencoder.decode(x_t, z, x_cond, ph=self.prediction_horizon_eval, adj=adj, **kwargs)
        return out, samples

    def validation_step(self, engine: Engine, batch: Sequence[torch.Tensor]):
        autoencoder = self.autoencoder
        if self.if_use_ema:
            self.ema.ema_model.eval()
        self.model.eval()
        with torch.no_grad():
            x, y, a = batch
            mask = (a.sum(-1) != 0).unsqueeze(-1)
            autoencoder.eval()
            _, (z_past_motion, z_past_bone), _ = autoencoder.get_past_embedding(x, adj=a, mask=mask)
            x_cond = self.model.model.get_cond(
                z_past_motion=z_past_motion, z_past_bone=z_past_bone, adj=a, mask=mask
            ).unsqueeze(dim=0)
            model = self.ema.ema_model if self.if_use_ema else self.model
            samples, _ = model.sample(
                batch_size=x.shape[0], num_samples=self.num_prob_samples, x_cond=x_cond, adj=a, mask=mask
            )
            out, samples = self.decode_diffusion_sample(
                samples,
                obs=x,
                z_b=z_past_bone,
                autoencoder=autoencoder,
                x_cond=(z_past_motion, z_past_bone),
                adj=a,
                mask=mask,
            )
            out = out.transpose(0, 1)  # (batch, num_samples, T, J, 3)
            samples = samples.transpose(0, 1)
            return out, y, samples, x
