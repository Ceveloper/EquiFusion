"""Tests that EquiFusion is equivariant to permutations of the skeleton graph nodes.

For each test sample, the graph nodes (joints for the 3D representation, bones for the bone direction representation)
are randomly permuted: observation x -> P x, adjacency A -> P A P^T. The autoencoder latents and the full prediction
(encoder -> latent diffusion -> decoder, with the diffusion noise permuted by the same P) must satisfy
f(P x, P A P^T) = P f(x, A) up to floating-point reordering.

With the same (permuted) noise the model is deterministic and equivariance holds for every sample. With independent noise
it holds on a distribution level for all predictions. This latter case is evaluated with eval.py random_permutation=input (see README).

    python test_equivariance.py checkpoint_path=<path/to/checkpoint.pt> dataset=amass            # one batch
    python test_equivariance.py checkpoint_path=<...> dataset=h36m num_batches=-1 batch_size=64  # whole test set
"""
import os
from functools import partial

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.utils.reproducibility import set_seed
from src.utils.config import merge_cfg, flat_hydra_config
from src.eval_utils import prepare_eval_dataset
from src.eval_prepare_model import prepare_model, get_prediction, load_model_config_exp
from src.utils.permutation import random_node_permutation, permute_nodes

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
OmegaConf.register_new_resolver("eval", eval, replace=True)


def check_batch(obs, adj, model, config, num_samples, seed):
    autoencoder, diffusion = model
    mask = adj.sum(-1) != 0
    perm = random_node_permutation(mask)
    obs_p, adj_p = permute_nodes(obs, perm), perm @ adj @ perm.mT

    # 1) autoencoder: per-node latents of the observation
    _, (z, _), _ = autoencoder.get_past_embedding(obs, adj=adj, mask=mask.unsqueeze(-1))
    _, (z_p, _), _ = autoencoder.get_past_embedding(obs_p, adj=adj_p, mask=mask.unsqueeze(-1))
    err_latent = (z_p - permute_nodes(z, perm)).abs().max().item()

    # 2) full prediction with the same (permuted) diffusion noise
    fixed = ("num_samples", "pred_length", "extra", "adj", "model", "sampler_kwargs")
    predict = partial(get_prediction, model=model, num_samples=num_samples, pred_length=config["pred_length"],
                      extra={}, **{k: v for k, v in config.items() if k not in fixed})
    set_seed(seed)
    pred = predict(obs, adj=adj)                                          # [B, S, T, N, 3]
    set_seed(seed)
    pred_p = predict(obs_p, adj=adj_p, sampler_kwargs={"perm": perm})
    err_pred = (pred_p - permute_nodes(pred, perm)).abs().amax(dim=(-1, -2, -3)).flatten().cpu()   # of each prediction
    scale = pred.abs().max().item()
    return err_latent, err_pred, scale


@hydra.main(config_path="./configs/config_eval", config_name="config", version_base="1.3.2")
def main(cfg: DictConfig):
    cfg = flat_hydra_config(OmegaConf.to_container(cfg, resolve=True), to_flat_k_list=["dataset"])
    num_batches = cfg.get("num_batches", 1)
    num_samples = cfg.get("num_samples", 5)
    tolerance = cfg.get("tolerance", 1e-4)
    # In float32, a difference of ~1e-7 (reordered sums) is in rare cases amplified by the sampling steps into a different
    # sample (about 1 prediction in 100,000 for the released models): the test allows this fraction of predictions.
    tolerance_fraction = cfg.get("tolerance_fraction", 1e-4)
    set_seed(cfg["seed"])

    cfg_model, _ = load_model_config_exp(cfg["checkpoint_path"])
    cfg_model.pop("dataset", None)
    config = merge_cfg(cfg, cfg_model)
    config = merge_cfg(config, config["method_specs"])
    data_loader, dataset, skeleton = prepare_eval_dataset(config, split=config["dataset_split"], drop_last=False,
                                                          num_workers=0, batch_size=config["batch_size"],
                                                          stats_mode="deterministic")
    model, device, *_ = prepare_model(config, skeleton, **config)

    worst_latent, errs_pred = 0.0, []
    with torch.no_grad():
        for i, (obs, _, extra) in enumerate(data_loader):
            if num_batches > 0 and i >= num_batches:
                break
            err_latent, err_pred, scale = check_batch(obs.to(device), extra["adj"].to(device), model, config,
                                                      num_samples, seed=config["seed"] + i)
            worst_latent = max(worst_latent, err_latent)
            errs_pred.append(err_pred)
            print(f"batch {i}: max |latent(Px) - P latent(x)| = {err_latent:.2e}, "
                  f"|pred(Px) - P pred(x)|: median {err_pred.median():.2e}, max {err_pred.max():.2e} (prediction scale {scale:.2f})")
    errs_pred = torch.cat(errs_pred)
    above = (errs_pred > tolerance).float().mean().item()
    ok = worst_latent < tolerance and above <= tolerance_fraction
    print(f"{'PASSED' if ok else 'FAILED'}: permutation equivariance up to {tolerance:g} (max latent error {worst_latent:.2e}; "
          f"prediction error: median {errs_pred.median():.2e}, max {errs_pred.max():.2e}, "
          f"{int((errs_pred > tolerance).sum())} of {len(errs_pred)} predictions above the tolerance)")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
