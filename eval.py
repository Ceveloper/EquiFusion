import hydra
from omegaconf import DictConfig, OmegaConf
import os
import json
import yaml
import time
import argparse
import pandas as pd
from functools import partial
import numpy as np
import torch
from ignite.engine import Engine
from ignite.engine import Events, Engine
from ignite.contrib.handlers import ProgressBar

from src.utils.reproducibility import set_seed
from src.metrics.utils import draw_table
from src.config_metrics import attach_engine_to_metrics
from src.utils.config import merge_cfg, flat_hydra_config
from src.utils.store import SequenceStorer
from src.utils.time import AverageTimer
from src.eval_utils import prepare_eval_dataset
from src.utils.permutation import random_node_permutation, permute_nodes


NDEBUG = False

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def compute_metrics(
    dataset_split,
    store_folder,
    batch_size,
    num_samples=50,
    if_measure_time=False,
    prepare_model=None,
    get_prediction=None,
    process_evaluation_pair=None,
    stats_mode="probabilistic",
    metrics_at_cpu=False,
    if_store_output=False,
    if_store_gt=False,
    store_output_path=None,
    store_gt_path=None,
    random_permutation="none",
    **config,
):
    assert random_permutation in ("none", "input", "input_and_noise"), random_permutation

    torch.set_default_dtype(
        torch.float64 if config["dtype"] == "float64" else torch.float32
    )
    data_loader, dataset, skeleton = prepare_eval_dataset(
        config,
        split=dataset_split,
        drop_last=False,
        num_workers=0,
        batch_size=batch_size,
        dataset=None,
        stats_mode=stats_mode,
    )
    if store_folder is not None:
        store_folder = os.path.join(
            store_folder, f"obs{dataset.obs_length}pred{dataset.pred_length}"
        )
        os.makedirs(store_folder, exist_ok=True)
    model, device, *_ = prepare_model(config, skeleton)
    print("Computing metrics at ", "cpu." if metrics_at_cpu else "gpu.")
    pred_storer = (
        SequenceStorer(store_output_path, num_samples, len(dataset), config)
        if if_store_output
        else None
    )
    gt_storer = (
        SequenceStorer(store_gt_path, num_samples, len(dataset), config, if_gt=True)
        if if_store_gt
        else None
    )
    timer = AverageTimer() if if_measure_time else None

    def preprocess(engine: Engine):
        def mmgt_to_device(extradict):
            if "mm_gt" in extradict:
                extradict["mm_gt"] = [mmgt.to(device) for mmgt in extradict["mm_gt"]]
            return extradict

        engine.state.batch = [t.to(device) for t in engine.state.batch[:2]] + [
            engine.state.batch[2]["adj"].to(device),
            mmgt_to_device(engine.state.batch[2]),
        ]

    perm_generator = torch.Generator()

    def set_epoch_seed(engine: Engine):
        set_seed(config["seed"])
        perm_generator.manual_seed(config["seed"])  # separate stream: the diffusion noise is the same as without permutation

    def finalize_storing(engine: Engine):
        if if_store_gt:
            gt_storer.finalize_store()
        if if_store_output:
            pred_storer.finalize_store()
        if if_measure_time:
            timer.print_avg()

    def store_step(data, target, pred, extra, dataset):
        if if_store_output:
            pred_storer.store_batch(pred, extra, dataset)
        if if_store_gt:
            gt_storer.store_batch((target, data), extra, dataset)

    def process_function(engine, batch):
        with torch.no_grad():
            data, target, adj, extra = batch
            if if_measure_time:
                timer.start()
            data_in, adj_in, sampler_kwargs = data, adj, {}
            if random_permutation != "none":
                # random permutation of the graph nodes of each sample (observation and adjacency consistently);
                # with "input_and_noise" the diffusion noise is permuted too. The prediction is permuted back.
                perm = random_node_permutation(adj.sum(-1) != 0, generator=perm_generator)
                data_in, adj_in = permute_nodes(data, perm), perm @ adj @ perm.mT
                if random_permutation == "input_and_noise":
                    sampler_kwargs = {"sampler_kwargs": {**config.get("sampler_kwargs", {}), "perm": perm}}
            # [batch_size, n_samples, seq_length, num_joints, features]
            pred = get_prediction(
                data_in,
                model,
                num_samples=num_samples,
                pred_length=config["pred_length"],
                extra=extra,
                adj=adj_in,
                **sampler_kwargs,
            )
            if random_permutation != "none":
                pred = permute_nodes(pred, perm.mT)
            if if_measure_time:
                timer.end()
            target, pred, mm_gt, data = process_evaluation_pair(
                dataset,
                target=target,
                pred_dict={
                    "pred": pred,
                    "obs": data,
                    "mm_gt": extra["mm_gt"] if "mm_gt" in extra else None,
                },
                adj=adj,
                extra=extra,
            )

            store_step(data, target, pred, extra, dataset)
            if metrics_at_cpu:
                pred = pred.detach().cpu()
                target = target.detach().cpu()
                mm_gt = [mmgt.detach().cpu() for mmgt in mm_gt]
            outdict = {
                "pred": pred,
                "target": target,
                "extra": extra,
                "mm_gt": mm_gt,
                "obs": data,
            }
            return outdict

    engine = Engine(process_function)
    engine.add_event_handler(Events.ITERATION_STARTED, preprocess)
    engine.add_event_handler(Events.EPOCH_STARTED, set_epoch_seed)
    engine.add_event_handler(Events.EPOCH_COMPLETED, finalize_storing)
    pbar = ProgressBar()
    pbar.attach(engine)

    attach_engine_to_metrics(
        engine=engine,
        dataset_split=dataset_split,
        stats_mode=stats_mode,
        dataset=dataset,
        skeleton=skeleton,
        **config,
    )

    if NDEBUG:
        engine.run(data_loader, max_epochs=1, epoch_length=1)
    else:
        engine.run(data_loader)
    results = engine.state.metrics

    # ----------------------------- Printing results -----------------------------
    print("=" * 80)
    table = draw_table(results)
    print(table)
    for stats in results:
        print(f"Total {stats}: {results[stats]:.4f}")
    print("=" * 80)
    # ----------------------------- Storing overall results -----------------------------
    for key, value in results.items():
        results[key] = float(value)
    suffix = "" if random_permutation == "none" else f"_randomperm_{random_permutation}"
    ov_path = os.path.join(store_folder, f"results_{num_samples}_{stats_mode}{suffix}.yaml")
    with open(ov_path, "w") as f:
        yaml.dump(results, f, indent=4)

    print(f"Overall results saved to {ov_path}")
    print("=" * 80)


# model specific options go to hydra
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    config_path="./configs/config_eval", config_name="config", version_base="1.3.2"
)
def main_hydra(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    # for backwards compatibility with the code

    cfg = flat_hydra_config(cfg, to_flat_k_list=["dataset"])
    # if 'motion_repr_type' not in cfg:
    main(**cfg)


def main(
    checkpoint_path,
    method_name="EquiFusion",
    dataset_split="test",
    seed=0,
    stats_mode="probabilistic",
    **cfg,
):
    """setup"""
    set_seed(seed)

    # build the config/checkpoint path
    if "baseline" in method_name.lower():
        checkpoint_path = f"{cfg['method_specs']['baseline_out_path']}/{cfg['task_name']}/{method_name.lower().replace('baseline', '')}/{cfg['dataset_name']}"
    else:
        assert checkpoint_path.endswith((".pt", ".safetensors")), "Path should point to model weights (.pt or .safetensors)"
        if not os.path.exists(checkpoint_path):
            raise Exception(f"Checkpoint not found in: %s" % checkpoint_path)

    # decide from which file to load the functions depending on which model we are evaluating
    if method_name == "EquiFusion":
        from src.eval_prepare_model import (
            prepare_model,
            get_prediction,
            process_evaluation_pair,
            load_model_config_exp,
            get_eval_out_folder,
        )
    elif "baseline" in method_name.lower():
        from src.eval_prepare_algorithmic_baseline import (
            prepare_model,
            get_prediction,
            process_evaluation_pair,
            load_model_config_exp,
            get_eval_out_folder,
        )
    else:
        raise NotImplementedError()

    # load config
    cfg_orig, exp_folder = load_model_config_exp(checkpoint_path)
    # merge original experiment config with the current evaluation config
    cfg_orig.pop("dataset", None)
    cfg = merge_cfg(cfg, cfg_orig)
    cfg = merge_cfg(cfg, cfg["method_specs"])
    cfg["seed"] = seed

    # set up evaluation functions
    # can also be done method dependent wise
    # check wheter we have later conflict because of method specs
    prepare_model = partial(prepare_model, **cfg)
    get_prediction = partial(get_prediction, **cfg)

    stats_folder = get_eval_out_folder(exp_folder, checkpoint_path, dataset_split, cfg)
    with open(os.path.join(stats_folder, "eval_config.yaml"), "w") as config_file:
        OmegaConf.save(cfg, config_file)

    print("Experiment data loaded from ", exp_folder)

    data_loader_name = f"data_loader_{dataset_split}"

    print(f"> Dataset: '{cfg['dataset_name']}'")
    print(f"> Exp name: '{exp_folder.split('/')[-1]}'")
    print(f"> Checkpoint: '{checkpoint_path.split('/')[-1]}'")
    print(f"> Prediction Horizon: '{cfg['pred_length']}'")

    print(
        f"[WARNING] Remember: batch_size has an effect over the randomness of results. Keep batch_size fixed for comparisons, or implement several runs with different seeds to reduce stochasticity."
    )

    t0 = time.time()
    compute_metrics(
        dataset_split=dataset_split,
        stats_mode=stats_mode,
        prepare_model=prepare_model,
        get_prediction=get_prediction,
        process_evaluation_pair=process_evaluation_pair,
        store_folder=stats_folder,
        checkpoint_path=checkpoint_path,
        **cfg,
    )
    tim = int(time.time() - t0)
    print(f"[INFO] Evaluation took {tim // 60}min, {tim % 60}s.")


if __name__ == "__main__":
    main_hydra()
