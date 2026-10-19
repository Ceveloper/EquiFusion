import torch
from ignite.engine import Events, Engine
from ignite.metrics import Loss
from typing import Sequence
from functools import partial

from torch.utils.data import DataLoader
from ignite.handlers import Checkpoint
import os

from .utils.config import init_obj
from .utils.reproducibility import (
    seed_worker,
    seed_eval_worker,
    RandomStateDict,
)
from .data.loaders import (
    DynamicKinematicMotionDataset,
    get_DatasetClass_for_evaluation,
)
from .utils.load import get_latest_model_path
from .inference_utils import create_model
from .config_metrics import (
    MeanPerJointPositionError,
    FinalDisplacementError,
    ade,
    MetricStorer,
    get_body_realism_functs,
)
from .config_metrics import (
    apd,
    ade,
    limb_length_variance,
    MetricStorer,
    MeanPerJointPositionError,
    choose_best_sample,
)
from .data import create_skeleton, SkeletonVanilla as Skeleton

from .data import loaders as dataset_type


def create_train_dataloaders(
    batch_size,
    batch_size_eval,
    num_workers,
    if_run_validation=True,
    if_resume_training=False,
    **config
):

    random_state_manager = RandomStateDict()
    if if_resume_training:
        checkpoint = torch.load(config["load_path"], weights_only=False)
        Checkpoint.load_objects(
            to_load={"random_states": random_state_manager}, checkpoint=checkpoint
        )
    skeletons = {
        name: create_skeleton(
            **{
                **{k: v for k, v in config.items() if k != "dataset"},
                **config["dataset"][name],
            }
        )
        for name in config["dataset"].keys()
    }
    train_data_dict = {}
    for name, skel in skeletons.items():
        dataset = init_obj(
            {**config, **config["dataset"][name]},
            "dataset_type",
            dataset_type,
            split="train",
            skeleton=skel,
            **(config["dataset"][name]["data_loader_train"])
        )
        train_data_dict[name] = dataset

    dataset_train = DynamicKinematicMotionDataset(
        datasets_dict=train_data_dict, split="train", skeletons=skeletons, **config
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        pin_memory=True,
        num_workers=num_workers,
        generator=random_state_manager.generator,
    )

    if if_run_validation:
        segments_to_eval = dataset_train.dataset_names[0]
        # alternative for creating dataset_eval

        dataset = init_obj(
            {**config, **config["dataset"][segments_to_eval]},
            "dataset_type",
            dataset_type,
            split="valid",
            skeleton=skeletons[segments_to_eval],
            **(config["dataset"][segments_to_eval]["data_loader_valid"])
        )
        dataset_eval = DynamicKinematicMotionDataset(
            datasets_dict={segments_to_eval: dataset},
            split="valid",
            skeletons={segments_to_eval: skeletons[segments_to_eval]},
            **config
        )
        dataset_eval_train = dataset_train  # dataset_train.datasets[0]
        skeletons_generaliz = {
            n: create_skeleton(
                **{
                    **config["dataset_generalization"][n],
                    **{k: v for k, v in config.items() if k != "dataset"},
                }
            )
            for n in config["dataset_generalization"].keys()
        }
        generaliz_data_dict = {}
        for name, skel in skeletons_generaliz.items():
            dataset = init_obj(
                {**config, **config["dataset_generalization"][name]},
                "dataset_type",
                dataset_type,
                split="valid",
                skeleton=skel,
                **(config["dataset_generalization"][name]["data_loader_valid"])
            )
            generaliz_data_dict[name] = dataset

        dataset_eval_generalization = DynamicKinematicMotionDataset(
            datasets_dict=generaliz_data_dict,
            split="valid",
            skeletons=skeletons_generaliz,
            **config
        )

        data_loader_eval = DataLoader(
            dataset_eval,
            shuffle=False,
            worker_init_fn=seed_eval_worker,
            batch_size=batch_size_eval,
            num_workers=1,
            pin_memory=True,
        )
        data_loader_train_eval = DataLoader(
            dataset_eval_train,
            shuffle=False,
            worker_init_fn=seed_eval_worker,
            batch_size=batch_size_eval,
            num_workers=1,
            pin_memory=True,
        )
        data_loader_generalization_eval = DataLoader(
            dataset_eval_generalization,
            shuffle=False,
            worker_init_fn=seed_eval_worker,
            batch_size=batch_size_eval,
            num_workers=1,
            pin_memory=True,
        )
        skeletons = {**skeletons, **skeletons_generaliz}

    else:
        data_loader_eval = None
        data_loader_train_eval = None
        data_loader_generalization_eval = None
    return (
        data_loader_train,
        data_loader_eval,
        data_loader_train_eval,
        data_loader_generalization_eval,
        skeletons,
        random_state_manager,
    )


def resume_training(cfg):
    # output folder has been already created.
    assert "output_log_path" in cfg

    # decide whether to start from scratch (default if no checkpoints), from latest save (if checkpoints exists), or from given path (if load_path is given)

    assert (
        len(os.listdir(os.path.join(cfg["output_log_path"], "checkpoints"))) != 0
    ), "Checkpoints folder is empty. Please provide a valid path to load from."

    if len(cfg["load_path"]) == 0:
        #  load latest model
        cfg["load_path"] = get_latest_model_path(
            os.path.join(cfg["output_log_path"], "checkpoints")
        )
        print("Loading latest epoch: ", cfg["load_path"].split("/")[-1])
    else:
        output_path = os.path.dirname(os.path.dirname(cfg["load_path"]))
        assert cfg["output_log_path"] == output_path
        cfg["output_log_path"] = output_path
    return cfg

def setup_validation_autoencoder(
    model,
    preprocess,
    model_trainer,
    data_loader_eval,
    data_loader_train_eval,
    data_loader_generalization_eval,
    **kwargs
):
    """Evaluators for autoencoder reconstruction (valid split, a subset of train, and the generalization dataset)."""

    def validation_step(engine: Engine, batch: Sequence[torch.Tensor], skeleton: Skeleton):
        with torch.no_grad():
            model_out, y, x, z = model_trainer.validation_step(engine, batch)
            if skeleton.node_type == "bone":
                # bone directions -> joint positions, using the bone lengths of the observation (prediction)
                # and of the target itself (ground truth)
                y_metric = y[..., : skeleton.num_nozero_joints, :]
                x_metric = x[..., : skeleton.num_nozero_joints, :]
                model_out_metric = model_out[..., : skeleton.num_nozero_joints, :]
                limb_length_x = torch.linalg.norm(x_metric, dim=-1).mean(-2).unsqueeze(-1)
                model_out_metric = skeleton.transform_to_metric_space(model_out_metric, limb_length_x)
                limb_length_y = torch.linalg.norm(y_metric, dim=-1).unsqueeze(-1)
                y_metric = skeleton.transform_to_metric_space(y_metric, limb_length_y)
            else:
                model_out_metric = skeleton.transform_to_metric_space(model_out)
                y_metric = skeleton.transform_to_metric_space(y)
                y_metric = y_metric[..., : skeleton.num_nozero_joints, :]
                model_out_metric = model_out_metric[..., : skeleton.num_nozero_joints, :]
            return (model_out_metric, y_metric), (None, None), (model_out, y), z

    def build_evaluator(data_loader, with_body_realism=True):
        skeleton = data_loader.dataset.datasets[0].skeleton
        engine = Engine(partial(validation_step, skeleton=skeleton))
        if with_body_realism:
            body_realism_metrics = get_body_realism_functs(torch.tensor(skeleton.get_limbseq()), skeleton)
            for name in ["StretchMean", "JitterMean", "StretchRMSE", "JitterRMSE"]:
                MetricStorer(
                    output_transform=lambda x, funct=body_realism_metrics[name]: funct(
                        pred=x[0][0].unsqueeze(1).clone(), target=x[0][1].clone()
                    ),
                    return_op="avg",
                ).attach(engine, "BD_" + name)
        engine.add_event_handler(Events.ITERATION_STARTED, preprocess)
        Loss(model.loss, output_transform=lambda x: (x[0])).attach(engine, "Loss")
        MeanPerJointPositionError(output_transform=lambda x: (x[0])).attach(engine, "MPJPE")
        MetricStorer(output_transform=lambda x: ade(pred=x[0][0].unsqueeze(1), target=x[0][1])).attach(engine, "ADE")
        FinalDisplacementError(output_transform=lambda x: (x[0])).attach(engine, "FDE")

        def add_avg_mpjpe(engine: Engine):
            engine.state.metrics["MPJPE_AVG"] = engine.state.metrics["MPJPE"].mean(0)

        engine.add_event_handler(Events.COMPLETED, add_avg_mpjpe)
        return engine

    evaluator = build_evaluator(data_loader_eval)
    evaluator_train = build_evaluator(data_loader_train_eval)
    evaluator_generalization = build_evaluator(data_loader_generalization_eval)
    return evaluator, evaluator_train, evaluator_generalization


def setup_validation_diffusion(
    model,
    preprocess,
    diffusion_trainer,
    data_loader_eval,
    data_loader_train_eval,
    data_loader_generalization_eval,
    **kwargs
):
    """Evaluators for the diffusion model (valid split, a subset of train, and the generalization dataset)."""

    def validation_step(engine: Engine, batch: Sequence[torch.Tensor], skeleton: Skeleton):
        model_out, y, _, x = diffusion_trainer.validation_step(engine=engine, batch=batch)
        if skeleton.node_type == "bone":
            y_metric = y[..., : skeleton.num_nozero_joints, :]
            model_out_metric = model_out[..., : skeleton.num_nozero_joints, :]
            x_metric = x[..., : skeleton.num_nozero_joints, :]
            limb_length_x = torch.linalg.norm(x_metric, dim=-1).mean(-2).unsqueeze(-1)
            model_out_metric = skeleton.transform_to_metric_space(model_out_metric, limb_length_x.unsqueeze(1))
            limb_length_y = torch.linalg.norm(y_metric, dim=-1).unsqueeze(-1)
            y_metric = skeleton.transform_to_metric_space(y_metric, limb_length_y)
        else:
            model_out_metric = skeleton.transform_to_metric_space(model_out)
            y_metric = skeleton.transform_to_metric_space(y)
            y_metric = y_metric[..., : skeleton.num_nozero_joints, :]
            model_out_metric = model_out_metric[..., : skeleton.num_nozero_joints, :]
        return {"pred": model_out_metric, "target": y_metric}, None, (model_out, y), x

    def build_evaluator(data_loader):
        dataset = data_loader.dataset
        skeleton = dataset.datasets[0].skeleton if hasattr(dataset, "datasets") else dataset.skeleton
        limbseq = torch.tensor(skeleton.get_limbseq())
        engine = Engine(partial(validation_step, skeleton=skeleton))
        engine.add_event_handler(Events.ITERATION_STARTED, preprocess)
        MeanPerJointPositionError(output_transform=lambda x: choose_best_sample(x[0]["pred"], x[0]["target"])).attach(engine, "MPJPE")
        MetricStorer(output_transform=lambda x: apd(**x[0])).attach(engine, "APD")
        MetricStorer(output_transform=lambda x: ade(**x[0])).attach(engine, "ADE")

        def add_avg_metrics(engine: Engine):
            engine.state.metrics["MPJPE_AVG"] = engine.state.metrics["MPJPE"].mean(0)
            engine.state.metrics["LearningRate"] = diffusion_trainer.opt.param_groups[0]["lr"]

        engine.add_event_handler(Events.COMPLETED, add_avg_metrics)
        MetricStorer(
            output_transform=lambda x: limb_length_variance(pred=x[0]["pred"], limbseq=limbseq, mode="mean")
        ).attach(engine, "LLVar")
        Loss(model.loss, output_transform=lambda x: choose_best_sample(*x[-2])).attach(engine, "Loss")
        body_realism_metrics = get_body_realism_functs(limbseq, skeleton)
        for name in ["StretchMean", "JitterMean", "StretchRMSE", "JitterRMSE"]:
            MetricStorer(
                output_transform=lambda x, funct=body_realism_metrics[name]: funct(**x[0]), return_op="avg"
            ).attach(engine, "BD_" + name)
        return engine

    evaluator = build_evaluator(data_loader_eval)
    evaluator_train = build_evaluator(data_loader_train_eval)
    evaluator_generalization = build_evaluator(data_loader_generalization_eval)
    return evaluator, evaluator_train, evaluator_generalization
