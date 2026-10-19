from src.utils.load import load_autoencoder
from src.data import create_skeleton, create_motion_repr
from src.utils.config import (
    load_and_merge_autoenc_cfg,
    flat_hydra_config,
    config_exp_folder,
)
from src.core import get_diffusion_trainer
from src.core.diffusion_manager import DiffusionManager
from src.core.trainer import DiffusionTrainer, setup_scheduler_step
from src.train_utils import (
    setup_validation_diffusion,
    create_train_dataloaders,
    resume_training,
    create_model,
)
from src.utils.tensorboard import setup_tb_logging_valid, setup_tb_logging_train
from src.utils.reproducibility import set_seed
from ignite.handlers import Checkpoint, DiskSaver
from ignite.handlers.tensorboard_logger import TensorboardLogger
from ignite.metrics import RunningAverage
from ignite.engine import Events, Engine
from ignite.contrib.handlers import ProgressBar
import os
import socket
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def train(
    output_log_path: str,
    num_epochs: int,
    eval_frequency: int = None,
    num_iteration_eval: int = 0,
    device: str = "cpu",
    seed: int = 63485,
    num_iter_perepoch: int = None,
    detect_anomaly: bool = False,
    if_resume_training: bool = False,
    if_run_validation: bool = False,
    if_measure_time: bool = False,
    **kwargs,
) -> None:
    torch.autograd.set_detect_anomaly(detect_anomaly)

    # Init seed
    set_seed(seed)
    torch.set_default_dtype(
        torch.float64 if kwargs["dtype"] == "float64" else torch.float32
    )
    print(
        f"Default datatype: {torch.get_default_dtype()} . Using device: {device} - {torch.cuda.get_device_name(0).replace(' ', '_') if torch.cuda.is_available() else ''}"
    )
    print("Training on: ", f"{socket.gethostname().split('.')[0]}")

    motion_repr = create_motion_repr(**kwargs)

    # # Create model
    model = create_model(device, num_nodes=kwargs["max_num_nodes"], **kwargs)
    print(
        f"Loaded Autoencoder Model with {str(round(sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000, 3))}M parameters."
    )
    # Load GraphModel
    load_autoencoder(model, **kwargs)

    # Load dataset
    (
        data_loader_train,
        data_loader_eval,
        data_loader_train_eval,
        data_loader_generalization_eval,
        skeletons,
        random_state_manager,
    ) = create_train_dataloaders(
        if_run_validation=if_run_validation,
        if_resume_training=if_resume_training,
        **kwargs,
    )

    diffusionmanager = DiffusionManager(
        skeletons=skeletons,
        datasets=data_loader_train.dataset.datasets,
        num_nodes=kwargs["max_num_nodes"],
        **kwargs,
    )
    diffusion = diffusionmanager.get_diffusion().to(device)

    print(
        f"Created Diffusion Model with  {str(round(sum(p.numel() for p in diffusion.parameters() if p.requires_grad)/1000000, 3))}M trainable parameters."
    )

    diffusion_trainer = get_diffusion_trainer(
        diffusion,
        device=device,
        ema_decay=0.995,  # exponential moving average decay
        skeleton=motion_repr,
        autoencoder=model,
        **kwargs,
    )

    def set_epoch_seed(engine: Engine):
        set_seed(seed + engine.state.epoch)
        random_state_manager.reseed_generator(seed + engine.state.epoch)

    def preprocess(engine: Engine):
        # batch = (obs, pred, extra) -> (obs, pred, adj) on device. During training, `extra` also carries the
        # node permutation of each sample and its dataset name, needed by the permutable diffusion process.
        obs, pred, extra = engine.state.batch
        if not isinstance(extra, dict):  # already preprocessed (handler can be called more than once)
            return
        adj = extra["adj"].to(device)
        if "perm" in extra:
            adj = [{"perm": extra["perm"].to(device), "dataidx": extra["dataset_idx"]}, adj]
        engine.state.batch = [obs.to(device), pred.to(device), adj]

    # Define Training Engines
    trainer = Engine(diffusion_trainer.train_step)
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, "Loss")
    trainer.add_event_handler(Events.ITERATION_STARTED, preprocess)
    trainer.add_event_handler(Events.EPOCH_STARTED, set_epoch_seed)
    # Set up scheduler
    setup_scheduler_step(diffusion_trainer, train_engine=trainer)

    # Setup progressbar for training
    pbar = ProgressBar()
    pbar.attach(trainer, ["Loss"])

    # Setup tensorboard logging
    tb_logger = TensorboardLogger(log_dir=os.path.join(output_log_path, "tb"))
    setup_tb_logging_train(
        tb_logger,
        trainer=trainer,
        optimizer=diffusion_trainer.opt,
        model=diffusion,
    )
    if if_run_validation:
        evaluator, evaluator_train, evaluator_generalization = (
            setup_validation_diffusion(
                model,
                preprocess,
                diffusion_trainer,
                data_loader_eval,
                data_loader_train_eval,
                data_loader_generalization_eval,
                **kwargs,
            )
        )
        pbar.attach(evaluator)
        pbar.attach(evaluator_train)
        pbar.attach(evaluator_generalization)
        setup_tb_logging_valid(
            tb_logger,
            trainer=trainer,
            evaluator=evaluator,
            evaluator_train=evaluator_train,
            evaluator_generalization=evaluator_generalization,
            loss_names=["Loss"],
            metric_names=[
                "ADE",
                "APD",
                "MPJPE_AVG",
                "BD_StretchMean",
                "BD_JitterMean",
                "BD_StretchRMSE",
                "BD_JitterRMSE",
                "MPJPE",
                "LLVar",
                "LearningRate",
            ],
        )

    # Define objecs to save in checkpoint
    objects_to_checkpoint = {
        **diffusion_trainer.get_checkpoint_object(),
        **{"trainer": trainer, "random_states": random_state_manager},
    }
    checkpoint_handler = Checkpoint(
        objects_to_checkpoint,
        DiskSaver(
            os.path.join(output_log_path, "checkpoints"),
            create_dir=True,
            require_empty=False,
        ),
        n_saved=10,
        global_step_transform=lambda engine, event_name: trainer.state.epoch,
        score_function=lambda engine: -1 * round(engine.state.metrics["ADE"], 7),
        score_name="val_ade",
    )
    latest_checkpoint_handler = Checkpoint(
        objects_to_checkpoint,
        checkpoint_handler.save_handler,
        n_saved=1,
        global_step_transform=lambda engine, event_name: trainer.state.epoch,
    )
    if if_resume_training:
        checkpoint = torch.load(kwargs["load_path"], weights_only=False)
        Checkpoint.load_objects(to_load=objects_to_checkpoint, checkpoint=checkpoint)
        print(
            "Resuming training from: epoch = ",
            trainer.state.epoch,
            ". Iteration = ",
            trainer.state.iteration,
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_latest_model(engine: Engine):
        latest_checkpoint_handler(trainer)

    # Setup evaluation process between epochs
    

    @trainer.on(Events.EPOCH_COMPLETED(every=eval_frequency))
    def train_epoch_completed(engine: Engine):
        if if_run_validation:
            set_seed(0)
            random_state_manager.reseed_generator(0)
            diffusion_trainer.set_eval_on_data(
                data_loader_eval.dataset.dataset_names[0]
            )
            evaluator.run(data_loader_eval)
            data_loader_train_eval.dataset.eval()

            diffusion_trainer.set_eval_on_data(
                data_loader_train_eval.dataset.dataset_names[0]
            )
            evaluator_train.run(data_loader_train_eval, epoch_length=num_iteration_eval)
            data_loader_train_eval.dataset.train()
            set_seed(0)

            diffusion_trainer.set_eval_on_data(
                data_loader_generalization_eval.dataset.dataset_names[0]
            )
            evaluator_generalization.run(data_loader_generalization_eval)
            set_seed(seed + trainer.state.epoch)
            random_state_manager.reseed_generator(seed + trainer.state.epoch)
            checkpoint_handler(evaluator)
        else:
            os.makedirs(os.path.join(output_log_path, "checkpoints_static"), exist_ok=True)
            save_path = os.path.join(output_log_path, "checkpoints_static", f"checkpoint_{trainer.state.epoch}.pt")
            torch.save(diffusion.state_dict(), save_path)
            print(f"Saved checkpoint weights to: {save_path}")

    if if_measure_time:

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_epoch_time():
            print(
                f"Epoch {trainer.state.epoch}, Time Taken : {trainer.state.times['EPOCH_COMPLETED']}"
            )

        @trainer.on(Events.COMPLETED)
        def log_total_time():
            print(f"Total Time: {trainer.state.times['COMPLETED']}")

    trainer.run(
        data_loader_train,
        epoch_length=(
            num_iter_perepoch
            if num_iter_perepoch is not None
            else len(data_loader_train)
        ),
        max_epochs=num_epochs,
    )
    tb_logger.close()


@hydra.main(
    config_path="./configs/config_train_diffusion",
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig):

    cfg = flat_hydra_config(cfg)

    assert cfg["num_workers"] >= 1, "Necessary for reproducibility issues"
    if cfg["if_resume_training"]:
        print("Configuration file will be ignored. Considering old configuration file.")
        cfg = resume_training(cfg)
    else:
        # Start from scratch
        cfg = config_exp_folder(cfg)

    cfg = load_and_merge_autoenc_cfg(cfg)
    print("Traing data saved at ", cfg["output_log_path"])
    print("Training on device: ", f"{socket.gethostname().split('.')[0]}")
    train(**cfg)


if __name__ == "__main__":
    main()
