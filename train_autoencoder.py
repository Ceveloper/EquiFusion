from src.utils.config import flat_hydra_config, config_exp_folder
from src.core.trainer import setup_scheduler_step
from src.core import get_autoencoder_trainer
from src.utils.tensorboard import setup_tb_logging_valid, setup_tb_logging_train
from src.utils.reproducibility import set_seed
from src.data import create_skeleton, create_motion_repr
from src.train_utils import (
    setup_validation_autoencoder,
    create_train_dataloaders,
    resume_training,
    create_model,
)
from ignite.handlers import Checkpoint, DiskSaver
from ignite.handlers.tensorboard_logger import TensorboardLogger
from ignite.metrics import RunningAverage
from ignite.engine import Events, Engine
from ignite.contrib.handlers import ProgressBar
import os
import socket
import torch
from omegaconf import DictConfig, OmegaConf
import hydra

OmegaConf.register_new_resolver("eval", eval, replace=True)


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def train(
    output_log_path: str,
    num_epochs: int,
    save_frequency: int = 100,
    eval_frequency: int = None,
    num_iteration_eval: int = 0,
    device: str = "cpu",
    seed: int = 52345,
    num_iter_perepoch: int = 580,
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
        f"Created Autoencoder with {str(round(sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000, 3))}M trainable parameters."
    )

    # Load dataset
    (
        data_loader_train,
        data_loader_eval,
        data_loader_train_eval,
        data_loader_generalization_eval,
        _,
        random_state_manager,
    ) = create_train_dataloaders(
        if_run_validation=if_run_validation,
        if_resume_training=if_resume_training,
        **kwargs,
    )

    model_trainer = get_autoencoder_trainer(model, **kwargs)

    def set_epoch_seed(engine: Engine):
        set_seed(seed + engine.state.epoch)
        random_state_manager.reseed_generator(seed + engine.state.epoch)

    def preprocess(engine: Engine):
        # batch = (obs, pred, extra) -> (obs, pred, adjacency) on device
        obs, pred, extra = engine.state.batch
        engine.state.batch = [obs.to(device), pred.to(device), extra["adj"].to(device)]

    # Define Training Engines
    trainer = Engine(model_trainer.train_step)
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, "Loss")

    trainer.add_event_handler(Events.ITERATION_STARTED, preprocess)
    trainer.add_event_handler(Events.EPOCH_STARTED, set_epoch_seed)
    # Set up schedulers
    setup_scheduler_step(model_trainer, train_engine=trainer)

    pbar = ProgressBar()
    pbar.attach(trainer, ["Loss"])

    # Setup tensorboard logging and progressbar
    tb_logger = TensorboardLogger(log_dir=os.path.join(output_log_path, "tb"))
    setup_tb_logging_train(
        tb_logger,
        trainer=trainer,
        optimizer=model_trainer.optimizer,
        model=model,
    )
    if if_run_validation:
        evaluator, evaluator_train, evaluator_generalization = (
            setup_validation_autoencoder(
                model,
                preprocess,
                model_trainer,
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
                "FDE",
                "MPJPE_AVG",
                "MPJPE",
                "BD_StretchMean",
                "BD_JitterMean",
                "BD_StretchRMSE",
                "BD_JitterRMSE",
            ],
            if_consider_hip=False,
        )

    # Define handlers for saving checkpoints
    objects_to_checkpoint = {
        "trainer": trainer,
        "model": model,
        "random_states": random_state_manager,
        **model_trainer.objects_to_checkpoint(),
    }
    checkpoint_handler = Checkpoint(
        objects_to_checkpoint,
        DiskSaver(
            os.path.join(output_log_path, "checkpoints"),
            create_dir=True,
            require_empty=False,
        ),
        n_saved=20,
        global_step_transform=lambda engine, event_name: trainer.state.epoch,
        score_function=lambda engine: -1
        * round(engine.state.metrics["MPJPE_AVG"].item(), 7),
        score_name="val_mpjpe",
    )
    latest_checkpoint_handler = Checkpoint(
        objects_to_checkpoint,
        checkpoint_handler.save_handler,
        n_saved=1,
        global_step_transform=lambda engine, event_name: trainer.state.epoch,
    )
    static_checkpoint_handler = Checkpoint(
        objects_to_checkpoint,
        checkpoint_handler.save_handler,
        n_saved=None,
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

    # Regularly save checkpoints during training: every `save_frequency` epochs, and the latest one
    # every epoch (used to resume after preemption)
    @trainer.on(Events.EPOCH_COMPLETED(every=save_frequency))
    def save_static_model(engine: Engine):
        static_checkpoint_handler(trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_latest_model(engine: Engine):
        latest_checkpoint_handler(trainer)
    # Setup evaluation process between epochs
    if if_run_validation:

        @trainer.on(Events.EPOCH_COMPLETED(every=eval_frequency))
        def train_epoch_completed(engine: Engine):
            set_seed(0)
            random_state_manager.reseed_generator(0)
            evaluator.run(data_loader_eval)
            data_loader_train_eval.dataset.eval()
            evaluator_train.run(data_loader_train_eval, epoch_length=num_iteration_eval)
            data_loader_train_eval.dataset.train()

            set_seed(0)
            evaluator_generalization.run(data_loader_generalization_eval)

            checkpoint_handler(evaluator)
            set_seed(seed + trainer.state.epoch)
            random_state_manager.reseed_generator(seed + trainer.state.epoch)

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
        data_loader_train, epoch_length=num_iter_perepoch, max_epochs=num_epochs
    )
    static_checkpoint_handler(trainer)  # Save the last model after training
    tb_logger.close()


@hydra.main(
    config_path="./configs/config_train_autoencoder",
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    # for backwards compatibility with the code
    cfg = flat_hydra_config(cfg)

    assert cfg["num_workers"] >= 1, "Necessary for reproducibility issues"

    if cfg["if_resume_training"]:
        print("Configuration file will be ignored. Considering old configuration file.")
        cfg = resume_training(cfg)
    else:
        # Start from scratch
        cfg = config_exp_folder(cfg)
    print("Traing data saved at ", cfg["output_log_path"])
    print("Training on device: ", f"{socket.gethostname().split('.')[0]}")
    train(**cfg)


if __name__ == "__main__":
    main()
