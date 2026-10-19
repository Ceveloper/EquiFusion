import torch
from ignite.handlers.tensorboard_logger import TensorboardLogger, WeightsScalarHandler, WeightsHistHandler, GradsHistHandler, GradsScalarHandler
from ignite.engine import Engine, Events
from torch.optim import Optimizer



def setup_tb_logging_valid(tb_logger: TensorboardLogger, trainer: Engine, evaluator: Engine, evaluator_train: Engine, evaluator_generalization: Engine,
                  loss_names=["Loss"], metric_names=[ "ADE", "FDE", "MPJPE_AVG", "MPJPE"], if_consider_hip=False):
    
    if if_consider_hip:
        metric_names += ['ADE_POSE', 'ADE_HIP'] if 'ADE' in metric_names else []
        metric_names += ['FDE_POSE', 'FDE_HIP'] if 'FDE' in metric_names else []
        metric_names += ['MPJPE_POSE', "MPJPE_HIP"] if 'MPJPE' in metric_names else []
        loss_names += ['Loss Pose'] if 'Loss' in loss_names else [] # , 'Loss Hip'

    tb_custom_scalar_layout = {}
    #　We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch of the
    # `trainer` instead of `evaluator`.
    tb_logger.attach_output_handler(
        evaluator_train,
        event_name=Events.COMPLETED,
        **{'tag': "losses/train",
        'metric_names':   loss_names,
        'global_step_transform': lambda engine, event_name: trainer.state.epoch 
        }
    )
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.COMPLETED,
        **{'tag': "losses/validation",
        'metric_names': loss_names,
        'global_step_transform': lambda engine, event_name: trainer.state.epoch
        }
    )
        
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.COMPLETED,
        **{'tag': "validation",
        'metric_names': loss_names + metric_names,
        'global_step_transform': lambda engine, event_name: trainer.state.epoch 
        }
    )

    tb_logger.attach_output_handler(
        evaluator_train,
        event_name=Events.COMPLETED,
        **{'tag': "validation_train",
        'metric_names': loss_names + metric_names,
        'global_step_transform': lambda engine, event_name: trainer.state.epoch 
        }
    )

    tb_logger.attach_output_handler(
        evaluator_generalization,
        event_name=Events.COMPLETED,
        **{'tag': "validation_generalization",
        'metric_names': loss_names + metric_names,
        'global_step_transform': lambda engine, event_name: trainer.state.epoch 
        }
    )

    tb_custom_scalar_layout = {
        **tb_custom_scalar_layout,
        **{
            'Validation Metrics': {
                'MPJPE_T': ['Multiline', [rf"validation_train/MPJPE_T/.*"]],
            }
        }
    }
    tb_logger.writer.add_custom_scalars(tb_custom_scalar_layout)

def setup_tb_logging_train(tb_logger: TensorboardLogger, trainer: Engine, optimizer: Optimizer,
                                 model: torch.nn.Module):
    """
    Logs standard information to tensorboard during training.
    """
    # Attach the logger to the trainer to log training loss at each iteration
    
    def _setup_tb_train_curves(
        **kwargs,
    ):
        # trainer returns  loss, (loss_dict, loss_weights, raw_losses, grad_norms),  pred,  y, None
        def funct_losses(out):
            return out[1][0]
        def funct_weights(out):
            return out[1][1]
        def funct_raw_losses(out):
            return out[1][2]
        def funct_grad_norms(out):
            return out[1][3]

        return funct_losses, funct_weights, funct_raw_losses, funct_grad_norms
    
    funct_losses, funct_weights, funct_raw_losses, funct_grad_norms = _setup_tb_train_curves()

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        **{
            'tag': 'training',
            'output_transform': funct_losses,
        }
    )

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        **{
            'tag': 'losses_raw',
            'output_transform': funct_raw_losses,
        }
    )

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        **{
            'tag': 'loss_weights',
            'output_transform': funct_weights,
        }
    )

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        **{
            'tag': 'grad_norms',
            'output_transform': funct_grad_norms,
        }
    )

    # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
    tb_logger.attach_opt_params_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        **{
            'optimizer': optimizer,
            'param_name': 'lr',
            'tag': 'training'
        }
    )
    
    # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
    tb_logger.attach_opt_params_handler(
        trainer,
        event_name=Events.EPOCH_STARTED,
        **{
            'optimizer': optimizer,
            'param_name': 'lr',
            'tag': 'losses'
        }
    )

    # Attach the logger to the trainer to log model's weights norm after each iteration
    tb_logger.attach(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=100),
        log_handler=WeightsScalarHandler(model)
    )

    # Attach the logger to the trainer to log model's weights as a histogram after each epoch
    tb_logger.attach(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        log_handler=WeightsHistHandler(model)
    )

    # Attach the logger to the trainer to log model's gradients norm after each iteration
    tb_logger.attach(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=100),
        log_handler=GradsScalarHandler(model)
    )

    # Attach the logger to the trainer to log model's gradients as a histogram after each epoch
    tb_logger.attach(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        log_handler=GradsHistHandler(model)
    )
