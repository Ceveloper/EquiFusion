from ignite.engine import Engine, Events
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts
from ignite.handlers import create_lr_scheduler_with_warmup


class ExponentialLRSchedulerWarmup(object):
    def __init__(
        self,
        lr: float,
        optimizer,
        warmup_duration: int = 200,
        update_every: int = 1,
        min_lr: float = 0.,
        gamma_decay: float = 0.98,
        if_lr_scheduler_ramp_up: bool = True,
        ramp_up_init_lr: float = 0.0, # 0.0001,
        scheduler_name: str = "ExponentialLR",
        cosannealing_cycle_length: int = 150,
        **kwargs
    ):
        assert min_lr is None or min_lr <= lr
        self.minimal_lr_reached = False
        self.lr = lr
        self.lr_scheduler_warmup = warmup_duration
        self.lr_scheduler_min_lr = min_lr
        self.lr_scheduler_update_every = update_every
        self.optimizer = optimizer
        if scheduler_name == "ExponentialLR":
            self.torch_lr_scheduler = ExponentialLR(
                optimizer=optimizer, gamma=gamma_decay)
        elif scheduler_name == "CosineAnnealingWarmRestarts":
            self.torch_lr_scheduler = CosineAnnealingWarmRestarts(
                optimizer=optimizer, T_0=cosannealing_cycle_length, T_mult=1, eta_min=self.lr_scheduler_min_lr, last_epoch=-1)
        else:
            assert False, f"Scheduler {scheduler_name} not implemented yet"
        self.lr_scheduler = create_lr_scheduler_with_warmup(self.torch_lr_scheduler,
                                                            warmup_start_value=ramp_up_init_lr if if_lr_scheduler_ramp_up else lr,       # CHANGE FLO, was lr before so no warmup
                                                            warmup_end_value=lr,
                                                            warmup_duration=warmup_duration)  # warm-up phase duration, number of events.

    def step(self, engine: Engine):
        if engine.state.epoch <= self.lr_scheduler_warmup:
            self.lr_scheduler(engine)
        else:
            if not self.minimal_lr_reached and self.lr_scheduler_min_lr is not None:
                for param_group in self.optimizer.param_groups:
                    if param_group['lr'] <= self.lr_scheduler_min_lr:
                        param_group['lr'] = self.lr_scheduler_min_lr
                        self.minimal_lr_reached = True
                if engine.state.epoch % self.lr_scheduler_update_every == 0 and not self.minimal_lr_reached:
                    self.lr_scheduler(engine)


def LRScheduler(lr_scheduler_type: str, **kwargs):
    return globals()[lr_scheduler_type](**kwargs)


def setup_scheduler_step(model_trainer, train_engine: Engine):
    if model_trainer.lr_scheduler is not None:
        if isinstance(model_trainer.lr_scheduler, ExponentialLRSchedulerWarmup):
            train_engine.add_event_handler(
                Events.EPOCH_STARTED, model_trainer.lr_scheduler.step)
        else:
            assert False, f"Scheduler {model_trainer.lr_scheduler} not implemented yet"
