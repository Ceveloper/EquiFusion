from .utils.scheduler import setup_scheduler_step
from .network import EquivariantAutoEncoder
from .trainer import DiffusionTrainer, AutoEncoderTrainer

from .diffusion_manager import DiffusionManager


def get_autoencoder_trainer(*args, **kwargs):
    return AutoEncoderTrainer(*args, **kwargs)

def get_diffusion_trainer(*args, **kwargs):
    return DiffusionTrainer(*args, **kwargs)
