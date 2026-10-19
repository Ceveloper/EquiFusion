import torch
import ignite
import numpy as np
import random
import os


def set_seed(seed):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # WARNING: if cudnn.enabled=False => greatly reduces training/inference speed.
    torch.backends.cudnn.enabled = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    ignite.utils.manual_seed(seed)
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
    torch.manual_seed(worker_seed)
    ignite.utils.manual_seed(worker_seed)
    
def seed_eval_worker(worker_id):
    
    worker_seed = 0 # set to zero for evaluation
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
    torch.manual_seed(worker_seed)
    ignite.utils.manual_seed(worker_seed)
    
def get_torch_generator():
    g = torch.Generator()
    g.manual_seed(torch.initial_seed())
    return g

class RandomStateDict():
    """
    According to https://pytorch.org/docs/stable/data.html#data-loading-randomness under Multi-process data loading, there is no need to set seeds inbetween an epoch.
    <<Workers are shut down once the end of the iteration is reached, or when the iterator becomes garbage collected.>>
    I suppose seeds are colelcted from environment, so setting up environment seeds aftr lading or resuming training should be sufficient to ensure reproducibility
    """
    def __init__(self):
        self.generator = get_torch_generator()
    
    def reseed_generator(self, seed):
        self.generator.manual_seed(seed)

    def state_dict(self):
        state_dict = {}
        state_dict['random'] = random.getstate()
        state_dict['numpy'] = np.random.get_state()
        state_dict['torch'] = torch.get_rng_state()
        state_dict['torch_random'] = torch.random.get_rng_state()
        state_dict['torch_cuda'] = torch.cuda.get_rng_state(device='cuda')
        state_dict['generator'] = self.generator.get_state()
        return state_dict
    
    def load_state_dict(self, chkpt_obj):
        torch.random.set_rng_state(chkpt_obj['torch_random'])
        torch.set_rng_state(chkpt_obj['torch'])
        self.generator.set_state(chkpt_obj['generator'])
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.setstate(chkpt_obj['random'])
        np.random.set_state(chkpt_obj['numpy'])
        torch.cuda.set_rng_state(chkpt_obj['torch_cuda'], device='cuda')
  
  