from typing import Tuple, List, Dict
import numpy as np

import torch
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

from .multimodal import cmd
    
def motion_for_cmd(pred):
    motion = (torch.linalg.norm(pred[..., 1:, :, :] - pred[..., :-1, :, :], axis=-1)).mean(axis=1).mean(axis=-1)
    return motion


def resolve_cmd(histogram_data, all_obs_classes, idx_to_class: List[str], mean_motion_per_class: List[float]):
    results = 0.
    step_obs_classes = np.concatenate(all_obs_classes, axis=0)
    motion_data = np.concatenate(histogram_data, axis=0) #shape (num_batches*batch_size, num_timesteps-1) = (num_segments, num_timesteps-1)

    motion_per_class = np.zeros((len(idx_to_class), motion_data.shape[1]))
    # CMD weighted by class
    for i, (name, class_val_ref) in enumerate(zip(idx_to_class, mean_motion_per_class)):
        mask = step_obs_classes == i
        if mask.sum() == 0:
            continue
        motion_data_mean = motion_data[mask].mean(axis=0) # mean over samples
        motion_per_class[i] = motion_data_mean
        results += cmd(motion_data_mean, class_val_ref) * (mask.sum() / step_obs_classes.shape[0]) # this impelmented weighted average of cmd over dataset classes
    return results#, motion_datas


class CMDMetricStorer(Metric):
    def __init__(self, final_funct, output_transform=lambda x: x):

        self.final_funct = final_funct
        super().__init__(output_transform=output_transform)

    def reset(self):
        self.vals = []
        self.idxs = []
        super().reset()

    def update(self, output):
        # pred, class_idxs, step = self._output_transform(output) # This will be called by engine if we use it
        mot_cmd, class_idxs = output
        self.vals.append(mot_cmd.cpu().numpy())
        self.idxs.append(class_idxs)

    def compute(self):
        if self.vals is None or self.idxs is None:
            raise NotComputableError('MetricStorer must have at least one example before it can be computed.')
        ret = self.final_funct(self.vals, self.idxs)
        return ret
