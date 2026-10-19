from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Any
import os
import torch
import json
import hashlib
from scipy.spatial.transform import Rotation as R
from torch.utils.data._utils.collate import default_collate

from .motion_dataset import MotionDataset
from ...skeleton import create_skeleton, SkeletonVanilla as Skeleton

def pad_jointnodes(obs, pred, extra, num_nodes):
    missing_nodes = num_nodes - obs.shape[-2]
    obs = torch.cat([obs, torch.zeros(*obs.shape[:-2], missing_nodes, obs.shape[-1])], dim=-2)
    pred = torch.cat([pred, torch.zeros(*pred.shape[:-2], missing_nodes, pred.shape[-1])], dim=-2)
    adj = torch.zeros(*extra['adj'].shape[:-2], num_nodes, num_nodes)
    missing_nodes = num_nodes - extra['adj'].shape[-2]
    if missing_nodes > 0:
        adj[..., :-missing_nodes, :-missing_nodes] = extra['adj']
    else:
        adj = extra['adj']
    extra['adj'] = adj
    #############################
    
    return obs, pred, extra

def pad_perm_matrix(extra, num_nodes):
    if 'perm' in extra:
        perm = extra['perm']
        # TO DO here
        perm = torch.cat([perm, torch.zeros(num_nodes - perm.shape[0], perm.shape[-1])], dim=0)
        perm = torch.cat([perm, torch.zeros(num_nodes, num_nodes - perm.shape[-1])], dim=1)
        assert perm.shape == (num_nodes, num_nodes), f"perm.shape={perm.shape} != (num_nodes, num_nodes)={num_nodes}"
        extra['perm'] = perm
    return extra

class DynamicKinematicMotionDataset(Dataset):
    def __init__(self, datasets_dict: Dict[str, Dict[str, Any]], split: str, skeletons: Dict[str, Skeleton], dataset_sampling_style='uniform', **kwargs):     
        
        assert len(set([type(x).__name__.split('Skeleton')[-1] for  x in skeletons.values()])) == 1, "DynamicKinematicMotionDataset supports only one type of motion presentation"
        self.num_nodes = kwargs['max_num_nodes']
        dataset_names = sorted(list(datasets_dict.keys()))
        self.skeleton = skeletons[dataset_names[0]]
        assert split in ["valid", 'train', 'test','train_eval'], f"{split} not implemented"
        self.in_eval = split in ['valid', 'test']
        datasets = [ datasets_dict[name] for name in dataset_names]

        self.datasets = datasets
        self.dataset_names = dataset_names

        dataset_sizes = [len(d) for d in self.datasets]
        total_size = sum(dataset_sizes)
        probabilities = [size / total_size for size in dataset_sizes]
        self.sampling_prob = probabilities
        assert dataset_sampling_style in ['uniform', 'proportional'], f"{dataset_sampling_style} not implemented"
        self.sampling_style = dataset_sampling_style
        super().__init__()

    def __len__(self):
        return max(len(d) for d in self.datasets)
    
    @property
    def dataset(self):
        assert len(self.datasets) == 1, "This is a multiskel dataset for evaluation, only one dataset supported"
        return self.datasets[0]
    
    def eval(self):
        for d in self.datasets:
            d.eval()
        self._in_eval = self.in_eval 
        self.in_eval = True
        

    def train(self):
        for d in self.datasets:
            d.train()
        self.in_eval = self._in_eval

    def __getitem__(self, sample_idx):
        # choose one of the datasets. Either in percentage of the total number of samples per datasets or with equal chance among the number of datasets.
        if len(self.datasets) == 1 or self.in_eval:
            dataset_idx = 0
        else:
            if self.sampling_style == 'uniform':
                dataset_idx = np.random.choice(len(self.datasets))
            elif self.sampling_style == 'proportional':
                dataset_idx = np.random.choice(len(self.datasets), p=self.sampling_prob)
            else: 
                raise NotImplementedError
        dataset = self.datasets[dataset_idx]
        sample_idx = int(round(sample_idx/len(self) * len(dataset))) if len(self.datasets) > 1 else sample_idx
        obs, pred, extra = dataset[sample_idx]
        extra["dataset_idx"] = self.dataset_names[dataset_idx] #dataset_idx
        if obs.shape[-2] < self.num_nodes:
            obs, pred, extra = pad_jointnodes(obs, pred, extra, self.num_nodes)
            if not self.in_eval:
                extra = pad_perm_matrix(extra, pred.shape[-2])
        if dataset.if_forward_perm_to_graph and "perm" not in extra and not self.in_eval: # TO DO do we have to add if in_eval?
            extra["perm"] = torch.eye(pred.shape[-2])
        return obs, pred, extra
    


def get_DynamicKinematicMotionDatasetEval(DatasetClass):
    def __init__(self,  split: str, if_eval_from60_at50fps=False, if_eval_from60_at30fps=False, if_eval_from60_at20fps=False, **kwargs):
        self.num_nodes = kwargs['max_num_nodes']
        self.if_eval_from60_at50fps = if_eval_from60_at50fps
        self.if_eval_from60_at30fps = if_eval_from60_at30fps
        self.if_eval_from60_at20fps = if_eval_from60_at20fps


        if self.if_eval_from60_at50fps:
            self.mask_out = torch.from_numpy(np.ceil(np.arange(0, kwargs['pred_length'], 60/50))).to(int)
            kwargs['obs_length'] = 25
            kwargs['pred_length'] = 100
            self.mask = torch.from_numpy(np.floor(np.arange(0, kwargs['obs_length'], 25/30))).to(int)
        if self.if_eval_from60_at30fps:
            self.mask_out = torch.from_numpy(np.ceil(np.arange(0, kwargs['pred_length'], 60/30))).to(int)
            kwargs['obs_length'] = 15
            kwargs['pred_length'] = 60
            self.mask = torch.from_numpy(np.floor(np.arange(0, kwargs['obs_length'], 15/30))).to(int)
        if self.if_eval_from60_at20fps:
            self.mask_out = torch.from_numpy(np.ceil(np.arange(0, kwargs['pred_length'], 60/20))).to(int)
            kwargs['obs_length'] = 10
            kwargs['pred_length'] = 40
            self.mask = torch.from_numpy(np.floor(np.arange(0, kwargs['obs_length'], 10/30))).to(int)
        assert split in ["valid", 'test','train_eval'], f"{split} not implemented"
        if split == 'train_eval':
            split = 'train'
        super(self.__class__, self).__init__(split=split, **kwargs, **kwargs['data_loader_'+split])

        


    def __getitem__(self, sample_idx):
        obs, pred, extra = super(self.__class__, self).__getitem__(sample_idx)
        if obs.shape[-2] < self.num_nodes:
            obs, pred, extra = pad_jointnodes(obs, pred, extra, self.num_nodes)
        if self.if_eval_from60_at50fps or self.if_eval_from60_at30fps  or self.if_eval_from60_at20fps:
            # pad input to 60 frames
            obs = obs[..., self.mask, :, :]
        return obs, pred, extra
    return type('DynamicKinematicMotionDatasetEval', (DatasetClass,), {'__init__':__init__, '__getitem__':__getitem__})


    
