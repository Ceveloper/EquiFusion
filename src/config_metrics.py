import torch
import os
from functools import partial
import numpy as np

from .metrics.fid import get_classifier, MetricStorerFID
from .metrics.multimodal import apd, ade, mae, fde, mmade, mmfde, cmd
from .metrics.multimodal_unified import apd as apd_unified, ade as ade_unified, fde as fde_unified, mmade as mmade_unified, mmfde as mmfde_unified

from .metrics.apde import MetricStorerAPDE
from .metrics.body_realism import limb_length_variance, limb_length_jitter, limb_length_variation_difference_wrtGT, limb_length_error, limb_stretching_normed_rmse, limb_stretching_normed_mean, limb_jitter_normed_rmse, limb_jitter_normed_mean 
from .metrics.utils import draw_table
from .metrics.metric_storer import MetricStorer
from .metrics.cmd import CMDMetricStorer, resolve_cmd, motion_for_cmd
from .metrics.ignite_mpjpe import MeanPerJointPositionError
from .metrics.ignite_fde import FinalDisplacementError
from .metrics.utils import choose_best_sample


def get_body_realism_functs_per_sample(limbseq, skeleton, limb_stretching_normed_mean_f=None, limb_jitter_normed_mean_f=None, limb_stretching_normed_rmse_f=None, limb_jitter_normed_rmse_f=None):
    if limb_stretching_normed_mean_f is None:
        limb_stretching_normed_mean_f = limb_stretching_normed_mean
    if limb_jitter_normed_mean_f is None:
        limb_jitter_normed_mean_f = limb_jitter_normed_mean
    if limb_stretching_normed_rmse_f is None:
        limb_stretching_normed_rmse_f = limb_stretching_normed_rmse
    if limb_jitter_normed_rmse_f is None:
        limb_jitter_normed_rmse_f = limb_jitter_normed_rmse
    # Values are in cm in the paper
    limb_stretching_normed_mean_scaled = lambda *args, pred, target, **kwargs: limb_stretching_normed_mean_f(*args, pred=skeleton.add_zero_pad_center_hip(pred), target=skeleton.add_zero_pad_center_hip(target), reduction='none', **kwargs)*100
    limb_jitter_normed_mean_scaled = lambda *args, pred, target, **kwargs: limb_jitter_normed_mean_f(*args, pred=skeleton.add_zero_pad_center_hip(pred), target=skeleton.add_zero_pad_center_hip(target), reduction='none', **kwargs)*100
    limb_stretching_normed_rmse_scaled = lambda *args, pred, target, **kwargs: limb_stretching_normed_rmse_f(*args, pred=skeleton.add_zero_pad_center_hip(pred), target=skeleton.add_zero_pad_center_hip(target), reduction='none', **kwargs)*100
    limb_jitter_normed_rmse_scaled = lambda *args, pred, target, **kwargs: limb_jitter_normed_rmse_f(*args, pred=skeleton.add_zero_pad_center_hip(pred), target=skeleton.add_zero_pad_center_hip(target), reduction='none', **kwargs)*100
    if limbseq is not None:
        body_realism_metrics = {'StretchMean': partial(limb_stretching_normed_mean_scaled, limbseq=limbseq),
                        'JitterMean': partial(limb_jitter_normed_mean_scaled, limbseq=limbseq),
                        'StretchRMSE': partial(limb_stretching_normed_rmse_scaled, limbseq=limbseq),
                        'JitterRMSE': partial(limb_jitter_normed_rmse_scaled, limbseq=limbseq),
                        }
    else:
        body_realism_metrics = {'StretchMean': partial(limb_stretching_normed_mean_scaled),
                    'JitterMean': partial(limb_jitter_normed_mean_scaled),
                    'StretchRMSE': partial(limb_stretching_normed_rmse_scaled),
                    'JitterRMSE': partial(limb_jitter_normed_rmse_scaled),
                    }
    return body_realism_metrics

def get_body_realism_functs(limbseq, skeleton, limb_stretching_normed_mean_f=None, limb_jitter_normed_mean_f=None, limb_stretching_normed_rmse_f=None, limb_jitter_normed_rmse_f=None):
    if limb_stretching_normed_mean_f is None:
        limb_stretching_normed_mean_f = limb_stretching_normed_mean
    if limb_jitter_normed_mean_f is None:
        limb_jitter_normed_mean_f = limb_jitter_normed_mean
    if limb_stretching_normed_rmse_f is None:
        limb_stretching_normed_rmse_f = limb_stretching_normed_rmse
    if limb_jitter_normed_rmse_f is None:
        limb_jitter_normed_rmse_f = limb_jitter_normed_rmse
    # Values are in cm in the paper
    limb_stretching_normed_mean_scaled = lambda *args, pred, target, **kwargs: limb_stretching_normed_mean_f(*args, pred=skeleton.add_zero_pad_center_hip(pred), target=skeleton.add_zero_pad_center_hip(target), **kwargs)*100
    limb_jitter_normed_mean_scaled = lambda *args, pred, target, **kwargs: limb_jitter_normed_mean_f(*args, pred=skeleton.add_zero_pad_center_hip(pred), target=skeleton.add_zero_pad_center_hip(target), **kwargs)*100
    limb_stretching_normed_rmse_scaled = lambda *args, pred, target, **kwargs: limb_stretching_normed_rmse_f(*args, pred=skeleton.add_zero_pad_center_hip(pred), target=skeleton.add_zero_pad_center_hip(target), **kwargs)*100
    limb_jitter_normed_rmse_scaled = lambda *args, pred, target, **kwargs: limb_jitter_normed_rmse_f(*args, pred=skeleton.add_zero_pad_center_hip(pred), target=skeleton.add_zero_pad_center_hip(target), **kwargs)*100
    if limbseq is not None:
        body_realism_metrics = {'StretchMean': partial(limb_stretching_normed_mean_scaled, limbseq=limbseq),
                        'JitterMean': partial(limb_jitter_normed_mean_scaled, limbseq=limbseq),
                        'StretchRMSE': partial(limb_stretching_normed_rmse_scaled, limbseq=limbseq),
                        'JitterRMSE': partial(limb_jitter_normed_rmse_scaled, limbseq=limbseq),
                        }
    else:
        body_realism_metrics = {'StretchMean': partial(limb_stretching_normed_mean_scaled),
                    'JitterMean': partial(limb_jitter_normed_mean_scaled),
                    'StretchRMSE': partial(limb_stretching_normed_rmse_scaled),
                    'JitterRMSE': partial(limb_jitter_normed_rmse_scaled),
                    }
    return body_realism_metrics

def get_stats_funcs(stats_mode, skeleton, if_body_realism_per_sample=False, if_compute_mae=True, **kwargs):
    limbseq = torch.tensor(skeleton.limbseq)
    limbseq_mae = skeleton.get_limbseq().copy()
    limb_angles_idx = skeleton.limb_angles_idx.copy()

    body_realism_metrics = get_body_realism_functs(limbseq=limbseq, skeleton=skeleton)
    apd_f, ade_f, fde_f, mmade_f, mmfde_f, mae_f = apd, ade, fde, mmade, mmfde, mae

    mae_f = partial(mae_f, limbseq=limbseq_mae, limb_angles_idx=limb_angles_idx)
    assert not kwargs["if_consider_hip"]
    if "deterministic" in stats_mode.lower():
        stats_func = { 'ADE': ade_f, 'FDE': fde_f , 'MAE': mae_f,
                    'APD':  apd_f,
                    'uADE':lambda **kwargs: ade_unified(**kwargs)*100, 'uFDE':lambda **kwargs: fde_unified(**kwargs)*100, 'uAPD':lambda **kwargs: apd_unified(**kwargs),
                    **body_realism_metrics,
                    }
    elif "probabilistic_orig" == stats_mode.lower():
        stats_func = {  'APD': apd_f, 'ADE': ade_f, 'FDE': fde_f,
                        'MMADE': mmade_f, 'MMFDE': mmfde_f,
                    }
    elif "probabilistic" == stats_mode.lower():
        stats_func = {  'ADE': ade_f, 'FDE': fde_f , 'MAE': mae_f,
                        'uADE':lambda **kwargs: ade_unified(**kwargs)*100, 'uFDE':lambda **kwargs: fde_unified(**kwargs)*100, 'uAPD':lambda **kwargs: apd_unified(**kwargs),
                        'MMADE': mmade_f, 'MMFDE': mmfde_f,
                        'uMMADE':lambda **kwargs: mmade_unified(**kwargs)*100, 'uMMFDE':lambda **kwargs: mmfde_unified(**kwargs)*100,
                        'APD':  apd_f,
                        **body_realism_metrics,
                }
    else :
        raise NotImplementedError(f"stats_mode not implemented: {stats_mode}")
    
    if if_body_realism_per_sample:
        body_realism_metrics_per_sample = get_body_realism_functs_per_sample(limbseq=limbseq, skeleton=skeleton)
        for k, v in body_realism_metrics_per_sample.items():
            stats_func[k+'_perSample'] = v
    if not if_compute_mae:
        stats_func.pop('MAE')
    return stats_func

def get_apde_storer(**kwargs):
    return partial(MetricStorerAPDE, mmapd_gt_path= os.path.join(kwargs['annotations_folder'], "mmapd_GT.csv")), lambda pred, **kwargs: apd(pred) 
    
def get_fid_storer(**config):
    if config['dataset_name'] =='h36m':
        storer = partial(MetricStorerFID, classifier_path=config['precomputed_folder'], **config)    
        return storer, lambda dict: (dict['pred'], dict['target'])
    else:
        raise NotImplementedError(f"FID not implemented for dataset: {config['dataset_name']}")

def get_cmd_storer(dataset, if_consider_hip=False, **kwargs):
    assert not if_consider_hip
    cmdstorer = {"CMD": ( partial(CMDMetricStorer, final_funct=partial(resolve_cmd, idx_to_class=dataset.idx_to_class,  mean_motion_per_class=dataset.mean_motion_per_class)),
                              lambda pred, extra, **kwgs: (motion_for_cmd(pred.clone()), 
                                np.array([dataset.class_to_idx[c] for c in extra["metadata"][dataset.metadata_class_idx]]))
                            ),
                }
    return cmdstorer

def attach_engine_to_metrics(engine, dataset_split, stats_mode, dataset, skeleton, if_compute_cmd=False, if_compute_fid=False, if_compute_apde=False, **config):
    stats_func = get_stats_funcs(stats_mode, skeleton=skeleton, **config)
    
    stats_metrics = {k: MetricStorer(output_transform=lambda dict, funct=funct: funct(**(dict.copy())),
                                                    return_op='max' if '_max' in k else 'avg') for k, funct in stats_func.items()}
    
    for name, metric in stats_metrics.items():
        metric.attach(engine, name)

    if dataset_split == 'test' and if_compute_fid:
        fid_storer, funct = get_fid_storer(**config)
        apde_metrics = {'FID': fid_storer(output_transform=funct)}
        for name, metric in apde_metrics.items():
            metric.attach(engine, name)

    if dataset_split=='test' and if_compute_cmd:
        cmd_storer = get_cmd_storer(dataset, **config)
        cmd_metrics = {cmd_name: cmdclass(output_transform=lambda x_dict: funct(**x_dict.copy())) 
                            for cmd_name, (cmdclass, funct) in cmd_storer.items()}
        for name, metric in cmd_metrics.items():
            metric.attach(engine, name)
    if if_compute_apde:
        storer, funct = get_apde_storer(**config)
        apde_metrics = {'APDE': storer(output_transform=lambda x_dict: funct(**x_dict.copy()))}
        for name, metric in apde_metrics.items():
            metric.attach(engine, name)