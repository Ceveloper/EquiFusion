import numpy as np
import torch

def _extract_limb_length(kpts, limbseq):
    limbdist = []
    assert  np.array(limbseq).max()< kpts.shape[-2], f"Expected { np.array(limbseq).max()} < {kpts.shape[-2]}"
    if np.array(limbseq).max() == kpts.shape[-2]:
        shape = list(kpts.shape)
        shape[-2] = 1
        kpts = torch.cat([torch.zeros(shape, device=kpts.device), kpts], dim=-2)
    if len(limbseq.shape) > 2:
        for b, (l1,l2) in enumerate(zip(limbseq[..., 0], limbseq[..., 1])):
            limbdist.append( torch.linalg.norm(kpts[b, ..., l1, :] - kpts[b, ..., l2, :], dim=-1))
        return torch.stack(limbdist, dim=0)
    for l1,l2 in limbseq:
        limbdist.append( torch.linalg.norm(kpts[..., l1, :] - kpts[..., l2, :], dim=-1))
    return torch.stack(limbdist, dim=-1)

def limb_length_variation_difference_wrtGT(target, pred, limbseq, mode='mean', **kwargs):
    pred_llerr = limb_length_jitter(pred=pred, limbseq=limbseq, mode=mode, **kwargs) # [batch_size]
    target_llerr = limb_length_jitter(pred=target.unsqueeze(1), limbseq=limbseq, mode=mode, **kwargs) # [batch_size]
    
    # estimate is that target_llerr < pred_llerr, so it makes sense to output pred_llerr - target_llerr
    # if negative, it means we have higher precision than gt!
    if mode == 'mean':  
        pred_llerr, target_llerr = pred_llerr.mean(axis=-1), target_llerr.mean(axis=-1)
    elif mode == 'max':  
        pred_llerr, target_llerr = pred_llerr.max(axis=-1).values , target_llerr.max(axis=-1).values 
    elif mode == 'min':  
        pred_llerr, target_llerr = pred_llerr.min(axis=-1).values , target_llerr.min(axis=-1).values 
    elif mode == 'none':
        return (pred_llerr - target_llerr)
    return (pred_llerr - target_llerr).unsqueeze(0)
    

def limb_length_error(target, pred, limbseq, mode='mean', **kwargs):
    #(batch_size, num_samples, seq_length, num_joints, features)
    batch_size, num_samples, seq_length, num_joints, features = pred.shape

    target_ll, pred_ll = _extract_limb_length(target, limbseq), _extract_limb_length(pred, limbseq)
    assert [batch_size, num_samples, seq_length] == list(pred_ll.shape[:-1])
    llerr = torch.abs(target_ll.unsqueeze(1) - pred_ll).mean(dim=-1) # mean ll error over joints
    assert [batch_size, num_samples, seq_length] == list(llerr.shape)
    llerr = llerr.mean(axis=-1) # mean over time dim
    assert [batch_size, num_samples] == list(llerr.shape)
    assert mode in ['mean', 'max', 'min'], f"Mode {mode} not supported"
    if mode == 'mean':  
        return llerr.mean(axis=-1) # mean over extracted samples
    elif mode == 'max':  
        return llerr.max(axis=-1).values 
    elif mode == 'min':  
        return llerr.min(axis=-1).values 

def limb_length_variance(pred, limbseq, mode='mean', if_per_sample=False, **kwargs):
    #(batch_size, num_samples, seq_length, num_joints, features)
    batch_size, num_samples, seq_length, num_joints, features = pred.shape
    num_limbs = limbseq.shape[-2] if not isinstance(limbseq, list) else len(limbseq)

    pred_ll =  _extract_limb_length(pred, limbseq)
    assert [batch_size, num_samples, seq_length] == list(pred_ll.shape[:-1])
    assert mode in ['mean', 'max', 'min', 'none'], f"Mode {mode} not supported"
    llvar = pred_ll.var(dim=-2) # batch_size, num_samples, num_joints
    if mode == 'mean':  
        llvar = llvar.mean(axis=-1)
        if if_per_sample:
            assert len(llvar.shape) == 2, f"Expected 2, got {len(llvar.shape)}"
            return llvar
        return llvar.mean(axis=-1) # mean over extracted samples
    elif mode == 'max': 
        llvar = llvar.max(axis=-1).values 
        if if_per_sample:
            assert len(llvar.shape) == 2, f"Expected 2, got {len(llvar.shape)}"
            return llvar
        return llvar.max(axis=-1).values  
    elif mode == 'min':  
        llvar = llvar.min(axis=-1).values
        if if_per_sample:
            assert len(llvar.shape) == 2, f"Expected 2, got {len(llvar.shape)}"
            return llvar
        return llvar.min(axis=-1).values 
    elif mode == 'none':
        return llvar

def limb_length_jitter(pred, limbseq, mode='mean', if_per_sample=False, **kwargs):
    #(batch_size, num_samples, seq_length, num_joints, features)
    *B, seq_length, num_joints, features = pred.shape
    num_limbs = limbseq.shape[-2] if not isinstance(limbseq, list) else len(limbseq)
    pred_ll =  _extract_limb_length(pred, limbseq)
    assert [*B, seq_length] == list(pred_ll.shape[:-1])
    llerr = torch.abs(pred_ll[..., 1:, :] - pred_ll[..., :-1, :])
    assert [*B, seq_length-1, num_limbs ] == list(llerr.shape)

    assert mode in ['mean', 'max', 'min', 'none'], f"Mode {mode} not supported"
    if mode == 'mean':  
        llerr = llerr.mean(axis=-1).mean(dim=-1)
        if if_per_sample:
            assert len(llerr.shape) == 2, f"Expected 2, got {len(llerr.shape)}"
            return llerr
        return llerr.mean(axis=-1) # mean over extracted samples
    elif mode == 'max': 
        llerr = llerr.max(axis=-1).values.max(axis=-1).values 
        if if_per_sample:
            assert len(llerr.shape) == 2, f"Expected 2, got {len(llerr.shape)}"
            return llerr
        return llerr.max(axis=-1).values  
    elif mode == 'min':  
        llerr = llerr.min(axis=-1).values.min(axis=-1).values
        if if_per_sample:
            assert len(llerr.shape) == 2, f"Expected 2, got {len(llerr.shape)}"
            return llerr
        return llerr.min(axis=-1).values 
    elif mode == 'none':
        return llerr

def limb_stretching_normed_rmse(pred, target, limbseq, mode='std', reduction='mean',**kwargs):
    assert mode in ['std', 'var'], f"Mode {mode} not supported"
    B, S, T, N, _ = pred.shape
    limb_length = _extract_limb_length(pred, limbseq)
    B, S, T, J = limb_length.shape
    limb_length_gt = _extract_limb_length(target, limbseq)
    assert limb_length_gt.shape == (B, T, J)
    mean =limb_length_gt.mean(-2).unsqueeze(-2)
    mean = mean.unsqueeze(1) # B, 1, 1, J
    
    var = (limb_length - mean)**2 # B, S, T, J	
    var = var.mean(-2)
    if mode == 'std':
        var = torch.sqrt(var)
    var = var/ mean.squeeze(-2)
    if reduction == 'mean':
        return  var.view(B, -1).mean(-1) # B, S, J --> B
    elif reduction == 'persample':
        return var.mean(-1)
    else: 
        return var

def limb_stretching_normed_mean(pred, target, limbseq, reduction='mean', obs_as_target=False, **kwargs):
    B, S, T, N, _ = pred.shape
    limb_length = _extract_limb_length(pred, limbseq)
    B, S, T, J = limb_length.shape
    limb_length_gt = _extract_limb_length(target, limbseq)
    assert limb_length_gt.shape == (B, T, J) or obs_as_target
    assert len(limb_length_gt.shape) == 3
    mean_gt =limb_length_gt.mean(-2).unsqueeze(-2)  # B, 1, J
    mean = limb_length.mean(-2) # B, S, J

    normed_mean = (mean - mean_gt).abs()/ mean_gt
    if reduction == 'mean':
        return normed_mean.view(B, -1).mean(-1) # B, S, J --> B
    elif reduction == 'persample':
        return normed_mean.mean(-1)
    else: 
        return normed_mean


def limb_jitter_normed_rmse(pred, target, limbseq, mode='std', reduction='mean',**kwargs):
    """
    We assume the ground truth jitter to be zero.
    We compute the variance accordingly, without subtracting any gt mean.

    Normalization is done by dividing the variance by the mean of the ground truth limb length.
    In this way the output is the variance of the jitter in percentage of the GT limb length.
    """
    assert mode in ['std', 'var'], f"Mode {mode} not supported"
    B, S, T, N, _ = pred.shape
    jitter = limb_length_jitter(pred, limbseq, mode='none', if_per_sample=True)
    B, S, Tmin1, J = jitter.shape
    limb_length_gt = _extract_limb_length(target, limbseq)
    assert limb_length_gt.shape == (B, T, J)
    mean =limb_length_gt.mean(-2).unsqueeze(-2) # B, 1, J
    
    var = (jitter)**2 # B, S, T, J	
    var = var.mean(-2)
    if mode == 'std':
        var = torch.sqrt(var)
    var = var/ mean
    if reduction == 'mean':
        return var.view(B, -1).mean(-1) # B, S, J --> B
    elif reduction == 'persample':
        return var.mean(-1)
    else: 
        return var

def limb_jitter_normed_mean(pred, target, limbseq, reduction='mean', **kwargs):
    """
    We assume the ground truth jitter to be zero. 
    So we compute the normed mean jitter of the prediction without subtracting any gt mean.

    Normalization is done by dividing the mean jitter of the predictions by the mean of the ground truth limb length.
    In this way the output is the mean jitter in percentage of the GT limb length.
    """
    B, S, T, N, _ = pred.shape
    jitter = limb_length_jitter(pred, limbseq, mode='none', if_per_sample=True)
    B, S, Tmin1, J = jitter.shape
    limb_length_gt = _extract_limb_length(target, limbseq)
    assert limb_length_gt.shape == (B, T, J)
    mean_gt =limb_length_gt.mean(-2).unsqueeze(-2)  # B, 1, J

    normed_mean = jitter.mean(-2)/ mean_gt # B, S, J
    if reduction == 'mean':
        return normed_mean.view(B, -1).mean(-1) # B, S, J --> B
    elif reduction == 'persample':
        return normed_mean.mean(-1)
    else: 
        return normed_mean