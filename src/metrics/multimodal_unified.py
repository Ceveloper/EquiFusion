import numpy as np
import torch

def time_slice(array, t0, t, axis):
    if t == -1:
        return torch.index_select(array, axis, torch.arange(t0, array.shape[axis], device=array.device, dtype=torch.int32))
    else:
        return torch.index_select(array, axis, torch.arange(t0, t, device=array.device, dtype=torch.int32))


def apd(pred, t0=0, t=-1, **kwargs):
    pred = time_slice(pred, t0, t, 2)
    # (batch_size, num_samples, seq_length, num_joints, features) to (num_samples, all_others)
    batch_size, n_samples = pred.shape[:2]
    if n_samples == 1: # only one sample => no APD possible
        return torch.tensor([0] * batch_size, device=pred.device)
    n_joints = pred.shape[-2]
    arr = pred.permute(0,3, 1, 2, 4).reshape(batch_size, n_joints, n_samples, -1) # (batch_size, num_samples, others)
    dist = torch.cdist(arr, arr) # (batch_size, n_joints, num_samples, num_samples)

    # Get the upper triangular indices (excluding the diagonal)
    iu = torch.triu_indices(n_samples, n_samples, offset=1)

    # Select upper triangular values for each batch
    upper_triangular_values = dist[..., iu[0], iu[1]]  # (batch_size, n_joints, num_pairs)

    # Average the pairwise distances
    results = upper_triangular_values.mean(dim=-2).mean(dim=-1)  # (batch_size, n_joints, num_samples) --> (batch_size,)
    
    assert list(results.shape) == [batch_size]
    return results

    
def ade(target, pred, t0=0, t=-1, reduction='mean', **kwargs):
    pred, target = time_slice(pred, t0, t, 2), time_slice(target, t0, t, 1)
    batch_size, n_samples, seq_length , num_joints = pred.shape[:4]
    # from (batch_size, num_samples, seq_length, num_parts, num_joints, features) to (batch_size, num_samples, seq_length, num_parts * num_joints * features)
    pred = pred.reshape((batch_size, n_samples, seq_length, num_joints, -1))
    # from (batch_size, seq_length, num_parts, num_joints, features) to (batch_size, seq_length, num_parts * num_joints * features)
    target = target.reshape((batch_size, 1, seq_length, num_joints, -1))

    diff = pred - target
    dist = torch.linalg.norm(diff, axis=-1).mean(axis=-1).mean(axis=-1)
    if reduction=='mean':
        return dist.min(axis=-1).values
    else: 
        return dist


def fde(target, pred, t0=0, t=-1, reduction='mean', **kwargs):
    pred, target = time_slice(pred, t0, t, 2), time_slice(target, t0, t, 1)
    batch_size, n_samples, seq_length , num_joints = pred.shape[:4]
    # from (batch_size, num_samples, seq_length, num_parts, num_joints, features) to (batch_size, num_samples, seq_length, num_parts * num_joints * features)
    pred = pred.reshape((batch_size, n_samples, seq_length, num_joints, -1))
    # from (batch_size, seq_length, num_parts, num_joints, features) to (batch_size, seq_length, num_parts * num_joints * features)
    target = target.reshape((batch_size, 1, seq_length, num_joints, -1))
    
    diff = pred - target
    dist = torch.linalg.norm(diff, axis=-1).mean(axis=-1)[..., -1]
    if reduction=='mean':
        return dist.min(axis=-1).values
    else: 
        return dist


def mmade(target, pred, mm_gt, t0=0, t=-1, **kwargs): # memory efficient version
    pred, target = time_slice(pred, t0, t, 2), time_slice(target, t0, t, 1)
    batch_size, n_samples, seq_length , num_joints = pred.shape[:4]
    results = torch.zeros((batch_size, ))
    for i in range(batch_size):
        n_gts = mm_gt[i].shape[0]

        p = pred[i].reshape((n_samples, seq_length, num_joints, -1)).unsqueeze(0)
        gt = time_slice(mm_gt[i], t0, t, 1).reshape((n_gts, seq_length, num_joints, -1)).unsqueeze(1)

        dist = torch.linalg.norm(p - gt, axis=-1).mean(axis=-1).mean(axis=-1)
        results[i] = dist.min(axis=-1).values.mean()

    return results

def mmfde(target, pred, mm_gt, t0=0, t=-1, **kwargs):
    pred, target = time_slice(pred, t0, t, 2), time_slice(target, t0, t, 1)
    batch_size, n_samples, seq_length , num_joints = pred.shape[:4]
    results = torch.zeros((batch_size, ))
    for i in range(batch_size):
        n_gts = mm_gt[i].shape[0]

        p = pred[i].reshape((n_samples, seq_length, num_joints, -1)).unsqueeze(0)
        gt = time_slice(mm_gt[i], t0, t, 1).reshape((n_gts, seq_length, num_joints, -1)).unsqueeze(1)

        diff = p - gt
        dist = torch.linalg.norm(diff, axis=-1).mean(axis=-1)[..., -1]
        results[i] = dist.min(axis=-1).values.mean()

    return results
