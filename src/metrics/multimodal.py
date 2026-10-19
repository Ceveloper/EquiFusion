import numpy as np
import torch

def time_slice(array, t0, t, axis):
    if t == -1:
        return torch.index_select(array, axis, torch.arange(t0, array.shape[axis], device=array.device, dtype=torch.int32))
    else:
        return torch.index_select(array, axis, torch.arange(t0, t, device=array.device, dtype=torch.int32))

def cmd(val_per_frame, val_ref):
    # val_per_frame (array of floats) -> M_t, val_ref (float) -> M
    T = len(val_per_frame) + 1
    return np.sum([(T - t) * np.abs(val_per_frame[t-1] - val_ref) for t in range(1, T)])

def apd(pred, t0=0, t=-1, **kwargs):
    pred = time_slice(pred, t0, t, 2)
    # (batch_size, num_samples, seq_length, num_joints, features) to (num_samples, all_others)
    batch_size, n_samples = pred.shape[:2]
    if n_samples == 1: # only one sample => no APD possible
        return torch.tensor([0] * batch_size, device=pred.device)

    arr = pred.reshape(batch_size, n_samples, -1) # (batch_size, num_samples, others)
    dist = torch.cdist(arr, arr) # (batch_size, num_samples, num_samples)

    # Get the upper triangular indices (excluding the diagonal)
    iu = torch.triu_indices(n_samples, n_samples, offset=1)

    # Select upper triangular values for each batch
    upper_triangular_values = dist[:, iu[0], iu[1]]  # (batch_size, num_pairs)

    # Average the pairwise distances
    results = upper_triangular_values.mean(dim=-1)  # (batch_size,)
    
    assert list(results.shape) == [batch_size]
    return results

def ade(target, pred, t0=0, t=-1, reduction='mean', **kwargs):
    pred, target = time_slice(pred, t0, t, 2), time_slice(target, t0, t, 1)
    batch_size, n_samples, seq_length = pred.shape[:3]
    # from (batch_size, num_samples, seq_length, num_parts, num_joints, features) to (batch_size, num_samples, seq_length, num_parts * num_joints * features)
    pred = pred.reshape((batch_size, n_samples, seq_length, -1))
    # from (batch_size, seq_length, num_parts, num_joints, features) to (batch_size, seq_length, num_parts * num_joints * features)
    target = target.reshape((batch_size, 1, seq_length, -1))

    diff = pred - target
    dist = torch.linalg.norm(diff, axis=-1).mean(axis=-1)
    if reduction=='mean':
        return dist.min(axis=-1).values
    else: 
        return dist


def fde(target, pred, t0=0, t=-1, reduction='mean', **kwargs):
    pred, target = time_slice(pred, t0, t, 2), time_slice(target, t0, t, 1)
    batch_size, n_samples, seq_length = pred.shape[:3]
    # from (batch_size, num_samples, seq_length, num_parts, num_joints, features) to (batch_size, num_samples, seq_length, num_parts * num_joints * features)
    pred = pred.reshape((batch_size, n_samples, seq_length, -1))
    # from (batch_size, seq_length, num_parts, num_joints, features) to (batch_size, seq_length, num_parts * num_joints * features)
    target = target.reshape((batch_size, 1, seq_length, -1))
    
    diff = pred - target
    dist = torch.linalg.norm(diff, axis=-1)[..., -1]
    if reduction=='mean':
        return dist.min(axis=-1).values
    else: 
        return dist


def mae(target, pred, limbseq, limb_angles_idx, t0=0, t=-1, if_return_cossim=False, **kwargs):
    def compute_batched_limbseq(vector, lseq):
        vector = torch.cat([torch.zeros_like(vector[..., :1, :]), vector], dim=-2)  # add zero hip at index 0
        max_limbs = vector.shape[-2]
        limbseq_padded = torch.sort(lseq, dim=-1, descending=False)[0]
        # # limbseq_padded = torch.sort(limbseq_padded, dim=-1, descending=False)[0]
        assert torch.isnan(limbseq_padded).sum() == 0, f"Found NaN in cos_sim: {limbseq_padded}"

        *B, T, J, D = vector.shape
        _, L, _ = limbseq_padded.shape

        idx0 = limbseq_padded[..., 0]
        idx1 = limbseq_padded[..., 1]

        idx0_exp = idx0.unsqueeze(1).unsqueeze(1).unsqueeze(-1).expand(*B, T, L, D)
        idx1_exp = idx1.unsqueeze(1).unsqueeze(1).unsqueeze(-1).expand(*B, T, L, D)

        target_exp = vector.unsqueeze(-3).expand(*B, T, L, J, D).cpu()
        v0 = torch.gather(target_exp, -2, idx0_exp.unsqueeze(-2)).squeeze(-2)
        v1 = torch.gather(target_exp, -2, idx1_exp.unsqueeze(-2)).squeeze(-2)
        assert torch.isnan(v1 - v0).sum() == 0, f"Found NaN in cos_sim: {v1 - v0}"

        return v1 - v0  # [B, T, L, D]
    def cos_similarity(limb_vectors, limb_angles_idx_tup):
        cos_sim = torch.einsum('bstjd,bstjd->bstj', limb_vectors[..., limb_angles_idx_tup[:,0], :], limb_vectors[..., limb_angles_idx_tup[:,1], :])
        den = (limb_vectors[..., limb_angles_idx_tup[:,0], :]**2).sum(-1).sqrt() * (limb_vectors[..., limb_angles_idx_tup[:,1], :]**2).sum(-1).sqrt()
        den[den<1e-7] = 1e-7
        cos_sim = cos_sim/den
        return cos_sim
    def cos_similarity_batched(limb_vectors, limb_angles_idx_tup):
        # limb_vectors: [B, S, T, J, D]
        # limb_angles_idx_tup: [B, num_pairs, 2]
        # mask: [B, num_pairs]

        # gather limb vectors for each angle pair
        v1 = torch.gather(limb_vectors, -2, limb_angles_idx_tup[..., 0].cpu().unsqueeze(-1).unsqueeze(1).unsqueeze(1).expand(-1, limb_vectors.shape[1], limb_vectors.shape[2], -1, limb_vectors.size(-1)))
        v2 = torch.gather(limb_vectors, -2, limb_angles_idx_tup[..., 1].cpu().unsqueeze(-1).unsqueeze(1).unsqueeze(1).expand(-1, limb_vectors.shape[1], limb_vectors.shape[2], -1, limb_vectors.size(-1)))

        cos_sim = (v1 * v2).sum(-1)
        den = (v1.pow(2).sum(-1).sqrt() * v2.pow(2).sum(-1).sqrt()).clamp_min(1e-7)
        cos_sim = cos_sim / den
        assert torch.isnan(cos_sim).sum() == 0, f"Found NaN in cos_sim: {cos_sim}"
        # masking


        # apply mask (set invalid ones to 0)
        return cos_sim
    pred, target = time_slice(pred, t0, t, 2), time_slice(target, t0, t, 1)
    if (isinstance(limbseq, list) and not isinstance(limbseq[0][0], list)) or len(limbseq.shape) == 2:
        limbseq = torch.tensor(limbseq).to(pred.device)
        limb_angles_idx_tup = torch.tensor([[kin[i], kin[i+1]] for kin in limb_angles_idx for i in range(len(kin)-1)])
        limbseq = torch.sort(limbseq, dim=-1, descending=False)[0]
        limb_vectors_target = target[ ..., limbseq[:, 1], :] - target[..., limbseq[:, 0], :]
        limb_vectors_pred = pred[ ..., limbseq[:, 1], :] - pred[..., limbseq[:, 0], :]
        cossim_target = cos_similarity(limb_vectors_target.unsqueeze(1), limb_angles_idx_tup)
        cossim_pred = cos_similarity(limb_vectors_pred, limb_angles_idx_tup)
    else:

        # Build per-batch limb_angles_idx_tup
        limb_angles_idx_tup = []
        for kin in limb_angles_idx:
            pairs = []
            for chain in kin:
                pairs.extend([[chain[i], chain[i+1]] for i in range(len(chain)-1)])
            limb_angles_idx_tup.append(torch.tensor(pairs, dtype=torch.long))
        max_pairs = max(len(p) for p in limb_angles_idx_tup)
        limb_angles_idx_tup_padded = torch.zeros(len(limb_angles_idx_tup), max_pairs, 2, dtype=torch.long, device=pred.device)
        for b, p in enumerate(limb_angles_idx_tup):
            n = len(p)
            limb_angles_idx_tup_padded[b, :n] = p
            # limb_angle_mask[b, :n] = True
        limb_vectors_target  = compute_batched_limbseq(target.unsqueeze(1), limbseq)
        limb_vectors_pred = compute_batched_limbseq(pred, limbseq)
        cossim_target = cos_similarity_batched(limb_vectors_target, limb_angles_idx_tup_padded)
        cossim_pred = cos_similarity_batched(limb_vectors_pred, limb_angles_idx_tup_padded)


    if if_return_cossim:
        diff = cossim_pred - cossim_target
    else: 
        diff = (torch.acos(cossim_pred.clamp(-1, 1)) - torch.acos(cossim_target.clamp(-1, 1))) # in radiant. To get degrees *(180/pi)
        assert torch.isnan(diff).sum() == 0, f"Found NaN in cos_sim: {diff}"
    dist = torch.abs(diff).mean(-1).mean(axis=-1) # batch, samples 
    dist_deg =  dist*(180/np.pi)
    return dist_deg.min(axis=-1).values


def mmade(target, pred, mm_gt, t0=0, t=-1, **kwargs): # memory efficient version
    pred, target = time_slice(pred, t0, t, 2), time_slice(target, t0, t, 1)
    batch_size, n_samples, seq_length = pred.shape[:3]
    results = torch.zeros((batch_size, ))
    for i in range(batch_size):
        n_gts = mm_gt[i].shape[0]

        p = pred[i].reshape((n_samples, seq_length, -1)).unsqueeze(0)
        gt = time_slice(mm_gt[i], t0, t, 1).reshape((n_gts, seq_length, -1)).unsqueeze(1)

        dist = torch.linalg.norm(p - gt, axis=-1).mean(axis=-1)
        results[i] = dist.min(axis=-1).values.mean()

    return results

def mmfde(target, pred, mm_gt, t0=0, t=-1, **kwargs):
    pred, target = time_slice(pred, t0, t, 2), time_slice(target, t0, t, 1)
    batch_size, n_samples, seq_length = pred.shape[:3]
    results = torch.zeros((batch_size, ))
    for i in range(batch_size):
        n_gts = mm_gt[i].shape[0]

        p = pred[i].reshape((n_samples, seq_length, -1)).unsqueeze(0)
        gt = time_slice(mm_gt[i], t0, t, 1).reshape((n_gts, seq_length, -1)).unsqueeze(1)

        diff = p - gt
        dist = torch.linalg.norm(diff, axis=-1)[..., -1]
        results[i] = dist.min(axis=-1).values.mean()

    return results



