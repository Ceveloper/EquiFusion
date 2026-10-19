import torch

def greates_minimum_distance(cdist: torch.Tensor, other_indices: list) -> int:
    min_dist = cdist[..., other_indices, :].min(dim=-1)[0] # (N - M)
    greatest_min_distance = min_dist.max()

    indices = (cdist.min(dim=1)[0] == greatest_min_distance).nonzero().tolist()

    for index in indices:
        if index[0] in other_indices:
            return index[0]
    raise ValueError("No index found")


def get_highest_diversity(cdist: torch.Tensor, num_chosen_samples: int, decision_fn):
    indices = [0] # index of GT
    other_indices = list(range(1, cdist.shape[0]))

    for i in range(0, num_chosen_samples):
        chosen_idx = decision_fn(cdist[..., indices], other_indices)
        indices.append(chosen_idx)
        other_indices.remove(chosen_idx)

    return [ i-1 for i in indices[1:]] # remove GT from indices


def find_samples_maxapd_wrtGT(pred, target, num_chosen_samples=5):    
    batch_size, n_samples = pred.shape[:2]
    arr = torch.cat([target.unsqueeze(1), pred], dim=1)
    arr = arr.reshape(batch_size, n_samples+1, -1)
    dist = torch.cdist(arr, arr) # (batch_size, num_samples, num_samples)
    chosen_set = [get_highest_diversity(dist_, num_chosen_samples=num_chosen_samples, decision_fn=greates_minimum_distance) for dist_ in dist]# batch_size x 5 
    chosen_set_apd = None #apd(torch.stack([pred[i, idxs] for i, idxs in enumerate(chosen_set)], dim=0))
    # val, idxs = chosen_set_apd.max(-1)
    return chosen_set_apd, chosen_set

def get_closest_and_nfurthest_maxapd(y_pred, y_gt, nsamples):
    assert len(y_pred.shape) == 4 and y_pred.shape[-1] == 3, f"Expected SxTxJx3 tensors, got {len(y_pred.shape)}"
    S, T, J, _ = y_pred.shape
    assert len(y_gt.shape) == 3 and  y_gt.shape[-2:] == y_pred.shape[-2:], f"Expected TxJx3 tensors, got and {len(y_gt.shape)}"
    arr = torch.cat([y_pred, y_gt.unsqueeze(0)], dim=0)
    arr = arr.reshape(arr.shape[0], -1)# (num_samples_orig+1, others)
    dist = torch.cdist(arr, arr) # (num_samples_orig+1, num_samples_orig)
    sorted, indices = torch.sort(dist[-1, :-1], dim=0, descending=True) # in descending order
    # sorted, indices = sorted[:-1], indices[:-1]
    assert len(indices) == y_pred.shape[0]

    _, closest_idx = sorted[-1], indices[-1]
    pred_closest = y_pred[closest_idx]
    pred_closest, y_pred = pred_closest.unsqueeze(0), y_pred.unsqueeze(0)

    _, sorted_preds_idxs = find_samples_maxapd_wrtGT(y_pred, pred_closest, num_chosen_samples=nsamples)
    sorted_preds_idxs = sorted_preds_idxs[0] # remove batch dim
    y_pred = y_pred.squeeze(0)
    pred_closest = pred_closest.squeeze(0)
    sorted_preds = torch.stack([y_pred[i] for i in sorted_preds_idxs], dim=0)
    sorted_preds_idxs = [closest_idx.item()] + sorted_preds_idxs
    return pred_closest, sorted_preds, sorted_preds_idxs