import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Set
from tqdm import tqdm


def compute_mean_motions(dataset, batch_size=128, if_traj_only=False): 
    if isinstance(dataset, Dataset):
        data_loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=1, pin_memory= True, drop_last=False,)   
    else: 
        assert isinstance(dataset, DataLoader)
        data_loader = dataset

    counter = 0
    average_3d = 0
    CLASS_TO_IDX = data_loader.dataset.class_to_idx
    class_average_3d = {k: 0 for k in CLASS_TO_IDX}
    class_counter = {k: 0 for k in CLASS_TO_IDX}
    for batch_idx, batch in enumerate(data_loader):
        data, target, extra = batch
        classes = np.array([CLASS_TO_IDX[c] for c in extra["metadata"][data_loader.dataset.metadata_class_idx]])
        target = data_loader.dataset.skeleton.transform_to_metric_space(target)
        if if_traj_only:
            target = target[..., 0:1, :]
        target_3d = target
        
        average_3d += (np.linalg.norm(target_3d[..., 1:, :, :] - target_3d[..., :-1, :, :], axis=-1)).mean()
        counter += 1
        for class_label in CLASS_TO_IDX:
            c = CLASS_TO_IDX[class_label]
            class_mask = classes == c
            target_class_3d = target_3d[class_mask] 
            class_average_3d[class_label] += (np.linalg.norm(target_class_3d[:, 1:] - target_class_3d[:, :-1], axis=-1)).mean(axis=-1).mean(axis=-1).sum()
            class_counter[class_label] += target_class_3d.shape[0]

    print(f"AVERAGE={average_3d/counter:.8f}")

    total_class_counter = sum(class_counter.values())
    list_of_motions_3d = []
    frequencies = []
    for c in class_average_3d:
        print(f"{c}: {class_average_3d[c]/class_counter[c]:.8f}")
        list_of_motions_3d.append(float((class_average_3d[c]/class_counter[c])))
        frequencies.append(class_counter[c]/total_class_counter)
    return class_average_3d, list_of_motions_3d, frequencies

############################################################################################################
# For multimodal GT

def get_multimodal_gt(data_loader, multimodal_threshold, recover_landmarks=True):
    def get_mmgt_idxs(seq1: torch.Tensor, seq2: torch.Tensor, multimodal_threshold:float):
        seq1 = seq1.reshape(seq1.shape[0], -1)
        seq2 = seq2.reshape(seq2.shape[0], -1)
        pd = torch.cdist(seq1, seq2, p=2, compute_mode='donot_use_mm_for_euclid_dist') #  obtain same exact results to scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(obs)) as in previous works (BeLFusion)
        ind = np.nonzero(pd < multimodal_threshold) # (n_similar, 2). The last dimension indexes the pd matrix
        return ind
    
    def store_mmgt_idxs(i: int, j: int, batch_size: int, idxs: torch.Tensor, gt_idces_dict: Dict[int, Set[int]]):
        # shape of tensor idxs is [Any, 2]
        idxs_mapped = idxs.clone()
        idxs_mapped[:, 0:1] += i * batch_size
        idxs_mapped[:, 1:] += j * batch_size
        for k in range(idxs_mapped.shape[0]):
            segment1 = idxs_mapped[k][0].item()
            segment2 = idxs_mapped[k][1].item()
            if segment1 not in gt_idces_dict:
                gt_idces_dict[segment1] = set()
            if segment2 not in gt_idces_dict:
                gt_idces_dict[segment2] = set()
            gt_idces_dict[segment1].add(segment2)
            gt_idces_dict[segment2].add(segment1)
        return gt_idces_dict
    
    batch_size = data_loader.batch_size
    gt_idces_dict2 = {}
    for b_i, batch_i in enumerate(tqdm(data_loader, total = len(data_loader),  desc ="Number of batches processed")):
        data, target, extra = batch_i
        if recover_landmarks:
            data = data_loader.dataset.skeleton.transform_to_metric_space(data) 
        idx = extra["segment_idx"]
        assert (idx == torch.arange(start=b_i*batch_size, end=min((b_i+1)*batch_size, len(data_loader.dataset)))).all(), "The segment idxs are not as expected! Dataloaders does noto deliver segments in order."
        # seq_length, n_joints, n_features = target.shape[1:]
        # dataset is too big we need to compute the pairwise distance in chunks
        for b_j, batch_j in enumerate(tqdm(data_loader, total = len(data_loader) - b_i, leave=False,  desc ="subprocess")):
            if b_i > b_j :
                # we have already computed th mmgt between batch_i and batch_j
                continue
            data2, target2, extra2 = batch_j
            if recover_landmarks:
                data2 = data_loader.dataset.skeleton.transform_to_metric_space(data2) 
            mmgt_idxs = get_mmgt_idxs(data[:, -1], data2[:, -1], multimodal_threshold)        
            gt_idces_dict2 = store_mmgt_idxs(i=b_i, j=b_j, batch_size=batch_size, idxs=mmgt_idxs, gt_idces_dict=gt_idces_dict2)
    
    assert all([ k in gt_idces_dict2[seg] for k, v in gt_idces_dict2.items() for seg in v ]), "The multimodal GT is not symmetric."
    gt_idces_dict2 = {k: {v for v in sorted(gt_idces_dict2[k])} for k in sorted(gt_idces_dict2)}
    avg_num_mmgt = sum([len(v) for v in gt_idces_dict2.values()])/len(gt_idces_dict2)
    std_num_mmmgt = np.std([len(v) for v in gt_idces_dict2.values()])/len(gt_idces_dict2)
    print(f'Average number of similar trajectories: {avg_num_mmgt}. Standard deviation: {std_num_mmmgt}. Total n of mmgt: {sum([len(v) for v in gt_idces_dict2.values()])}')
    return gt_idces_dict2