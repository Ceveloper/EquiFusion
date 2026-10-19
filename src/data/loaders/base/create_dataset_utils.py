import os
import numpy as np
import json
import torch
from torch.utils.data import DataLoader
from functools import partial

from .math_utils import compute_mean_motions, get_multimodal_gt
from .motion_dataset import MotionDataset


def compute_stats(precomputed_folder, Skeleton, Dataset: MotionDataset, if_consider_hip:bool, batch_size = 128, overwrite=False):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    skeleton = Skeleton[0](if_consider_hip=if_consider_hip)
    dataset = Dataset( split='test', skeleton=skeleton, precomputed_folder=precomputed_folder, 
                        if_consider_hip=if_consider_hip, augmentation=0, da_mirroring=0.0, da_rotations=0.0,
                        dtype="float32")
    dest_file = os.path.join(precomputed_folder, "mean_motion_test.txt")
    if not os.path.exists(dest_file) or overwrite:
        class_average_3d, list_of_motions_3d, frequencies = compute_mean_motions(dataset, batch_size=batch_size)

        with open(dest_file, 'w') as filehandle:
            filehandle.write('\n'.join([f"{c},{meanmot},{freq}" for c, meanmot, freq in zip(list(class_average_3d.keys()), list_of_motions_3d, frequencies)]))
    else: 
        print("Files for Mean motion values already exist. Skipping.")
    
    if if_consider_hip == True: # dataset must deliver items with hip
        dest_file = os.path.join(precomputed_folder, "mean_motion_traj_test.txt")
        if not os.path.exists(dest_file) or overwrite:
            class_average_3d, list_of_motions_3d, frequencies = compute_mean_motions(dataset, batch_size=batch_size, if_traj_only=True)
            with open(dest_file, 'w') as filehandle:
                filehandle.write('\n'.join([f"{c},{meanmot},{freq}" for c, meanmot, freq in zip(list(class_average_3d.keys()), list_of_motions_3d, frequencies)]))
        else: 
            print("Files for Mean motion values for MOTPRED task - hip - already exist. Skipping.")
        # compute stats for centered pose
        if_consider_hip = False 
        dest_file = os.path.join(precomputed_folder, "mean_motion_pose_test.txt")
        if not os.path.exists(dest_file) or overwrite:
            skeleton = Skeleton[1](if_consider_hip=if_consider_hip)
            dataset = Dataset(split='test', skeleton=skeleton, precomputed_folder=precomputed_folder, if_consider_hip=if_consider_hip,
                                augmentation=0, da_mirroring=0.0, da_rotations=0.0,
                                dtype="float32")
            
            class_average_3d, list_of_motions_3d, frequencies = compute_mean_motions(dataset, batch_size=batch_size)
            with open(dest_file, 'w') as filehandle:
                filehandle.write('\n'.join([f"{c},{meanmot},{freq}" for c, meanmot, freq in zip(list(class_average_3d.keys()), list_of_motions_3d, frequencies)]))
        else: 
            print("Files for Mean motion values for MOTPRED task - pose - already exist. Skipping.")        
    
    
def compute_multimodal_gt(Dataset: MotionDataset,
                          annotations_folder: str, precomputed_folder: str, split='test',
                          multimodal_threshold=0.5, segment_path="/hmp/segments_0.5s-2s", batch_size=2048):
    mmgt_path = os.path.join(precomputed_folder, f"mmgt_{split}.txt")
    if os.path.exists(mmgt_path):
        print("Multimodal GT already exists. Skipping.")
        return
    mmgt = compute_multimodal_gt_onsplit(Dataset=partial(Dataset, split=split, segments_path = segment_path, 
                                    annotations_folder=annotations_folder, precomputed_folder=precomputed_folder),
                        dest_path=mmgt_path, 
                        multimodal_threshold=multimodal_threshold, batch_size=batch_size)

    import ast
    with open(os.path.join(precomputed_folder, f"mmgt_{split}.txt"), 'r') as filehandle:
        mmgt_loaded = ast.literal_eval(json.load(filehandle))
    assert mmgt_loaded==mmgt, "Loaded multimodal GT is different from the original one."

def compute_multimodal_gt_onsplit(Dataset: MotionDataset,
                          dest_path: str, 
                          multimodal_threshold=0.5, batch_size=2048):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    dataset = Dataset(#annotations_folder=annotations_folder, precomputed_folder=precomputed_folder,
                        augmentation=0, da_mirroring=0.0, da_rotations=0.0, dtype="float32")

    data_loader = DataLoader(dataset, pin_memory= True, shuffle=False, batch_size=batch_size, 
                                num_workers=0, drop_last=False)    
    mmgt = get_multimodal_gt(data_loader, multimodal_threshold=multimodal_threshold, recover_landmarks=True)#, store_tempfile=dest_path if 'train' in dest_path else None) # recover_landmarks makes no difference for CenterPoseSkeleton
    # analyze and store
    with open(dest_path, 'w') as filehandle:
        json.dump(str(mmgt), filehandle)
    return mmgt
