import os
import numpy as np
from tqdm import tqdm
import cdflib
from glob import glob
from functools import partial

from .skeleton.kinematic.h36m import CONVERSION_IDX_H36M_32TO17, CONVERSION_IDX_H36M_32TO25
from .loaders import H36MDataset
from .loaders.base.create_dataset_utils import compute_stats, compute_multimodal_gt
from .skeleton import get_skeleton_class

OUTPUT_3D = 'data_3d_h36m'
SUBJECTS = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']


def preprocess_dataset(dataset_folder, num_joints=17, output_path=OUTPUT_3D, subjects=SUBJECTS):
    
    if os.path.exists(f"{output_path}.npz"):
        print('The dataset already exists at', f"{output_path}.npz")
        raise Exception(f'The dataset already exists at {output_path}')
        
    print('Converting original H36M dataset from', dataset_folder)
    output = {}
    count = 0
    nframes = 0
    mean_vel = 0

    for subject in tqdm(subjects):
        output[subject] = {}
        file_list = glob(os.path.join(os.path.abspath(dataset_folder), subject, 'MyPoseFeatures', 'D3_Positions', '*.cdf'))
        assert len(file_list) == 30, "Expected 30 files for subject " + subject + ", got " + str(len(file_list))
        for f in file_list:
            action = os.path.splitext(os.path.basename(f))[0]
            
            if subject == 'S11' and action == 'Directions':
                continue # Discard corrupted video
                
            # Use consistent naming convention
            canonical_name = action.replace('TakingPhoto', 'Photo') \
                                    .replace('WalkingDog', 'WalkDog').replace(' ', '_')
            
            hf = cdflib.CDF(f)
            positions = hf['Pose'].reshape(-1, 32, 3)
            if num_joints == 17:
                positions = positions[:, CONVERSION_IDX_H36M_32TO17, :]
            elif num_joints == 25:
                positions = positions[:, CONVERSION_IDX_H36M_32TO25, :]
            else: 
                assert 0, "Not implemented for this number of joints" 
            
            positions /= 1000 # Meters instead of millimeters
            output[subject][canonical_name] = positions.astype('float32')
            if len(output[subject][canonical_name]) > 1:
                mean_vel += ((np.diff(output[subject][canonical_name], axis=0)** 2).sum(axis=-1)**0.5).mean()

            count += 1
            nframes += positions.shape[0]
            
    print(f"Processed total {count} sequences.")
    print("Mean velocity:", mean_vel/count)
    print(f"Total number of frames: {nframes}")
    print(f'Saving into "{output_path}"...')
    np.savez_compressed(output_path, positions_3d=output)        
    print('Done.')
    return output
        
import argparse


# python -m data_loader.h36m
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Human3.6M Process Raw Data')

    parser.add_argument('-gpu', '--gpu', action='store_true', help='Use GPU for processing')

    args = parser.parse_args()

    np.random.seed(0)
    comp_device = 'cpu' if not args.gpu else 'cuda'
    print("Using device:", comp_device)
        
    SUBJ_training = ["S1", "S5", "S6", "S7", "S8"]
    SUBJ_valid = ["S8"]
    SUBJ_test = ["S9", "S11"]
    actions = "all" 

    batch_size = 128
    original_folder = "./datasets/raw/Human36M/"
    annotations_folder = "./datasets/annotations/Human36M/"
    annotations_folder_csv = f"./datasets/annotations/Human36M"

    dataset_name = 'H36M'
    FPS = 50
    stride = 1 # or 10????
    augmentation = 0
    multimodal_threshold = 0.5
    
    # precompute data and statistics
    for task in ["hmp"]:
        if_consider_hip = False if task == "hmp" else True
        precomputed_folder_task = f"./datasets/processed/Human36M/{task}/"
        annotations_folder_task = os.path.join(annotations_folder, task)
        os.makedirs(precomputed_folder_task, exist_ok=True)
        
        if task == "hmp":
            obs_length = 25
            pred_length = 100
            num_joints = 17
        else:
            obs_length = 25
            pred_length = 100
            num_joints = 17        
            
        output_npz_path = os.path.join(precomputed_folder_task, f"{OUTPUT_3D}.npz")

        if os.path.exists(output_npz_path):
            print('The npz dataset already (partially) exists at', output_npz_path)        
        else: 
            sequences = preprocess_dataset(original_folder, output_path=output_npz_path, num_joints=num_joints, subjects=SUBJECTS)            
        

        H36MSkeletonVanilla = get_skeleton_class(motion_repr_type="SkeletonVanilla", dataset_name=dataset_name.lower())
        H36MSkeletonCenterPose = get_skeleton_class(motion_repr_type="SkeletonCenterPose", dataset_name=dataset_name.lower() )
        segments_path = os.path.join(annotations_folder_task, "segments_test.csv")

        print("="*50)
        if os.path.exists(os.path.join(precomputed_folder_task, f"mean_motion_test.txt")):
            print('Values for CMD already (partially) exists at', os.path.join(precomputed_folder_task, f"mean_motion_test.txt"))        
        else: 
            print("Computing values for CMD.")

            # below to compute the average motion per class for the TEST set, for the CMD computation
            def return_skeletons_for_stats_funct(current_task):
                def partialize_skel_class(skel_class):
                    return partial(skel_class, obs_length=obs_length, pred_length=pred_length, num_joints=num_joints,)
                if current_task == "hmp":
                    skeletons = [partialize_skel_class(H36MSkeletonCenterPose), None]
                else: 
                    skeletons = [partialize_skel_class(H36MSkeletonVanilla), partialize_skel_class(H36MSkeletonCenterPose)]
                return skeletons
            compute_stats(precomputed_folder=precomputed_folder_task,
                          Dataset=partial(H36MDataset,annotations_folder=annotations_folder_task, num_joints=num_joints, subjects=SUBJ_test, actions=actions, stride=stride,  segments_path=segments_path, obs_length=obs_length, pred_length=pred_length),
                          Skeleton=return_skeletons_for_stats_funct(task), 
                          if_consider_hip=if_consider_hip, batch_size=batch_size)
        if task == "hmp":
            skeleton= H36MSkeletonCenterPose(if_consider_hip=if_consider_hip, obs_length=obs_length, pred_length=pred_length, num_joints=num_joints, actions=actions) #H36MSkeletonCenterPose
        else:
            skeleton= H36MSkeletonCenterPose(if_consider_hip=False, obs_length=obs_length, pred_length=pred_length, num_joints=num_joints, actions=actions)
            multimodal_threshold = 0.79
        Dataset=partial(H36MDataset, skeleton=skeleton, obs_length=obs_length, pred_length=pred_length, if_consider_hip=False,
                                            actions=actions, stride=stride)
        print("="*50)
        if os.path.exists(os.path.join(precomputed_folder_task, f"mmgt_test.txt")):
            print('Multimodal Ground Truth already (partially) exists at', os.path.join(precomputed_folder_task, f"mmgt_test.txt"))        
        else: 
            print(f"Computing Multimodal Ground Truth for task {task}.")
            def create_dataset(split, **kwargs):
                if split=="train":
                    return Dataset(subjects=SUBJ_training, split=split,  **kwargs)
                elif split=="test":
                    return Dataset(subjects=SUBJ_test, split=split, **kwargs)
                elif split=="valid":
                    return Dataset(subjects=SUBJ_valid, split=split, **kwargs)
            compute_multimodal_gt(Dataset=create_dataset,
                                multimodal_threshold=multimodal_threshold, segment_path= segments_path, split='test',
                                annotations_folder=annotations_folder_task, precomputed_folder=precomputed_folder_task,
                                    )
