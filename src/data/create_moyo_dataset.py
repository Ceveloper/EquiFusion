import os
import argparse
import tarfile
from io import BytesIO
import zarr
import numpy as np
import torch
from tqdm import tqdm
from functools import partial


from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c

from .loaders import MOYOZeroShotDataset, AMASSDataset
from .skeleton import get_skeleton_class
from .loaders.base.create_dataset_utils import compute_stats, compute_multimodal_gt, compute_multimodal_gt_onsplit


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['PYOPENGL_PLATFORM'] = 'egl' # https://github.com/mmatl/pyrender/issues/13
TARGET_OPEN_GL_MAJOR = 3
TARGET_OPEN_GL_MINOR = 3
OUTPUT_3D = 'data_3d_moyo'

        
def create_npz_dataset_file(datasets, precomputed_folder, destination_folder, models_dir, comp_device, num_joints=22):
    output_path = os.path.join(destination_folder, f"{OUTPUT_3D}.npz")

    if os.path.exists(output_path):
        print('The npz dataset already (partially) exists at', output_path)
        raise Exception(f'The dataset already exists at {output_path}')
    
    # initialize all needed resources
    num_betas = 16 # number of body parameters
    bm_fname = os.path.join(models_dir, 'smplh/female/model.npz')
    print("Using comp_device:", comp_device)
    bm_female = BodyModel(bm_path=bm_fname, num_betas=num_betas, model_type='smplh').to(comp_device)
    print("All resources initialized")


    output = {}
    counter = 0
    n_frames = 0

    print("Loading datasets: ", datasets)
    for i,dataset in enumerate(datasets):
        split_path = os.path.join(precomputed_folder, dataset)
        sequences = sorted([p.split(".")[0] for p in os.listdir(split_path) if p.endswith('npz')])
        print(f"[{i+1}/{len(datasets)}] Processing {dataset}...")
        output[dataset] = {}

        for file_idx in sequences:
            seq = np.load(os.path.join(split_path, f'{file_idx}.npz'))
            body_parms = {
                'root_orient': torch.Tensor(seq['poses'][:, :3]).to(comp_device), # controls the global root orientation
                'pose_body': torch.Tensor(seq['poses'][:, 3:66]).to(comp_device), # controls the body
                'pose_hand': torch.Tensor(seq['poses'][:, 66:]).to(comp_device), # controls the finger articulation
                'trans': torch.Tensor(seq['trans']).to(comp_device), # controls the global body position
                'betas': torch.Tensor(np.repeat(seq['betas'][:num_betas][np.newaxis], repeats=len(seq['poses']), axis=0)).to(comp_device), # controls the body shape. Body shape is static
                            }  
            body_pose_hand = bm_female(**body_parms) 
            body_joints = c2c(body_pose_hand.Jtr).copy()

            output[dataset][file_idx] = body_joints[:, :num_joints, ...]
            counter += 1
            n_frames += len(seq)

    print(f"Processed total {counter} sequences.")
    print(f"Total number of frames: {n_frames}")
    print(f'Saving into "{output_path}"...')
    np.savez_compressed(output_path, positions_3d=output)       
    return output 

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process Raw Data')

    parser.add_argument('-gpu', '--gpu', action='store_true', help='Use GPU for processing')
    parser.add_argument('-include_hands', '--include_hands', action='store_true', help='Include 30 hand joints and create -MANO version of the dataset')

    args = parser.parse_args()

    FPS = 60
    include_hands = args.include_hands

    num_joints = 22 if not include_hands else 22+30
    dataset_name = "MOYO" if not include_hands else "MOYO-MANO"
    batch_size = 128
    original_folder = f"./datasets/raw/MOYO/"
    annotations_folder = f"./datasets/annotations/{dataset_name}/dataset"
    annotations_folder_csv = f"./datasets/annotations/{dataset_name}"
    precomputed_folder = f"./datasets/processed/{dataset_name}/"
    models_dir = f'./datasets/annotations/AMASS/bodymodels/'
    stride = 1
    augmentation = 0
    
    multimodal_threshold = 0.4 # as in BeLFusion

    
    comp_device = 'cpu' if not args.gpu else 'cuda'
    print("Using comp_device:", comp_device)
            
    datasets = [ "train", "val"]
    actions = "all" 
    

    
    np.random.seed(0)

    file_idces = "all"

    for task in ["hmp"]:
        if_consider_hip = False if task == "hmp" else True
        precomputed_folder_task = f"{precomputed_folder}/{task}/"
        annotations_folder_task = os.path.join(annotations_folder_csv, task)
        os.makedirs(precomputed_folder_task, exist_ok=True)
        
        try:
            sequences = create_npz_dataset_file(datasets, original_folder, precomputed_folder_task, models_dir=models_dir, comp_device=comp_device, num_joints=num_joints)
        except Exception as e:
            print(e)
        
        obs_length = 30
        pred_length = 120


        print("="*50)
        print("Computing values for CMD.")
        segments_path = os.path.join(annotations_folder_task, f"segments_test_zero_shot.csv")
        # below to compute the average motion per class for the TEST set, for the CMD computation
        
                
        MOYOSkeletonCenterPose = get_skeleton_class(motion_repr_type="SkeletonCenterPose", dataset_name="moyo")
        skeleton = MOYOSkeletonCenterPose(if_consider_hip=False, obs_length=obs_length, pred_length=pred_length, actions=actions, num_joints=num_joints)
        Dataset=partial(MOYOZeroShotDataset, skeleton=skeleton, obs_length=obs_length, pred_length=pred_length, if_consider_hip=False,
                                            file_idces=file_idces, stride=stride)
        print("="*50)
        print(f"Computing Multimodal Ground Truth for task {task}.")
        compute_multimodal_gt(Dataset=partial(MOYOZeroShotDataset, skeleton=skeleton, obs_length=obs_length, pred_length=pred_length, if_consider_hip=if_consider_hip,
                                                 stride=stride),
                            multimodal_threshold=multimodal_threshold, split='test', segment_path=os.path.join(annotations_folder_task,f"segments_test_zero_shot.csv"),
                            annotations_folder=annotations_folder_task, precomputed_folder=precomputed_folder_task,
                                )