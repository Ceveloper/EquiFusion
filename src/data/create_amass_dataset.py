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

from .loaders import AMASSDataset
from .loaders.base.create_dataset_utils import compute_stats, compute_multimodal_gt, compute_multimodal_gt_onsplit
from .skeleton import get_skeleton_class


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['PYOPENGL_PLATFORM'] = 'egl' # https://github.com/mmatl/pyrender/issues/13
TARGET_OPEN_GL_MAJOR = 3
TARGET_OPEN_GL_MINOR = 3
OUTPUT_3D = 'data_3d_amass'

def process_data(path, out, target_fps, bm, num_betas, num_joints=52):
    print('Processing', path)
    if os.path.exists(os.path.join(out, 'poses.zarr') or os.path.join(out, 'trans.zarr') or os.path.join(out, 'poses_index.zarr')):
        print('The dataset already (partially) exists at', out)
        raise Exception(f'The dataset already exists at {out}')
    from human_body_prior.tools.omni_tools import copy2cpu as c2c

    z_poses = zarr.open(os.path.join(out, 'poses.zarr'), mode='w', shape=(0, num_joints, 3), chunks=(1000, num_joints, 3), dtype=np.float32)
    z_trans = zarr.open(os.path.join(out, 'trans.zarr'), mode='w', shape=(0, 3), chunks=(1000, 3), dtype=np.float32)
    z_index = zarr.open(os.path.join(out, 'poses_index.zarr'), mode='w', shape=(0, 2), chunks=(1000, 2), dtype=int)
    i = 0
    mean_vel = 0
    count = 0
    tar = tarfile.open(path, 'r')
    for member in tqdm(tar):
        file_name = os.path.basename(member.name)
        if file_name.endswith('.npz') and not file_name.startswith('.'):
            try:
                with tar.extractfile(member) as f:
                    array_file = BytesIO()
                    array_file.write(f.read())
                    array_file.seek(0)
                    bdata = np.load(array_file)

                    if 'mocap_framerate' not in bdata and 'mocap_frame_rate' not in bdata:
                        print(f"WARNING: we skip '{member.name}' because it is corrupted (no framerate)")
                        continue
                    else:
                        frame_rate = bdata['mocap_framerate'] if 'mocap_framerate' in bdata else bdata['mocap_frame_rate']
                    gender = str(bdata["gender"])
                    if gender == "b'female'":
                        gender = "female" # this is a common problem in SSM dataset

                    if target_fps == -1:
                        fps = frame_rate
                    else:
                        fps = target_fps

                    frame_multiplier = int(np.round(frame_rate / fps))

                    time_length = len(bdata['trans'])
                    body_parms = {
                        'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device), # controls the global root orientation
                        'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device), # controls the body
                        'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(comp_device), # controls the finger articulation
                        'trans': torch.Tensor(bdata['trans']).to(comp_device), # controls the global body position
                        'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(comp_device), # controls the body shape. Body shape is static
                        #'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(comp_device) # controls soft tissue dynamics
                    }

                    body_pose_hand = bm[gender](**{k:v for k,v in body_parms.items() if k in [
                        'pose_body', 'betas', 'pose_hand', 'root_orient', 'trans'
                        ]})
                    
                    body_joints = c2c(body_pose_hand.Jtr)[:, :num_joints].copy()[::frame_multiplier]
                    body_trans = bdata['trans'][::frame_multiplier]
                    mean_vel += ((np.diff(body_joints, axis=0)** 2).sum(axis=-1)**0.5).mean()
                    count += 1
                    z_poses.append(body_joints, axis=0)
                    z_trans.append(body_trans, axis=0)
                    z_index.append(np.array([[i, i + body_joints.shape[0]]]), axis=0)
                    i = i + body_joints.shape[0]
            except Exception as e:
                print(e, ". Filename:", file_name)
    print("Mean velocity:", mean_vel/count)


def extract_dataset(original_folder, precomputed_folder, models_dir, datasets=None):

    
    # initialize all needed resources
    genders = "male", "female"
    num_betas = 16 # number of body parameters
    num_dmpls = 8 # number of DMPL parameters

    bm, faces = {}, {}
    for gender in genders:
        bm_fname = os.path.join(models_dir, 'smplh/{}/model.npz'.format(gender))
        dmpl_fname = os.path.join(models_dir, 'dmpls/{}/model.npz'.format(gender))

        bm[gender] = BodyModel(bm_path=bm_fname, num_betas=num_betas, model_type='smplh').to(comp_device)
        faces[gender] = c2c(bm[gender].f)

    print("All resources initialized")

    if datasets is None:
        # we process all datasets in 'original_folder' folder
        datasets = sorted([p.split(".")[0] for p in os.listdir(original_folder)])
        
    for i, dataset in enumerate(datasets):
        print(f"[{i+1}/{len(datasets)}] Processing {dataset}...")
        try:
            process_data(os.path.join(original_folder, dataset + '.tar.bz2'), os.path.join(precomputed_folder, dataset), fps, bm, num_betas)
        except Exception as e:
            print(e)
        
def create_npz_dataset_file(datasets, precomputed_folder, output_path, num_joints=22):

    output = {}
    counter = 0
    n_frames = 0
    mean_vel = 0

    print("Loading datasets: ", datasets)
    for dataset in datasets:
        output[dataset] = {}

        z_poses = zarr.open(os.path.join(precomputed_folder, dataset, 'poses.zarr'), mode='r')
        z_trans = zarr.open(os.path.join(precomputed_folder, dataset, 'trans.zarr'), mode='r')
        z_index = zarr.open(os.path.join(precomputed_folder, dataset, 'poses_index.zarr'), mode='r')

        # we build the feature vectors for each dataset and file_idx
        for file_idx in range(z_index.shape[0]):

            i0, i = z_index[file_idx]
            seq = z_poses[i0:i]

            output[dataset][file_idx] = seq[:, :num_joints, ...]
            if len(output[dataset][file_idx]) > 1:
                mean_vel += ((np.diff(output[dataset][file_idx], axis=0)** 2).sum(axis=-1)**0.5).mean()

            counter += 1
            n_frames += len(seq)

    print(f"Processed total {counter} sequences.")
    print(f"Total number of frames: {n_frames}")
    print("Mean velocity:", mean_vel/counter)
    print(f'Saving into "{output_path}"...')
    np.savez_compressed(output_path, positions_3d=output)       
    return output 

amass_official_splits = {
        'validation': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
        'test': ['Transitions_mocap', 'SSM_synced'],
        'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BML', 'EKUT', 'TCD_handMocap', 'ACCAD']#
    }


# it will pre-process the AMASS dataset
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='AMASS Process Raw Data')

    parser.add_argument('--fps',
                        type=int,
                        default=60,
                        help='FPS')
    parser.add_argument('--datasets',
                        type=str,
                        nargs="+",
                        help='The names of the datasets to process',
                        default=None)
    parser.add_argument('-gpu', '--gpu', action='store_true', help='Use GPU for processing')
    parser.add_argument('-if_extract_zip', '--if_extract_zip', action='store_true', help='Extract the zip files into zarr archives')
    parser.add_argument('-include_hands', '--include_hands', action='store_true', help='Create AMASS dataset including the hand joints from MANO')


    args = parser.parse_args()

    fps = args.fps
    FPS = fps
    datasets = args.datasets
    include_hands = args.include_hands
    if_extract_zip = args.if_extract_zip


    num_joints = 22 if not include_hands else 22+30
    dataset_name = "AMASS" if not include_hands else "AMASS-MANO"
    batch_size = 128
    original_folder = f"./datasets/raw/AMASS/"
    annotations_folder = f"./datasets/annotations/{dataset_name}/dataset"
    annotations_folder_csv = f"./datasets/annotations/{dataset_name}"
    precomputed_folder = f"./datasets/processed/{dataset_name}/"
    models_dir = f'./datasets/annotations/AMASS/bodymodels/'
    stride = 10
    augmentation = 0
    
    multimodal_threshold = 0.4 # as in BeLFusion

    
    comp_device = 'cpu' if not args.gpu else 'cuda'
    print("Using device:", comp_device)
    
    if if_extract_zip:
        # list all datasets
        print("Datasets to process:")
        print(f"Extracting all tar.bz2 files from the datasets folder {original_folder} and create zarr files {annotations_folder}")
        print(datasets)
        extract_dataset(original_folder, annotations_folder, models_dir, datasets)
        
        
    dataset_training = ["ACCAD", "BMLhandball", "BMLmovi", "BMLrub", "CMU", "EKUT", "EyesJapanDataset", "KIT", "PosePrior", "TCDHands", "TotalCapture"]
    dataset_validation = [ "HumanEva", "HDM05", "SFU", "MoSh" ]
    dataset_test = ['DFaust', 'DanceDB', 'GRAB', 'HUMAN4D', 'SOMA', 'SSM', 'Transitions']
    
    datasets = [*dataset_training, *dataset_validation, *dataset_test]
    

    
    np.random.seed(0)

    file_idces = "all"

    for task in ["hmp"]:
        if_consider_hip = False if task == "hmp" else True
        precomputed_folder_task = f"{precomputed_folder}/{task}/"
        annotations_folder_task = os.path.join(annotations_folder_csv, task)
        os.makedirs(precomputed_folder, exist_ok=True)
        
        output_npz_path = os.path.join(precomputed_folder_task, f"{OUTPUT_3D}.npz")

        if os.path.exists(output_npz_path):
            print('The npz dataset already (partially) exists at', output_npz_path)        
        else: 
            sequences = create_npz_dataset_file(datasets, annotations_folder, output_npz_path, num_joints=num_joints)
        
        if task == "hmp":
            obs_length = 30
            pred_length = 120

        else:
            obs_length = 30
            pred_length = 120           

        AMASSSkeletonVanilla = get_skeleton_class(motion_repr_type="SkeletonVanilla", dataset_name=dataset_name.lower())
        AMASSSkeletonCenterPose = get_skeleton_class(motion_repr_type="SkeletonCenterPose", dataset_name=dataset_name.lower() )
        segments_path = os.path.join(annotations_folder_csv, task, "segments_test.csv")

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
                    skeletons = [partialize_skel_class(AMASSSkeletonCenterPose), None]
                else: 
                    skeletons = [partialize_skel_class(AMASSSkeletonVanilla), partialize_skel_class(AMASSSkeletonCenterPose)]
                return skeletons
            compute_stats(precomputed_folder=precomputed_folder_task,
                            Dataset=partial(AMASSDataset,annotations_folder=annotations_folder_task, datasets=dataset_test, file_idces=file_idces, stride=stride,  segments_path=segments_path, obs_length=obs_length, pred_length=pred_length),
                            Skeleton=return_skeletons_for_stats_funct(task), 
                            if_consider_hip=if_consider_hip, batch_size=batch_size)        
                    
        if task == "hmp":
            skeleton= AMASSSkeletonCenterPose(if_consider_hip=if_consider_hip, obs_length=obs_length, pred_length=pred_length, file_idces=file_idces, num_joints=num_joints) #H36MSkeletonCenterPose
        else:
            skeleton= AMASSSkeletonVanilla(if_consider_hip=False, obs_length=obs_length, pred_length=pred_length, file_idces=file_idces, num_joints=num_joints)
            multimodal_threshold = 0.79
        Dataset=partial(AMASSDataset, skeleton=skeleton, obs_length=obs_length, pred_length=pred_length, if_consider_hip=False,
                                            file_idces=file_idces, stride=stride)
        print("="*50)
        if os.path.exists(os.path.join(precomputed_folder_task, f"mmgt_test.txt")):
            print('Multimodal Ground Truth already (partially) exists at', os.path.join(precomputed_folder_task, f"mmgt_test.txt"))        
        else: 
            print(f"Computing Multimodal Ground Truth for task {task}.")
            def create_dataset(split, **kwargs):
                if split=="train":
                    return Dataset(datasets=dataset_training, split=split,  **kwargs)
                elif split=="test":
                    return Dataset(datasets=dataset_test, split=split, **kwargs)
                elif split=="valid":
                    return Dataset(datasets=dataset_validation, split=split, **kwargs)
            compute_multimodal_gt(Dataset=create_dataset,
                                multimodal_threshold=multimodal_threshold, segment_path= segments_path, split='test',
                                annotations_folder=annotations_folder_task, precomputed_folder=precomputed_folder_task,
                                    )
