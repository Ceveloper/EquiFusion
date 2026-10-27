"""Pre-processing of the Nymeria dataset for human motion prediction (first used for this task in EquiFusion).

    python -m src.data.create_nymeria_dataset          (from the repository root)
    python -m src.data.create_nymeria_dataset --full   (full version, see below)

Input:  datasets/raw/Nymeria/<sequence>/body_motion_unzipped/body/xdata.npz  (XSens motion capture, 240 fps, 23 joints)
        datasets/annotations/Nymeria/hmp/{train,valid,test}.txt              (sequences of each split)
Output: datasets/processed/Nymeria/hmp/data_3d_nymeria.npz                   (positions_3d[split][sequence] = [T, 23, 3] float32, meters)
        datasets/processed/Nymeria/hmp/data_3d_nymeria_idxs.json             (frame window kept for each sequence)

Processing of each sequence:
  1. subsample 240 fps -> 60 fps (every 4th frame);
  2. trim the static start and end: drop the leading (trailing) frames until at least one joint has moved by more
     than 4 cm from its position in the first (last) frame (`trim_static_start_end`);
  3. select the frame windows of the subset used in the paper (`select_motion_windows`): about the size of the AMASS
     training set (43 h of training data); windows shorter than 150 frames (0.5 s observation + 2 s prediction) are dropped.
  4. sequences with more than one window are stored as <sequence>_<i>.
The sequences are written in alphabetical order. (The file used for the paper has the same content in a different order
of the sequences, which only changes the order of the training samples.)

Full version (--full): keeps all motion of each sequence (211 h of training data instead of 43 h); in our experiments it gave no
gain in the metrics. Steps 1, 2 and 4 are the same; step 3 is replaced by `select_motion_windows_full`: a frame is kept if
the joint speed relative to the root, averaged over the joints and over a centered window of 150 frames, is at least
2 cm/s; pauses of at most 30 frames are bridged and windows shorter than 150 frames are dropped. Same split of the
sequences; the test segments (a new segment every 150 frames, i.e. non-overlapping) are created by the script.
Output: datasets/processed/Nymeria_full/hmp/ and datasets/annotations/Nymeria_full/hmp/{segments_test.csv, mean_motion_test.txt}.
"""
import argparse
import os
import json
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial

from .loaders import NymeriaDataset
from .skeleton import get_skeleton_class
from .loaders.base.create_dataset_utils import compute_stats


OUTPUT_3D = 'data_3d_nymeria'
RAW_FPS = 240
TRIM_THRESHOLD = 0.04     # m, max over joints of the displacement from the first (last) frame (step 2)
MOTION_THRESHOLD = 0.01   # m per frame, mean over joints (step 3)
MIN_LENGTH = 30 + 120     # frames, obs_length + pred_length at 60 fps
FULL_SPEED_THRESHOLD = 0.02  # m/s, joint speed relative to the root, averaged over MIN_LENGTH frames (--full)
FULL_MAX_GAP = 30            # frames, longest pause bridged inside a window (--full)


def trim_static_start_end(seq, threshold=TRIM_THRESHOLD):
    """Drop the frames at the start (end) of the sequence until a joint has moved more than `threshold` from its position in the first (last) frame."""
    def _first_moving_frame(seq):
        for i, joints3d in enumerate(seq[1:], start=1):
            max_move = (((joints3d - seq[0]) ** 2).sum(axis=-1)**0.5).max(axis=-1)
            if max_move >= threshold:
                return i
        assert 0, "No motion detected in sequence"

    startidx = _first_moving_frame(seq)
    endidx = _first_moving_frame(np.flip(seq, axis=0))
    return seq[startidx:-endidx]


def get_consecutive_idxs(unique_idxs, merge_threshold=5, min_length=MIN_LENGTH):
    """Windows (start, end) of consecutive entries of `unique_idxs`, merging windows closer than `merge_threshold` and dropping the ones shorter than `min_length`.
    The windows are positions in `unique_idxs`."""
    diff = np.diff(unique_idxs, axis=0) == 1
    start = 0
    end = 0
    val_prev = False
    consecutive_idxs = []
    for idx, val in enumerate(diff):
        if not val and val_prev:
            end = idx
            consecutive_idxs.append((start, end))
        if not val_prev and val:
            start = idx
        val_prev = val
    if val_prev:
        end = idx
        consecutive_idxs.append((start, end))
    merged_consecutive_idxs = []
    i = 0
    while i < len(consecutive_idxs) - 1:
        start = consecutive_idxs[i][0]
        end = consecutive_idxs[i][1]
        while i < len(consecutive_idxs) - 1 and consecutive_idxs[i][1] + merge_threshold >= consecutive_idxs[i + 1][0]:
            end = consecutive_idxs[i + 1][1]
            i += 1
        merged_consecutive_idxs.append((start, end))
        i += 1
    if i == len(consecutive_idxs) - 1:
        merged_consecutive_idxs.append(consecutive_idxs[-1])
    return [window for window in merged_consecutive_idxs if (window[1] - window[0] >= min_length)]


def select_motion_windows(kpts3d, threshold=MOTION_THRESHOLD):
    """Windows of the sequence that are kept (step 3 of the module docstring)."""
    mean_joint_displacement = ((np.diff(kpts3d, axis=0) ** 2).sum(axis=-1)**0.5).mean(axis=-1)
    moving_frames = np.argwhere(mean_joint_displacement > threshold)[..., 0]
    return get_consecutive_idxs(moving_frames)


def select_motion_windows_full(kpts3d, threshold=FULL_SPEED_THRESHOLD, window=MIN_LENGTH, max_gap=FULL_MAX_GAP, fps=60):
    """Windows [start, end) of the sequence that are kept in the full version (--full, see the module docstring)."""
    rel = kpts3d - kpts3d[:, :1]
    speed = np.linalg.norm(np.diff(rel, axis=0), axis=-1).mean(-1) * fps  # [T-1] m/s
    keep = np.append(np.convolve(speed, np.ones(window) / window, mode="same") >= threshold, False)
    idx = np.flatnonzero(keep)
    if not len(idx):
        return []
    windows, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > max_gap + 1:
            windows.append((start, prev + 1))
            start = i
        prev = i
    windows.append((start, prev + 1))
    return [(int(a), int(b)) for a, b in windows if b - a >= MIN_LENGTH]


def write_test_segments(test_data, path, obs_length=30, pred_length=120):
    """Non-overlapping test segments (a new one every obs_length + pred_length frames) of the full version."""
    seg = obs_length + pred_length
    rows = [(name, init, init + obs_length, init + seg - 1)
            for name in sorted(test_data) for init in range(0, len(test_data[name]) - seg + 1, seg)]
    pd.DataFrame(rows, columns=["name", "init", "pred_init", "pred_end"]).to_csv(path, index=False)
    print(f"{len(rows)} test segments written to {path}")


def load_sequence(raw_data_path, seq, fps):
    data = np.load(os.path.join(raw_data_path, seq, 'body_motion_unzipped/body/xdata.npz'))
    assert data['frameRate'] == RAW_FPS, f"The frame rate is not {RAW_FPS}."
    kpts3d = data['segment_tXYZ']
    kpts3d = kpts3d.reshape(kpts3d.shape[0], -1, 3)
    return kpts3d[::int(np.round(RAW_FPS / fps))]


def create_npz_dataset_file(raw_data_path, annotations_folder, destination_folder, fps=60, full=False):
    output_path = os.path.join(destination_folder, f"{OUTPUT_3D}.npz")
    if os.path.exists(output_path):
        raise Exception(f'The dataset already exists at {output_path}')

    output = {}
    idx_dict = {}
    n_frames, n_frames_discarded, n_windows = 0, 0, []
    for split in ['train', 'valid', 'test']:
        with open(os.path.join(annotations_folder, f"{split}.txt"), 'r') as file:
            sequences = sorted(set('_'.join(line.strip().split('_')[:6]) for line in file.readlines()))
        print("Processing split: ", split)
        output[split] = {}
        for seq in tqdm(sequences):
            kpts3d = load_sequence(raw_data_path, seq, fps)
            n_frames_seq = kpts3d.shape[0]
            kpts3d = trim_static_start_end(kpts3d)
            windows = select_motion_windows_full(kpts3d) if full else select_motion_windows(kpts3d)
            if len(windows) == 1:
                output[split][seq] = kpts3d[windows[0][0]:windows[0][1]].astype('float32')
            else:
                for i, (start, end) in enumerate(windows):
                    output[split][seq + f'_{i}'] = kpts3d[start:end].astype('float32')
                n_windows.append(len(windows))
            kept = sum(end - start for start, end in windows)
            n_frames += kept
            n_frames_discarded += n_frames_seq - kept
            idx_dict[seq] = windows

    if full:  # alphabetical order of the stored windows
        output = {split: {k: output[split][k] for k in sorted(output[split])} for split in output}
    print(f"Processed {len(idx_dict)} sequences, kept {n_frames} frames, discarded {n_frames_discarded} frames.")
    print(f"{len(n_windows)} sequences are split into several windows ({sum(n_windows)} in total).")
    print(f'Saving into "{output_path}"...')
    np.savez_compressed(output_path, positions_3d=output)
    with open(os.path.join(destination_folder, f"{OUTPUT_3D}_idxs.json"), "w") as f:
        json.dump(idx_dict, f, sort_keys=True, indent=4)
    print('Done.')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="full version: keep all motion of each sequence (see the module docstring)")
    args = parser.parse_args()
    fps = 60
    obs_length, pred_length = 30, 120
    num_joints = 23
    dataset_name = "Nymeria"
    original_folder = "./datasets/raw/Nymeria/"
    annotations_folder = f"./datasets/annotations/{dataset_name}"
    precomputed_folder = f"./datasets/processed/{dataset_name}{'_full' if args.full else ''}/hmp/"
    annotations_folder_task = os.path.join(annotations_folder, "hmp")  # splits of the sequences
    os.makedirs(precomputed_folder, exist_ok=True)
    np.random.seed(0)

    output = create_npz_dataset_file(original_folder, annotations_folder_task, precomputed_folder, fps=fps, full=args.full)
    if args.full:
        annotations_folder_task = f"./datasets/annotations/{dataset_name}_full/hmp"
        os.makedirs(annotations_folder_task, exist_ok=True)
        write_test_segments(output["test"], os.path.join(annotations_folder_task, "segments_test.csv"), obs_length, pred_length)

    # Reference values of the CMD metric. For the subset used in the paper they are part of the repository
    # (datasets/annotations/Nymeria/hmp/mean_motion_test.txt, read by the evaluation) and are not recomputed; the
    # computation below reproduces them exactly. The full version has its own test segments and computes its values.
    if args.full:
        print("=" * 50)
        print("Computing values for CMD.")
        NymeriaSkeletonCenterPose = get_skeleton_class(motion_repr_type="SkeletonCenterPose", dataset_name=dataset_name.lower())
        compute_stats(precomputed_folder=precomputed_folder,
                      Dataset=partial(NymeriaDataset, annotations_folder=annotations_folder_task, split='test', file_idces="all", stride=10,
                                      segments_path=os.path.join(annotations_folder_task, "segments_test.csv"), obs_length=obs_length, pred_length=pred_length),
                      Skeleton=[partial(NymeriaSkeletonCenterPose, obs_length=obs_length, pred_length=pred_length, num_joints=num_joints), None],
                      if_consider_hip=False, batch_size=128)
        shutil.copy(os.path.join(precomputed_folder, "mean_motion_test.txt"), os.path.join(annotations_folder_task, "mean_motion_test.txt"))
