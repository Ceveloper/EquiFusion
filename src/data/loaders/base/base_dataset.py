from torch.utils.data import Dataset
import numpy as np
import os
import torch
import json
import ast

MEAN_NAME = "mean_landmarks.npy"
VAR_NAME = "var_landmarks.npy"
MIN_NAME = "min_landmarks.npy"
MAX_NAME = "max_landmarks.npy"

NORMALIZATION_TYPES = ["standardize", "normalize"]

class BaseDataset(Dataset):
    def __init__(self, precomputed_folder, obs_length: int, pred_length: int, augmentation=0, stride=1, num_workers=8, normalize_data=False, normalize_type="standardize", dtype='float64', if_consider_hip=False, use_vel=False, silent=False):
        super().__init__()
        self.silent = silent
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.seg_length = obs_length + pred_length

        # 'annotations' and 'segments' need to be lists of numpy arrays of shape [frame, num_people, num_joints, num_features]
        # each list are clips where segments can be extracted.
        # if there is a clip which needs to be cut for a given reason, two lists need to be created.
        self.annotations = None 

        # segments are pre-processed and stored as: (idx, init, end), with idx being the idx of the clip stored in self.annotations
        # and 'init' and 'end' the starting and ending points of the segments to be predicted
        self.segments = []
        self.clip_idx_to_metadata = None # this needs to match each clip with its metadata. It is a list of tuples of length num_segments. The tuples have length2 and can contain only strings.
        self.segment_idx_to_metadata = None # this needs to match each clip with its metadata

        # augmentation and stride can be used together to make the epoch more thorough or not.
        # e.g., augmentation=5 and stride=20 => first segment will start at a random frame between 0 to 10 (augmentation of 5: 5 +/- 5), 
        # and next one at a random frame between 20 to 30 (stride of 20: 25 +/- 5)
        self.augmentation = augmentation
        self.stride = stride
        assert self.augmentation >= 0, f"[ERROR_AUGMENTATION] Augmentation must be non-negative, {self.augmentation}"

        # folder where precomputed values will be stored (e.g., mean and variance)
        self.precomputed_folder = precomputed_folder
        if not os.path.exists(self.precomputed_folder):
            os.makedirs(self.precomputed_folder)

        assert dtype.lower() in ["float64", "float32"], "Only dtypes=float64/32 supported in this project."
        self.dtype = np.float64 if dtype.lower() == 'float64' else np.float32

        # Define task: hmp without global translation or motion prediction with global translation
        self.drop_root = not if_consider_hip 
        self.if_consider_hip = if_consider_hip
        self.use_vel = use_vel
        assert use_vel == False, "Not implemented"

        self._prepare_data(num_workers=num_workers)
        assert not normalize_data, "Not implemented for the task"
        self.normalize_data = normalize_data
        self.normalize_type = normalize_type
        assert self.normalize_type in NORMALIZATION_TYPES
        if self.normalize_data:
            statistics_folder = os.path.join(self.precomputed_folder, "statistics")
            self.mean = np.load(os.path.join(statistics_folder, MEAN_NAME), allow_pickle=True)
            self.var = np.load(os.path.join(statistics_folder, VAR_NAME), allow_pickle=True)
            self.min = np.load(os.path.join(statistics_folder, MIN_NAME), allow_pickle=True)
            self.max = np.load(os.path.join(statistics_folder, MAX_NAME), allow_pickle=True)
            self.mean_tensor = torch.tensor(self.mean)
            self.var_tensor = torch.tensor(self.var)
            self.min_tensor = torch.tensor(self.min)
            self.max_tensor = torch.tensor(self.max)

    def _prepare_data(self, num_workers=8):
        # to be implemented in subclass, reading and processing of data
        raise NotImplementedError()

    def get_classifier(self, device):
        # to be implemented in subclass, in case classifier for fid has been trained
        raise NotImplementedError()

    def __len__(self):
        return len(self.segments) // self.stride
    
    def _load_mean_motion(self):
        clas2meanfreq = {}
        if not self.drop_root:
            mm_file_per_task = {'motpred': "mean_motion_test.txt", "pose": "mean_motion_pose_test.txt", "traj": "mean_motion_traj_test.txt"}
        else: 
            mm_file_per_task = {"hmp": "mean_motion_test.txt",}
        for task, mm_file_name in mm_file_per_task.items():
            # released reference file next to the test segments (datasets/annotations/...), else the processed data folder
            shipped = os.path.join(os.path.dirname(self.segments_path), mm_file_name) if getattr(self, "segments_path", None) else None
            motion_avg_path = shipped if shipped and os.path.isfile(shipped) else os.path.join(self.precomputed_folder, mm_file_name)
            if not os.path.isfile(motion_avg_path):
                from .math_utils import compute_mean_motions
                print("Computing mean motions...")
                class_average_3d, list_of_motions_3d, frequencies = compute_mean_motions(self)
                with open(motion_avg_path, 'w') as filehandle:
                    filehandle.write('\n'.join([f"{c},{meanmot},{freq}" for c, meanmot, freq in zip(list(class_average_3d.keys()), list_of_motions_3d, frequencies)]))
                print("Saved to ", motion_avg_path)
            with open(motion_avg_path) as f:
                # action name, mean motion, freq
                for line in f:
                    c, meanmot, freq = line.strip().split(',')
                    clas2meanfreq[c] = [meanmot, freq]
            meanmot, freq = zip(*[[float(clas2meanfreq[c][0]),float(clas2meanfreq[c][1])]  for i,c in enumerate(self.idx_to_class)])  
            if not self.drop_root:
                setattr(self, f'mean_motion_per_class_{task}' if task!="motpred" else f'mean_motion_per_class', meanmot) 
            else:
                setattr(self, f'mean_motion_per_class', meanmot)

    def __getitem__(self, sample_idx):
        segment_idx = int(self.stride * sample_idx + self.augmentation)
        if self.augmentation != 0:
            offset = np.random.randint(-self.augmentation, self.augmentation + 1)
            segment_idx = max(0, min(segment_idx + offset, len(self.segments) - 1))
            (i, init, end) = self.segments[segment_idx]
        else:
            (i, init, end) = self.segments[segment_idx]

        obs, pred = self._get_segment(i, init, end)

        # we include in the observation the zero, first (second) orders
        if self.normalize_data:
            obs = self.normalize(obs)
            pred = self.normalize(pred)
        # the identificative idx must be the sample_idx. This is helpful for generating samplers
        return obs, pred, {
            "sample_idx": sample_idx,
            "clip_idx": i, # index of self.annotations, i.e. of whole sequence
            "init": init,
            "end": end,
            "metadata": self.segment_idx_to_metadata[segment_idx],
            "segment_idx": segment_idx
        }
    
    def unique_sample_string(self, extra):
        def _unique_sample_string(extra):
            return f"{extra['metadata'][0]}-{extra['metadata'][1]}_clip{extra['clip_idx']}_{extra['init']}:{extra['end']}-segment{extra['segment_idx']}"
        if torch.is_tensor(extra['clip_idx']):
            return [_unique_sample_string({k: (extra[k][i] if k!= 'metadata' else [extra[k][0][i], extra[k][1][i]]) for k in extra if 'non_rotated' not in k }) for i in range(len(extra['clip_idx']))]
        else: 
            return _unique_sample_string(extra)
    
    def load_mmgt(self, path):
        # load json from file at 'path'
        with open(path, 'r') as filehandle:
            self.mm_indces = ast.literal_eval(json.load(filehandle))
        self.mm_indces = {k: sorted(self.mm_indces[k]) for k in sorted(self.mm_indces)}
        print(f"Multimodal GT loaded from '{path}'")

    def find_segment(self, clip_idx, init, end=None):
        # find the segment index given the clip_idx, init and end
        for i, (i_, init_, end_) in enumerate(self.segments):
            if i_ == clip_idx and init_ == init and (end is None or end_ == end):
                return i
        return None

    def find_sample(self, clip_idx, init, end=None):
        # find the sample index given the clip_idx, init and end
        assert self.augmentation == 0, "Cannot find sample if augmentation is not 0"
        sample_idx = self.find_segment(clip_idx, init, end) / self.stride
        return int(sample_idx)


    def _get_segment(self, i, init, end):
        # i corresponds to the segment idx inside self.annotations
        # EX: is 2 segments, first seg for third session will be in the third position and second in the fourth position. 
        # Then, the fourth session will be in the fifth position
        assert init >= 0, "init point for segment must be > 0"

        data = self.annotations[i][init:end + 1]
        
        obs, pred = data[:self.obs_length], data[self.obs_length:]
        assert len(obs) == self.obs_length and len(pred) == self.pred_length

        return obs, pred

    def _get_mmgt_for_segment(self, segment_idx):
        # we need to find the closest (may not be computed for all indices)
        mm_gt_idces = self.mm_indces[segment_idx]
        # we get all prediction multimodal GT segments
        mm_gts = [self._get_segment(*self.segments[idx])[1] for idx in mm_gt_idces]
        mm_gt = np.stack(mm_gts, axis=0)
        return mm_gt
        
        
    def _generate_segments(self):
        # this strategy is followed to ensure each epoch visits all samples from the dataset.
        assert self.clip_idx_to_metadata is not None, "idx_to_metadata must be initialized before generating segments"
        both = [((idx, init, init + self.seg_length - 1), self.clip_idx_to_metadata[idx]) for idx in range(len(self.annotations)) 
                                                        for init in range(0, self.annotations[idx].shape[0] - self.seg_length)]
        
        # unzip to separate the segments from the metadata
        segments, segment_idx_to_metadata = list(zip(*both))
        assert -1 not in [init for idx,init,ed in segments]
        return segments, segment_idx_to_metadata

    def _generate_statistics_full(self, anns_list):
        # basic statistics computation for which all segments for all participants loaded are concatenated and computed one mean and one variance per landmark
        # more complex statistics computations can be done by overriding this function (normalization per participant, for instance)
        statistics_folder = os.path.join(self.precomputed_folder, "statistics")
        if not os.path.exists(statistics_folder):
            os.makedirs(statistics_folder)
        mean_file = os.path.join(statistics_folder, MEAN_NAME)
        var_file = os.path.join(statistics_folder, VAR_NAME)
        min_file = os.path.join(statistics_folder, MIN_NAME)
        max_file = os.path.join(statistics_folder, MAX_NAME)
        if np.array([os.path.exists(p) for p in [mean_file, var_file, min_file, max_file]]).all():
            if not self.silent:
                print("Skipping statistics generation...")
            return

        all_concatenated = np.concatenate(anns_list, axis=0)
        _, N_landmarks, N_dims = all_concatenated.shape
        ps = all_concatenated.reshape((-1, N_landmarks, N_dims)) # (NumParts * SegmentsEach, NumLandmarks, NumDims) generalized to N people
        
        np.mean(ps, axis=0).dump(mean_file)
        np.var(ps, axis=0).dump(var_file)
        np.min(ps, axis=0).dump(min_file)
        np.max(ps, axis=0).dump(max_file)


    def normalize(self, x):
        # x := [seq_length, num_people, num_landmarks, num_dims]
        if self.normalize_type == "standardize":
            return (x - self.mean) / np.sqrt(self.var)
        elif self.normalize_type == "normalize":
            return 2 * (x - self.min) / (self.max - self.min) - 1
        raise NotImplementedError(f"'{self.normalize_type}' normalization not implemented.")

    def _get_hash_str(self):
        # this function is used to generate a unique hash for the dataset, based on the parameters used to generate it
        raise NotImplementedError()