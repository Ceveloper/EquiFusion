import numpy as np
import os
import torch
import hashlib
from scipy.spatial.transform import Rotation as R
from torch.utils.data._utils.collate import default_collate

from .base_dataset import BaseDataset
from ...skeleton.motion.utils import center_kpts_around_hip


def custom_collate_for_mmgt(data):
    have_mmgt = "mm_gt" in data[0][2]
    have_limbseq_eval = "limbseq_eval" in data[0][2]
    have_angles_idx = "limb_angles_idx" in data[0][2]
    if have_mmgt:
        mm_gt = [d[2].pop("mm_gt") for d in data]
    if have_angles_idx:
        limb_angles_idx = [d[2].pop("limb_angles_idx") for d in data]
    if have_limbseq_eval:
        limbseq_eval = [d[2].pop("limbseq_eval") for d in data]
    obs, pred, extra = default_collate(data)
    if have_mmgt:
        extra["mm_gt"] = mm_gt
    if have_angles_idx:
        extra["limb_angles_idx"] = limb_angles_idx
    if have_limbseq_eval:
        extra["limbseq_eval"] = limbseq_eval
    return obs, pred, extra


class MotionDataset(BaseDataset):
    def __init__(
        self,
        split,
        precomputed_folder,
        skeleton,
        obs_length: int,
        pred_length: int,
        segments_path=None,
        stride=1,
        augmentation=0,
        da_mirroring=0.0,
        da_rotations=0.0,
        dtype="float32",
        if_permute_keypoints: bool = False,
        permutation_prob: float = 1.0,
        if_root_perm: bool = True,
        if_consider_hip=False,
        if_load_mmgt=False,
        if_forward_perm_to_graph: bool = False,
        if_train_interpfps: bool = False,
        silent=False,
        **kwargs,
    ):
        """
        Motion segments of one dataset, returned in the input space of the skeleton's motion representation.

        Training-time augmentation: random mirroring (x/y) and rotation around z, and a random
        node permutation of the keypoints and of the adjacency matrix (with probability `permutation_prob`).
        """
        self.precomputed_folder = precomputed_folder
        self.segments_path = segments_path
        self.split = split
        self.skeleton = skeleton
        self.if_load_mmgt = if_load_mmgt
        self.if_permute_keypoints = if_permute_keypoints
        self.permutation_prob = permutation_prob
        self.if_root_perm = if_root_perm
        self.if_forward_perm_to_graph = if_forward_perm_to_graph
        # Resample a dataset recorded at `fps` (e.g. H36M, 50 fps) to the 60-fps windows the models use:
        # load 0.5 s of observation and 2 s of prediction at the native fps, then index 30 / 120 frames.
        self.if_train_interpfps = if_train_interpfps
        if self.if_train_interpfps:
            obs_length = int(kwargs["fps"] / 2)
            pred_length = int(kwargs["fps"] * 2)

        if not silent:
            print(f"Constructing {type(self).__name__} for split ", self.split)
        assert self.split in ["valid", "train", "test"], f"{self.split} not implemented"
        if self.split != "test":
            assert self.segments_path is None or self.split in self.segments_path
        assert 0.0 <= da_mirroring <= 1.0 and 0.0 <= da_rotations <= 1.0, "Data augmentation strategies must be in [0, 1]"
        self.da_mirroring = da_mirroring
        self.da_rotations = da_rotations

        self.in_eval = split in ["test", "valid"]

        super().__init__(
            precomputed_folder,
            obs_length,
            pred_length,
            augmentation=augmentation,
            stride=stride,
            dtype=dtype,
            if_consider_hip=if_consider_hip,
            silent=silent,
        )
        self.load_mmgt()
        if self.split in ["test"] and kwargs.get("if_compute_cmd", False):
            self._load_mean_motion()
        if self.if_train_interpfps:
            self.mask_fps_pred = torch.from_numpy(np.floor(np.arange(0, kwargs["fps"] * 2, kwargs["fps"] / 60))).to(int)
            self.mask_fps_obs = torch.from_numpy(np.floor(np.arange(0, kwargs["fps"] / 2, kwargs["fps"] / 60))).to(int)
        if not silent:
            print(f"Constructed {type(self).__name__} for split {self.split} with a total of {len(self.segments)} samples")

    def extract_action_label(self, extra):
        return extra["metadata"][self.metadata_class_idx]

    def eval(self):
        self.in_eval = True

    def train(self):
        self.in_eval = False

    def load_mmgt(self, path=None):
        if self.if_load_mmgt:
            if self.if_consider_hip:
                suffix = "_hmp"
            else:
                suffix = ""
            if path is None:
                mmgt_path = os.path.join(
                    self.precomputed_folder, f"mmgt_{self.split}{suffix}.txt"
                )
            else:
                mmgt_path = path.replace(".txt", f"{suffix}.txt")
            assert os.path.exists(
                mmgt_path
            ), "Multimodal GT file does not exist. Please compute it first."
            super().load_mmgt(mmgt_path)
            assert len(self.mm_indces) == len(
                self.segments
            ), "Multimodal GT file does not have the same number of samples as the dataset"

    def get_classifier(self, device):
        raise NotImplementedError(
            f"We don't have a classifier for the {type(self).__name__} dataset"
        )

    def _get_hash_str(self, use_all=False):
        use_all = (
            [
                str(self.obs_length),
                str(self.pred_length),
                str(self.stride),
                str(self.augmentation),
            ]
            if use_all
            else []
        )
        to_hash = "".join(
            tuple(
                self.subjects
                + list(self.actions)
                + [str(self.drop_root), str(self.use_vel)]
                + use_all
            )
        )
        return str(hashlib.sha256(str(to_hash).encode("utf-8")).hexdigest())


    def _get_mmgt_idx_(self, extra):
        if self.mm_indces is not None and self.if_load_mmgt:
            mm_gt = self._get_mmgt_for_segment(extra["segment_idx"])
            if self.normalize_data:
                mm_gt = self.normalize(mm_gt)
            extra["mm_gt"] = mm_gt
        else:
            ...  # extra["mm_gt"] = -1
        return extra

    def data_augmentation(self, obs, pred, extra):
        if self.if_load_mmgt:
            mm_gt = extra["mm_gt"]

        if self.da_mirroring != 0:
            # apply mirroring with probability 0.5
            mirroring_idces = [
                0,
                1,
            ]  # 2 is not used because the person would be upside down
            for m in mirroring_idces:
                if np.random.rand() < self.da_mirroring:
                    # make a copy of obs, pred
                    obs, pred = obs.copy(), pred.copy()
                    # invert sign of first coordinate at last axis for obs, pred
                    obs[..., m] *= -1
                    pred[..., m] *= -1

                    if self.if_load_mmgt:
                        mm_gt = mm_gt.copy()
                        mm_gt[..., m] *= -1

        if self.da_rotations != 0:
            # apply random rotations with probability 1
            rotation_axes = [
                "z"
            ]  # 'x' and 'y' not used because the person could be upside down
            for a in rotation_axes:
                if np.random.rand() < self.da_rotations:
                    degrees = np.random.randint(0, 360)
                    r = (
                        R.from_euler(a, degrees, degrees=True)
                        .as_matrix()
                        .astype(np.float32)
                    )
                    obs = (r @ obs.reshape((-1, 3)).T).T.reshape(obs.shape)
                    pred = (r @ pred.reshape((-1, 3)).T).T.reshape(pred.shape)

                    if self.if_load_mmgt:
                        mm_gt = (r @ mm_gt.reshape((-1, 3)).T).T.reshape(mm_gt.shape)

        if self.if_load_mmgt:
            extra["mm_gt"] = mm_gt
        return obs, pred, extra

    ##############################################################################################
    ### Start: Permutation of Keypoints
    ##############################################################################################

    def sample_permutation_params(self, J, extra):
        if self.if_root_perm:
            perm_idxs = np.random.permutation(J)
        else:
            perm_idxs = np.random.permutation(J - 1) + 1
            perm_idxs = np.concatenate([[0], perm_idxs])  # root joint is always first
        perm = torch.zeros(J, J)
        perm[range(len(perm_idxs)), perm_idxs] = 1
        return perm, perm_idxs

    def apply_permutation_to_adjacency(self, perm, extra):
        # Random graph permutation/augmentation: keypoints are permuted with P, the adjacency with P^T A P, so the
        # model also sees graphs that do not match the motion. Mainly, this prevents spurious correlations in the
        # diffusion training: with only two training graphs (AMASS and Nymeria), the denoiser could otherwise tie the
        # latent statistics to these specific graphs. It also keeps the model from relying on the graphs of the
        # training skeletons, which improves generalization to unseen kinematic trees
        # (A+N, zero-shot H36M ADE: 0.395 with permutation_prob=0.5, 0.460 with 0.0).
        if self.if_forward_perm_to_graph:
            extra["perm"] = perm
        extra["adj"] = perm.mT @ extra["adj"] @ perm
        return extra

    def apply_permutation_to_keypoints(self, obs, pred, extra, perm_idxs):
        obs = obs[..., :, perm_idxs, :]
        pred = pred[..., :, perm_idxs, :]

        if self.if_load_mmgt:
            extra["mm_gt"] = extra["mm_gt"][..., perm_idxs, :]
        return obs, pred, extra

    def apply_permutation(self, perm, perm_idxs, obs, pred, extra):
        extra = self.apply_permutation_to_adjacency(perm, extra)
        obs, pred, extra = self.apply_permutation_to_keypoints(
            obs, pred, extra, perm_idxs
        )
        return obs, pred, extra

    def permute_keypoints(self, obs, *args, extra: torch.Tensor):
        if self.if_permute_keypoints and not self.in_eval:
            if np.random.rand() < self.permutation_prob:
                *B, T, J, _ = obs.shape
                perm, perm_idxs = self.sample_permutation_params(J, extra)
                args = self.apply_permutation(perm, perm_idxs, obs, *args, extra=extra)
                args[-1]["perm_idxs"] = perm_idxs
                return args
        return (obs, *args, extra)

    ##############################################################################################
    ### End: Permutation of Keypoints
    ##############################################################################################


    def get_segment_from_dataset(self, idx):
        obs, pred, extra = super().__getitem__(idx)
        extra["adj"] = self.skeleton.adj_matrix
        return obs, pred, extra

    def pad_sequence_fps(self, obs, pred, extra):
        if self.if_train_interpfps:
            pred = pred[..., self.mask_fps_pred, :, :]
            obs = obs[..., self.mask_fps_obs, :, :]
            if "mm_gt" in extra:
                extra["mm_gt"] = extra["mm_gt"][..., self.mask_fps_pred, :, :]
        return obs, pred, extra

    def tranform2inputspace(self, obs, pred, extra):
        data = self.skeleton.tranform_to_input_space(
            torch.cat([torch.from_numpy(obs), torch.from_numpy(pred)], dim=-3)
        )
        if self.skeleton.node_type == "bone":
            extra["adj"] = self.skeleton.bone_adj_matrix()
        obs, pred = data[..., : obs.shape[-3], :, :], data[..., obs.shape[-3] :, :, :]
        if self.if_load_mmgt:
            if self.skeleton.node_type == "bone":
                centered, hips = center_kpts_around_hip(torch.from_numpy(extra["mm_gt"]), hip_idx=0)
                extra["mm_gt"] = centered[..., 1:, :]
            else:
                extra["mm_gt"] = self.skeleton.tranform_to_input_space(torch.from_numpy(extra["mm_gt"]))
        return obs, pred, extra

    def _adapt_extra(self, extra, x):
        extra.pop("perm_idxs", None)
        return extra

    def __getitem__(self, idx):
        obs, pred, extra = self.get_segment_from_dataset(idx)
        extra = self._get_mmgt_idx_(extra)
        obs, pred, extra = self.pad_sequence_fps(obs, pred, extra)
        if not self.in_eval:
            obs, pred, extra = self.data_augmentation(obs, pred, extra)
        obs, pred, extra = self.tranform2inputspace(obs, pred, extra)
        obs, pred, extra = self.permute_keypoints(obs, pred, extra=extra)
        extra = self._adapt_extra(extra, obs)
        return obs, pred, extra
