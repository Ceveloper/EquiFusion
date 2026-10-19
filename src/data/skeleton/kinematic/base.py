import itertools
import torch
import numpy as np
from .utils import plot_matrix, get_adj_matrix


class Kinematic:
    """
    It is a abstract class, needs to be subclassed.
    """

    def __init__(self, if_consider_hip=False, **kwargs):
        self.if_consider_hip = if_consider_hip


        # TO DO define following variables

    

    def init(
        self,
        drop_limb_ids=[1],
        if_eval_generate_dropped_limbs=False,
        if_eval_drop_limb=False,
        drop_bones=None,
        **kwargs,
    ):
        self.drop_bones = drop_bones  # explicit bone indices to drop (bone direction models), instead of drop_limb_ids
        self.if_eval_generate_dropped_limbs = if_eval_generate_dropped_limbs
        self.if_eval_drop_limb = if_eval_drop_limb
        self.num_nozero_joints = self.num_joints - int(not self.if_consider_hip)
        self.num_nozero_nodes = self.num_nodes
        if self.if_eval_drop_limb:
            self.idx_to_limb = {
                0: "head",
                1: "left_arm",
                2: "right_arm",
                3: "left_leg",
                4: "right_leg",
                5: "right_arm_single",
            }
            # missing limbs are dropped from the graph for bone direction models only; the 3D joint models take
            # missing joints as NaN in the observation (see src/inference_utils.py)
            assert "Direction" in self.__class__.__name__, "Dropping limbs is implemented for bone direction models only"
            # compute the new skeleton (joints, limbseq) without the dropped limbs; the node indices of the
            # dropped bones are stored in self.drop_joints_ids
            limbseq_new, limb_angles_idx, sym_limb_idxs = self._create_new_skel_drop_limb(drop_limb_ids)
            self.limbseq = np.array(limbseq_new)
            if not self.if_eval_generate_dropped_limbs:
                drop_bone_ids = self.drop_joints_ids
                self.num_nozero_nodes = self.num_nodes - len(drop_bone_ids)
                self.num_nozero_joints = self.num_nozero_joints - len(drop_bone_ids)
                self.limb_angles_idx = limb_angles_idx
            else:
                self.sym_limb_idxs = sym_limb_idxs  # bones mirroring the dropped ones (left <-> right)

    @staticmethod
    def adj_after_dropping_nodes(adj: torch.Tensor, drop_idxs, keep_size: bool = False) -> torch.Tensor:
        """Adjacency matrix after removing the nodes `drop_idxs`: the neighbours of a removed node are connected
        to each other. Unless `keep_size`, the remaining nodes are moved to the front (zeros at the end)."""
        adj = adj.clone()
        ndrop = len(drop_idxs)
        for i in drop_idxs:
            neighbours = np.argwhere(adj[i] == 1)[0]
            if len(neighbours) >= 2:
                for a, b in itertools.combinations(neighbours, 2):
                    adj[a, b] = 1
                    adj[b, a] = 1
            adj[i] = 0
            adj[:, i] = 0
        if keep_size:
            return adj
        adj_new = torch.zeros_like(adj)
        mask = adj.sum(-1) != 0
        adj_new[..., : adj.shape[0] - ndrop, : adj.shape[0] - ndrop] = adj[mask, :][:, mask]
        return adj_new

    def _create_new_skel_drop_limb(
            self,
            drop_bone_ids=[1],
            **kwargs,
        ):
        drop_limb_ids = []
        for i in drop_bone_ids:
            drop_limb_ids += self.limb_bones[
                self.idx_to_limb[i]] 
        if self.drop_bones is not None:
            drop_limb_ids = list(self.drop_bones)
        drop_joints_ids_orig = np.unique(
            np.array([self.bones[i] for i in drop_limb_ids]).flatten()
        ).tolist()
        # exactly the dropped bones (a bone whose two joints are both in dropped limbs is not necessarily dropped,
        # e.g. H36M neck -> shoulder when the head and an arm are dropped)
        dropped_bones = [list(self.bones[i]) for i in drop_limb_ids]
        limbseq_toremove_idx = [i for i, ls in enumerate(self.limbseq) if list(ls) in dropped_bones]
        joints_still_involved_in_other_bones = np.unique(
            [
                j
                for i, ls in enumerate(self.limbseq.tolist())
                for j in ls
                if i not in limbseq_toremove_idx
            ]
        ).tolist()
        connection_joint = set.intersection(set(drop_joints_ids_orig), set(joints_still_involved_in_other_bones)) # set(joints_in_limbs_to_remove) - set(drop_joints_ids_orig)

        drop_joints_ids_orig = [
            j
            for j in drop_joints_ids_orig
            if j not in joints_still_involved_in_other_bones
        ]
        self.drop_joints_ids_orig = drop_joints_ids_orig
        self.drop_joints_ids = drop_limb_ids

        
        if self.if_eval_drop_limb and self.if_eval_generate_dropped_limbs:
            sym_limb_idxs_new = []
            if self.drop_bones is not None:  # mirrored bone of each dropped bone (left <-> right)
                mirror = {int(a): int(b) for a, b in self.bone_symmetry} | {int(b): int(a) for a, b in self.bone_symmetry}
                drop_bone_ids = []
                sym_limb_idxs_new = [mirror[b] for b in self.drop_bones]
            for bone_idx in drop_bone_ids:
                bone_tup = self.limb_bones[self.idx_to_limb[bone_idx]]
                bone_name = {tuple(v):k for k, v in self.limb_bones.items()}[tuple(bone_tup)]
                symm_limb_name = bone_name.replace('left', 'temp').replace('right', 'left').replace('temp', 'right')
                sym_limb_idxs = self.limb_bones[symm_limb_name]
                sym_limb_idxs_new.extend(sym_limb_idxs) #[map_old2new[i] for i in sym_limb_idxs]

            limb_angles_idx = None
            limbseq_new = self.limbseq.tolist()
        else:


            map_old2new = {
                k: v
                for v, k in enumerate(
                    sorted(
                        set(range(len(self.limbseq)))
                        - set(limbseq_toremove_idx)
                    )
                )
            }

            map_old2new_joint = {
                k: v
                for v, k in enumerate(
                    sorted(
                        set(range(len(self.joint_dict_orig)))
                        - set(drop_joints_ids_orig)
                    )
                )
            }
            # ]
            # ]
            # ]

            # to compute new limbs
            self.map_old2new_joint = map_old2new_joint
            limbseq_new = [
                [map_old2new_joint[limb[0]], map_old2new_joint[limb[1]]]
                for i, limb in enumerate(self.limbseq.tolist())
                if i not in limbseq_toremove_idx
            ]

            limb_angles_idx = [
                            [
                                map_old2new[seg]
                                for seg in chain
                                if seg not in limbseq_toremove_idx
                            ]
                            for chain in self.limb_angles_idx
                        ]  # this must be updated if we remove joints and not limbs
            sym_limb_idxs_new = None
        return limbseq_new, limb_angles_idx, sym_limb_idxs_new

    def drop_limbs(self, kpts):
        kpts = self.set_droplimbs_tozero(kpts)
        return kpts

    def set_droplimbs_tozero(self, kpts):
        # if "Direction" not in self.__class__.__name__ or (
        # ):
        kpts[..., self.drop_joints_ids, :] = 0.0
        if not self.if_eval_generate_dropped_limbs:
            kpts = self._shift_zerojoints(kpts, self.drop_joints_ids)
        # move dropped limbs to the end of the vector
        # change the adjacency matrix
        return kpts

    def _shift_zerojoints(self, kpts, drop_joints_ids):
        assert self.num_nodes == kpts.shape[-2]
        num_nozero_nodes = self.num_nodes - len(drop_joints_ids)
        new_idxs = sorted(set(range(self.num_nodes)) - set(drop_joints_ids))
        kpts_new = torch.zeros_like(kpts)
        kpts_new[..., :num_nozero_nodes, :] = kpts[..., new_idxs, :]
        return kpts_new

    def out_postprocess_droplimb(self, kpts):
        if not self.if_eval_generate_dropped_limbs:
            # set output to zero
            kpts[..., self.num_nozero_joints :, :] = 0 # for our PE model this is not necessary, it is automatically zero
        return kpts

    # @property
    def parents(self, mode="original"):
        num_joints = self.num_joints if mode == "original" else self.num_nodes
        parents = [None] * num_joints
        parents[0] = -1
        limbseq = self.limbseq if mode == "original" else self.node_limbseq
        for tup in limbseq:
            assert tup[0] < tup[1]
            parents[tup[1]] = tup[0]
        return parents

    @property
    def num_joints(self):
        return len(self.joint_dict_orig)

    @property
    def num_nodes(self):
        return len(self.node_dict)

    @property
    def left_right_limb(self):
        return self.left_right_limb_list.copy()

    def bone_adj_matrix(self, mode='bone', limbseq=None):
        assert mode in ['bone', 'limbseq'], "mode must be 'bone' or 'limbseq'"
        if limbseq is not None:
            assert mode=='limbseq', "If limbseq is provided, mode must be 'limbseq'"
        limbseq = self.limbseq if limbseq is None else limbseq

        # Construct list of bones as (parent, child)
        if mode == 'bone':
            bones = [
                (self.bone_parent[i], self.bone_child[i])
                for i in range(len(self.bone_child))
            ]
        else:
            bones = [
                (limb[0], limb[1]) for limb in limbseq.tolist()
            ]

        # Find adjacency based on shared joints
        adjacent_pairs = []
        for i in range(len(bones)):
            pi, ci = bones[i]
            for j in range(i + 1, len(bones)):
                pj, cj = bones[j]
                # Two bones are adjacent if they share a joint (node)
                if pi in (pj, cj) or ci in (pj, cj):
                    adjacent_pairs.append((i, j))

        bone_adj = torch.zeros(
            (self.num_joints - 1, self.num_joints - 1), dtype=torch.float32
        )
        for limb in adjacent_pairs:
            bone_adj[limb[0], limb[1]] = 1
            bone_adj[limb[1], limb[0]] = 1
        return bone_adj

    # @property

    @property
    def adj_matrix(self):
        return get_adj_matrix(limbseq=self.node_limbseq, num_nodes=self.num_nodes)

    def get_limbseq(self):
        limbseq = self.limbseq if self.if_consider_hip else self.node_limbseq
        return limbseq

    def extract_limbs(self, kpts, mode="metric"):
        if mode == 'bone':
            limbseq = self.bones
        else: 
            limbseq = self.limbseq if mode == "metric" else self.node_limbseq
        l1s, l2s = zip(*limbseq)
        limbdist = kpts[..., l1s, :] - kpts[..., l2s, :]
        return limbdist

    def compute_pose_from_directions_and_bone_lengths(self, limb_dirs, bone_lengths, limbseq=None):
        J = self.num_joints
        *B, T, L, C = limb_dirs.shape
        if len(bone_lengths.shape) == len(limb_dirs.shape) - 1:
            assert (
                bone_lengths.shape[-2:] == (L, 1)
                and len(bone_lengths.shape) == len(limb_dirs.shape) - 1
                and bone_lengths.shape[0] == B[0]
            ), "bone_lengths should be of shape (B, n_bones)"
            bone_lengths = bone_lengths[..., None, :, :]
        else:
            assert (
                bone_lengths.shape[:-1] == limb_dirs.shape[:-1]
                and limb_dirs.shape[-1] == 3
            ), "bone_lengths should be of shape (B, T, n_bones)"
        pose = torch.zeros((*B, T, J, C), device=limb_dirs.device)
        bones = limb_dirs * bone_lengths  # (n_bones, 3)
        limbseq = self.limbseq if limbseq is None else limbseq
        bone_parents, bone_children = zip(*limbseq)
        for i, (p, c) in enumerate(zip(bone_parents, bone_children)):
            pose[..., c, :] = pose[..., p, :] + bones[..., i, :]
        return pose
