from matplotlib.pyplot import bone, cla
import torch
import numpy as np

from .base import Kinematic


class NymeriaKinematic(Kinematic):
    def __init__(self, num_joints=23, **kwargs):

        super().__init__(**kwargs)

        nymeria_kinematic_tree = [
            [1, 2, 3, 4, 5, 6],  # body
            [11, 12, 13, 14],  # right arm
            [7, 8, 9, 10],  # left arm
            [1, 19, 20, 21, 22],  # right leg
            [1, 15, 16, 17, 18],  # left leg
        ]

        self.joint_dict_orig = {
            0: "GlobalRoot",
            1: "Spine1",
            2: "Spine2",
            3: "Spine3",
            4: "BMN",
            5: "Neck",
            6: "Head",
            7: "LSI",
            8: "LShoulder",
            9: "LElbow",
            10: "LWrist",
            11: "RSI",
            12: "RShoulder",
            13: "RElbow",
            14: "RWrist",
            15: "LHip",
            16: "LKnee",
            17: "LHeel",
            18: "LFoot",
            19: "RHip",
            20: "RKnee",
            21: "RHeel",
            22: "RFoot",
        }

        self.limbseq = [
            [0, 1],  # global to hip
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],  # body
            [4, 11],
            [11, 12],
            [12, 13],
            [13, 14],  # right arm
            [4, 7],
            [7, 8],
            [8, 9],
            [9, 10],  # left arm
            [0, 19],
            [19, 20],
            [20, 21],
            [21, 22],  # right leg
            [0, 15],
            [15, 16],
            [16, 17],
            [17, 18],  # left leg
        ]
        assert num_joints in [23]

        self.left_right_limb_list = [
            (
                False
                if (joint[0] == "L" and joint[1].isupper()) or "left" in joint
                else True
            )
            for joint in list(self.joint_dict_orig.values())
        ]

        if not self.if_consider_hip:
            self.node_dict = self.joint_dict_orig.copy()
            self.node_dict.pop(0)
            self.node_dict = {i: v for i, v in enumerate(list(self.node_dict.values()))}
            node_dict_reversed = {v: i for i, v in self.node_dict.items()}
            self.node_limbseq = [
                sorted([node_dict_reversed["RHip"], node_dict_reversed["LHip"]]),
                sorted([node_dict_reversed["RHip"], node_dict_reversed["Spine1"]]),
                sorted([node_dict_reversed["LHip"], node_dict_reversed["Spine1"]]),
                *[
                    [limb[0] - 1, limb[1] - 1]
                    for limb in self.limbseq
                    if limb[1] != 0 and limb[0] != 0
                ],
            ]
            self.limb_angles_idx = [
                [3, 4, 5, 6, 7],
                [5, 8, 9, 10, 11],
                [5, 12, 13, 14, 15],
                [1, 16],
                [2, 0, 16, 17, 18],
                [2, 19, 20, 21],
            ]
        else:
            self.node_dict = {
                k: v
                for k, v in enumerate(
                    list(self.node_hip.values())
                    + list(self.joint_dict_orig.values())[1:]
                )
            }
            self.node_limbseq = self.limbseq.copy()

        self.bones = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [4, 11],
            [11, 12],
            [12, 13],
            [13, 14],
            [4, 7],
            [7, 8],
            [8, 9],
            [9, 10],
            [0, 19],
            [19, 20],
            [20, 21],
            [21, 22],
            [0, 15],
            [15, 16],
            [16, 17],
            [17, 18],
        ]
        self.bone_parent = [
            0,
            1,
            2,
            3,
            4,
            5,
            4,
            11,
            12,
            13,
            4,
            7,
            8,
            9,
            0,
            19,
            20,
            21,
            0,
            15,
            16,
            17,
        ]
        self.bone_child = [
            1,
            2,
            3,
            4,
            5,
            6,
            11,
            12,
            13,
            14,
            7,
            8,
            9,
            10,
            19,
            20,
            21,
            22,
            15,
            16,
            17,
            18,
        ]
        self.bone_names = [
            "Spine1",
            "Spine2",
            "Spine3",
            "Chest",
            "Neck",
            "Head",
            "RCollarbone",
            "RShoulder",
            "RUpperArm",
            "RLowerArm",
            "LCollarbone",
            "LShoulder",
            "LUpperArm",
            "LLowerArm",
            "RHip",
            "RThigh",
            "RCalf",
            "RFoot",
            "LHip",
            "LThigh",
            "LCalf",
            "LFoot",
        ]
        self.bone_symmetry = np.array(
            [
                [6, 10],
                [7, 11],
                [8, 12],
                [9, 13],
                [14, 18],
                [15, 19],
                [16, 20],
                [17, 21],
            ]
        )
        self.bone_left = [10, 11, 12, 13, 18, 19, 20, 21]
        self.bone_right = [6, 7, 8, 9, 14, 15, 16, 17]
        self.limb_bones = {
            "head": [4, 5],
            "right_arm": [8, 9],
            "left_arm": [12, 13],
            "right_leg": [16, 17],
            "left_leg": [20, 21],
        }
        # Following stats are computed with train + validation subjects, by
        # 1) obtaining bones lengths for each of the total 1013 sequences
        # 2) computing mean over the 1013 sequences (14 unique bone types)
        # 3) as often single sequences have std > 1.e-4, we compute the overalll std as  E[Var] + Var[E], so the mean of the var of each sequence + the var of the sequence means
        self.bone_mean = [
            0.115202,
            0.083127,
            0.076074,
            0.075989,
            0.101834,
            0.089286,
            0.061616,
            0.12827,
            0.271222,
            0.251174,
            0.061616,
            0.12827,
            0.271222,
            0.251174,
            0.098863,
            0.513135,
            0.386203,
            0.182497,
            0.098863,
            0.513135,
            0.386203,
            0.182497,
        ]
        self.bone_std = [
            0.008991,
            0.010129,
            0.009202,
            0.009192,
            0.013387,
            0.008053,
            0.14346,
            0.140439,
            0.126437,
            0.134807,
            0.14346,
            0.140439,
            0.126437,
            0.134807,
            0.13175,
            0.14498,
            0.146577,
            0.133449,
            0.13175,
            0.14498,
            0.146577,
            0.133449,
        ]
        self.limbseq = np.array(self.limbseq)
        self.init(**kwargs)
