import torch
import numpy as np

from .base import Kinematic


class AMASSKinematic(Kinematic):
    def __init__(self, num_joints=22, **kwargs):
        
        super().__init__(**kwargs)
        
        self.joint_dict_orig  = {0: 'GlobalRoot', 1: 'LHip',  2: 'RHip',   3: 'Spine1',
                                    4: 'LKnee',   5: 'RKnee',   6: 'Spine3',
                                    7: 'LHeel', 8: 'RHeel',  9: 'Neck',
                                    10: 'LFoot', 11: 'RFoot',
                                    12: 'BMN',  13: 'LSI',  14: 'RSI',  15: 'Head',
                                    16: 'LShoulder', 17: 'RShoulder',
                                    18: 'LElbow', 19: 'RElbow', 20: 'LWrist', 21: 'RWrist'}

        self.limbseq = [ [0,3], [3,6], [6,9], [9,12], [12,15], # body
                        [9,14], [14,17], [17,19], [19,21], # right arm
                        [9,13], [13,16], [16,18], [18,20], # left arm
                        [0,2], [2,5], [5,8], [8,11], # right leg
                        [0,1], [1,4], [4,7], [7,10] # left leg
                    ]
        
        assert num_joints in [22, 52]
        if num_joints == 52:
            self.joint_dict_orig = {**self.joint_dict_orig, **{22: 'left_index1', 23: 'left_index2',  24: 'left_index3',
                                                                25: 'left_middle1', 26: 'left_middle2', 27: 'left_middle3',
                                                                28: 'left_pinky1', 29: 'left_pinky2', 30: 'left_pinky3',
                                                                31: 'left_ring1',  32: 'left_ring2', 33: 'left_ring3',
                                                                34: 'left_thumb1', 35: 'left_thumb2', 36: 'left_thumb3',
                                                                37: 'right_index1', 38: 'right_index2', 39: 'right_index3',
                                                                40: 'right_middle1', 41: 'right_middle2', 42: 'right_middle3',
                                                                43: 'right_pinky1', 44: 'right_pinky2', 45: 'right_pinky3',
                                                                46: 'right_ring1', 47: 'right_ring2', 48: 'right_ring3',
                                                                49: 'right_thumb1', 50: 'right_thumb2', 51: 'right_thumb3'
                                                                }}
            self.limbseq +=  [ [20, 22], [20, 25], [20, 28], [20, 31], [20,34],
                                [22,23], [23,24], [25,26], [26,27], [28,29], [29,30], [31,32], [32,33], [34,35], [35,36], # falanges left hand
                                [21,37], [21,40],[21,43], [21,46], [21,49],
                                [37,38], [38,39],  [40,41], [41,42],  [43,44], [44,45],  [46,47], [47,48],  [49,50], [50,51], # falanges right hand
                            ]     

        self.left_right_limb_list = [False if (joint[0] == "L" and joint[1].isupper()) or 'left' in joint else True for joint in list(self.joint_dict_orig.values())]

        if not self.if_consider_hip:
            self.node_dict = self.joint_dict_orig.copy()
            self.node_dict.pop(0)
            self.node_dict = {i:v for i,v in enumerate(list(self.node_dict.values()))}
            node_dict_reversed = {v:i for i,v in self.node_dict.items()}
            self.node_limbseq = [[node_dict_reversed['LHip'], node_dict_reversed['RHip']], 
                                    [node_dict_reversed['LHip'], node_dict_reversed['Spine1']], 
                                    [node_dict_reversed['RHip'], node_dict_reversed['Spine1']],
                                    *[[limb[0] -1,  limb[1] -1] for limb in self.limbseq if limb[1] != 0 and limb[0] != 0]]
            # # index of limbs in limb_angles_idx for mae
            self.limb_angles_idx = [[0,2,3,4,5,6], [0,3], [4,7,8,9,10], [4,11,12,13,14], [0,15,16,17], [18,19,20]] 
        else: 
            self.node_dict = {k:v for k,v in enumerate(list(self.node_hip.values())+list(self.joint_dict_orig.values())[1:])}
            self.node_limbseq = self.limbseq.copy()

        self.bones = [[0, 3], [3, 6], [6, 9], [9, 12], [12, 15], [9, 14], [14, 17], [17, 19], [19, 21], [9, 13], [13, 16], [16, 18], [18, 20], [0, 2], [2, 5], [5, 8], [8, 11], [0, 1], [1, 4], [4, 7], [7, 10]]
        self.bone_parent = [0, 3, 6, 9, 12, 9, 14, 17, 19, 9, 13, 16, 18, 0, 2, 5, 8, 0, 1, 4, 7]
        self.bone_child = [3, 6, 9, 12, 15, 14, 17, 19, 21, 13, 16, 18, 20, 2, 5, 8, 11, 1, 4, 7, 10]
        self.bone_names = ['Spine1', 'Spine2', 'Spine3', 'Neck', 'Head', 'RCollarbone', 'RShoulder', 'RUpperArm', 'RLowerArm', 'LCollarbone', 'LShoulder', 'LUpperArm', 'LLowerArm', 'RHip', 'RThigh', 'RCalf', 'RFoot', 'LHip', 'LThigh', 'LCalf', 'LFoot']
        self.bone_symmetry = np.array([[5, 9], [6, 10], [7, 11], [8, 12], [13, 17], [14, 18], [15, 19], [16, 20]])
        self.bone_left     = [9, 10, 11, 12, 17, 18, 19, 20]
        self.bone_right    = [5, 6, 7, 8, 13, 14, 15, 16]
        self.limb_bones = {'head': [3, 4], 'right_arm': [7, 8], 'left_arm': [11, 12], 
                                'right_leg': [15, 16], 'left_leg': [19, 20]}

        # Following stats are computed with train + validation subdatasets, by 
        # 1) obtaining bones lengths for each of the total 13674 sequences (for each sequence, bone length (bone_lengths.std(dim=0) <= 1.e-5).all())
        # 2) computing mean and std over the 13674 sequences (13 unique bone types)
        # Warning: Sequence has only one frame, skipping bone length constancy check. Sequence name: EKUT-250
        # Warning: Sequence has only one frame, skipping bone length constancy check. Sequence name: KIT-1091
        self.bone_mean =  [0.122806, 0.140562, 0.057802, 0.218633, 0.097752, 0.142898, 0.115429, 0.258568, 0.260173, 0.142898, 0.115429, 0.258568, 0.260173, 0.10915, 0.380931, 0.410674, 0.140481, 0.10915, 0.380931, 0.410674, 0.140481]
        self.bone_std = [0.01312, 0.010341, 0.003632, 0.011702, 0.012187, 0.008586, 0.016599, 0.013716, 0.0165, 0.008586, 0.016599, 0.013716, 0.0165, 0.006514, 0.019815, 0.028376, 0.009626, 0.006514, 0.019815, 0.028376, 0.009626]
        
        
        if num_joints == 52:
            # HAND BONES (LEFT HAND: 14 bones, RIGHT HAND: 14 bones)
            self.bones += [
                [20, 22], [22, 23], [23, 24],                     # left index (3)           
              [20, 25], [25, 26], [26, 27],                     # left middle (3)
              [20, 28], [28, 29], [29, 30],                     # left pinky (3)
              [20, 31], [31, 32], [32, 33],                     # left ring (3)
              [20, 34], [34, 35], [35, 36],                     # left thumb (3)
              [21, 37], [37, 38], [38, 39],                     # right index (3)
              [21, 40], [40, 41], [41, 42],                     # right middle (3)
              [21, 43], [43, 44], [44, 45],                     # right pinky (3)
              [21, 46], [46, 47], [47, 48],                     # right ring (3)
              [21, 49], [49, 50], [50, 51]]                     # right thumb (3)
            
            self.bone_parent += [20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50]
            self.bone_child += [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
            self.bone_names += ['LIndex1', 'LIndex2', 'LIndex3', 'LMiddle1', 'LMiddle2', 'LMiddle3', 'LPinky1', 'LPinky2', 'LPinky3', 
                            'LRing1', 'LRing2', 'LRing3', 'LThumb1', 'LThumb2', 'LThumb3', 
                            'RIndex1', 'RIndex2', 'RIndex3', 'RMiddle1', 'RMiddle2', 'RMiddle3', 
                            'RPinky1', 'RPinky2', 'RPinky3', 'RRing1', 'RRing2', 'RRing3', 
                            'RThumb1', 'RThumb2', 'RThumb3']
            self.bone_symmetry = np.concatenate([self.bone_symmetry, np.array([[21,36],[22,37],[23,38],[24,39],[25,40],[26,41],[27,42],[28,43],[29,44],[30,45],[31,46],[32,47],[33,48],[34,49],[35,50]])])
            self.bone_left += list(range(21, 36))  # left hand bones 21-35
            self.bone_right += list(range(36, 45)) # right hand bones 36-44
            
            self.limb_bones.update({'left_hand': list(range(21,36)), 'right_hand': list(range(36,45))})
            
            # ADD HAND BONE STATISTICS (compute these from your dataset!)
            # Values are set to None in the hope that it triggers and error if these are used.
            self.bone_mean = None
            self.bone_std = None 
        
        self.limbseq = np.array(self.limbseq)
        self.init(**kwargs)

