import torch

from ..utils import center_kpts_around_hip
from .center import SkeletonCenterPose

class SkeletonDirectionsPose(SkeletonCenterPose):
    """
    Poses are expressed in terms of directions, with unit norm
    Abstract class
    """
    node_type = "bone"
    #################  functions to obtain input poses ########################################################################################
    
    def tranform_to_input_space_pose_only(self, data):
        #center kpts
        centered, hips = center_kpts_around_hip(data, hip_idx=0)
        limbs = self.extract_limbs(centered, mode='bone')
        kpts = torch.cat([hips, limbs], dim=-2)
        return kpts
    
    ##########################  function to tranform back to metric space ##########################################################################
        
    def transform_to_metric_space(self, data, bone_lengths, limbseq=None):
        # we start from a vectior B, T, L, 3 where L is the number of limbs
        limb_dirs = data/torch.linalg.norm(data, dim=-1).unsqueeze(-1)
        pose = -1*self.compute_pose_from_directions_and_bone_lengths(limb_dirs, bone_lengths, limbseq=limbseq)
        if not self.if_consider_hip:
            kpts = pose[..., 1:, :]
            # data[..., :1, :] = 0.
        else: 
            kpts = self.transform_hip_to_input_space(pose)
        return kpts
