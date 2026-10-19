import torch

from ..utils import center_kpts_around_hip
from .center import SkeletonCenterPose

class SkeletonRescalePose(SkeletonCenterPose):
    """
    Poses are not only centered like in Skeleton, but also scaled in a unit cube.
    Abstract class
    """
    def __init__(self, pose_box_size=1.1,  **kwargs):
        super().__init__(**kwargs)
        self.pose_box_size = pose_box_size # in m. not in mm

    #################  functions to obtain input poses ########################################################################################
    
    def tranform_to_input_space_pose_only(self, data):
        #center kpts
        centered, hips = center_kpts_around_hip(data, hip_idx=0)
        
        lower_box_shift = self.pose_box_size # in mm
        centered /= lower_box_shift
        kpts = torch.cat([hips, centered[...,1:,:]], dim=-2)
        return kpts
    
    ##########################  function to tranform back to metric space ##########################################################################
        
    def transform_to_metric_space_pose_only(self, kpts):  
            kpts = self._rescale_to_hip_box(kpts)
            return kpts
    
    def _rescale_to_hip_box(self, kpts):
        kpts = kpts.clone()
        
        lower_box_shift = self.pose_box_size # in mm
        # range [-1,1] around hips
        kpts *= lower_box_shift # scaled back to box around kpts
        return kpts
    
