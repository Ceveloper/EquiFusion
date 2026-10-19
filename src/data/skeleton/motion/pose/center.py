import torch

from ..utils import center_kpts_around_hip
from ..base import Skeleton

class SkeletonCenterPose(Skeleton):
    """
    Skeleton class for vanilla skekelton (no particulara data reresentation, no normalization, no direction)
    Does NOT center the pose around the hip. Leaves everythign as it is.
    Bsically instantiable version of basic abstract class Skeleton without centering.
    if we consider the hip, nothing is done but just the sequence start is shifted.
    """
    def tranform_to_input_space_pose_only(self, data):
        #center kpts
        centered, hips = center_kpts_around_hip(data, hip_idx=0)

        kpts = torch.cat([hips, centered[...,len(self.node_hip):,:]], dim=-2)
        return kpts
    
    def _merge_hip_and_poseinmetricspace(self, hip_coords, kpts):
        kpts += hip_coords
        return super()._merge_hip_and_poseinmetricspace(hip_coords, kpts)
    