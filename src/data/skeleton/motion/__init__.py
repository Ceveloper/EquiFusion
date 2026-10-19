

from .base import Skeleton,Skeleton as SkeletonVanillaAlignSeqStart, Skeleton as SkeletonVanilla
from .pose import SkeletonCenterPose, SkeletonRescalePose, SkeletonDirectionsPose
from .utils import center_kpts_around_hip_and_drop_root


def get_motion_representation_objclass(motion_repr_type):
    return globals()[motion_repr_type]