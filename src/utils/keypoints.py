import random
import torch
import math

def center_kpts_around_hip(kpts, hip_idx=0):
    #hip is always joint n0 in h36m regardless of number of joints considered
    center = kpts[..., hip_idx, :].unsqueeze(-2) #(n_frames, n_joints, 3)
    centered_kpts = kpts - center
    return centered_kpts, center

def center_kpts_around_hip_and_drop_root(kpts, hip_idx=0):
    centered_kpts, center = center_kpts_around_hip(kpts.clone(), hip_idx)
    centered_kpts = centered_kpts[..., 1:, :]
    return centered_kpts

def rotate_y_axis(kpts, angle_degrees, axis=1):
    phi = torch.tensor(angle_degrees * math.pi / 180)
    s = torch.sin(phi)
    c = torch.cos(phi)
    if axis == 1:
        rot = torch.tensor([[c, 0, -s],
                            [0, 1.0, 0],
                            [s, 0, c]]) # 3x3
    elif axis == 0:
        rot = torch.tensor([[1.0, 0, 0],
                            [0, c, -s],
                            [0, s, c]]) # 3x3
    elif axis == 2:
        rot = torch.tensor([[c, -s, 0],
                            [s, c, 0],
                            [0, 0, 1.0]]) # 3x3
    else: 
        assert 0, "Not implemented"

    x_rot = kpts @ rot.t() 
    return x_rot
