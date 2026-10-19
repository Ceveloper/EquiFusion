import torch  

class Skeleton():
    node_hip = {0: 'GlobalRoot'}
    node_type = "joint"  # graph nodes are joints ('joint') or bones ('bone')

    # """
    # This class creates a slkeleton that considers hip and pose separately. 
    # The hip is the first joint, and the pose is the rest of the joints.
    # The sequence is centered around the first joint in the sequence, or around the hip joint at frame seq_centering
    # It is possible to consider the hip or not.

    def __init__(self, if_consider_hip=True, obs_length=50, pred_length=100, seq_centering: int = 0, **kwargs):
        
        self.if_consider_hip = if_consider_hip
        self.obs_length = obs_length
        self.pred_length = pred_length
        assert all([len(val.replace('Hip', ''))>1 for val in self.node_hip.values()]) # in this way it will have a different joint_id mapping from the other centered hip joints.
        
        
        self.seq_centering = seq_centering
        assert self.seq_centering < 0 or self.seq_centering < self.obs_length+self.pred_length, "seq_centering should be less than total seq length"

    
    #################  functions to obtain input poses ########################################################################################
    # THese funcitons take all kpts as input and return all kpts as output (not only pose or hip)
    def _get_where_is_seq_centered(self):
        if self.seq_centering < 0:
            return self.obs_length+self.seq_centering
            assert 0, "This was changed from - to +, so your old exps are not reproducible..."
        else: 
            return self.seq_centering
        
    def transform_hip_to_input_space(self, data):
        centered, hips = data[..., 1:, :], data[..., 0:1, :]
        #set beginning of seq to where_isZero
        where_isZero = self._get_where_is_seq_centered()
        hips = hips - hips[...,where_isZero,:,:].unsqueeze(-3)
        return torch.cat([hips, centered], dim=-2)
    
    def tranform_to_input_space(self, data):
        data = self.tranform_to_input_space_pose_only(data)
        # centered, hips = data[..., 1:, :], data[..., 0, :].unsqueeze(-2)
        if not self.if_consider_hip:
            kpts = data[..., 1:, :]
        else: 
            kpts = self.transform_hip_to_input_space(data)
        return kpts
    
    def tranform_to_input_space_pose_only(self, data):
        ...
        return data
    
    def add_zero_pad_center_hip(self, kpts):
        shape = list(kpts.shape)
        shape[-2] = 1
        kpts = torch.cat([torch.zeros(shape, device=kpts.device), kpts], dim=-2)
        return kpts
    
    def if_add_zero_pad_center_hip(self, kpts):
        if not self.if_consider_hip and kpts.shape[-2] == self.num_joints - 1:
            kpts =self.add_zero_pad_center_hip(kpts)
        return kpts
    
    ##########################  function to tranform back to metric space ##########################################################################
    
    def transform_hip_to_metric_space(self, kpts):
        ... 
        return kpts

    def _merge_hip_and_poseinmetricspace(self, hip_coords, kpts):
        ...
        return torch.cat([hip_coords, kpts], dim=-2)
    
    def transform_to_metric_space(self, kpts):
        """_summary_

        Returns:
            kpts: kpts in 3D coordinates
        """

        if self.if_consider_hip:
            kpts = self.transform_hip_to_metric_space(kpts) # hip is now represented in metric space, so it is first location.     
            hip_coords =  kpts[...,:1,:]    
            kpts = self.transform_to_metric_space_pose_only(kpts[...,1:,:])
            return self._merge_hip_and_poseinmetricspace(hip_coords, kpts)

        else: 
            # input is without hips. 
            kpts = self.transform_to_metric_space_pose_only(kpts)
            return kpts
    
    def transform_to_metric_space_pose_only(self, kpts):  
            """This is the only funtion that tkaes as input the pose and returns the pose joints"""
            ...
            return kpts


    

