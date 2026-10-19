from .motion import get_motion_representation_objclass
from .kinematic import get_kinematic_objclass, Kinematic
from .motion import SkeletonVanilla
from ...utils.config import merge_cfg

def get_skeleton_class(**kwargs):
    """
    Returns a skeleton class based on the provided parameters.
    
    Parameters:
        **kwargs: Dictionary containing the parameters to determine the skeleton class.
            - motion_repr_type: Type of motion representation (e.g., 'CenterPose', 'Vanilla').
            - dataset_name: Name of the dataset (e.g., 'AMASS', 'Human3.6M').
    
    Returns:
        CompoundSkeletonClass: A class that combines kinematic and motion representation functionalities.
    """
    motion_repr_class = get_motion_representation_objclass(kwargs["motion_repr_type"])
    if "dataset" in kwargs and "dataset_name" not in kwargs["dataset"]: 
        # we have a list of datasets
        internal_dataset_name = list(kwargs["dataset"].values())[0]["dataset_name"] 
    else: 
        internal_dataset_name = kwargs["dataset_name"]
    kinematic_class, dataset_name = get_kinematic_objclass(internal_dataset_name)
    new_class_name = dataset_name + kwargs["motion_repr_type"]
    def init_both_classes(self, *args, **kwargs):
        # Initialize the kinematic class
        if "dataset" in kwargs and "dataset_name" not in kwargs["dataset"]:
            new_config = merge_cfg(kwargs["dataset"][internal_dataset_name], {k: v for k,v in kwargs.items() if k != "dataset"}) 
        else:
            new_config = kwargs   
        kinematic_class.__init__(self, *args, **new_config)
        # Initialize the motion representation class
        motion_repr_class.__init__(self, *args, **new_config)

    CompoundSkeletonClass = type(new_class_name, (kinematic_class, motion_repr_class), {"__init__": init_both_classes,})
    #### THe previous line is creating a class similar to the following and returning the constructor/Class name

    return CompoundSkeletonClass

def create_skeleton(**kwargs):
    CompoundSkeletonClass = get_skeleton_class(**kwargs)
   
    return CompoundSkeletonClass(**kwargs)

def create_motion_repr(**kwargs):
    motion_repr_class = get_motion_representation_objclass(kwargs["motion_repr_type"])
    return motion_repr_class(**kwargs)

