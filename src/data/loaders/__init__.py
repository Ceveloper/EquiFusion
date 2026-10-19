from .h36m  import  H36MDataset
from .amass  import  AMASSDataset
from .amass_zeroshot import ZeroShotAMASSDataset
from .nymeria import NymeriaDataset
from .base import custom_collate_for_mmgt


from .base import DynamicKinematicMotionDataset ,get_DynamicKinematicMotionDatasetEval




class MOYOZeroShotDataset(ZeroShotAMASSDataset):
    dataset_name = 'moyo'


################################################################################################ 

def get_DatasetClass_for_evaluation(dataset_name):
    dataset_name = {'h36m': 'H36M', 'amass': 'AMASS', 'moyo': 'MOYOZeroShot', 'nymeria': 'Nymeria'}[dataset_name.lower()]
    return get_DynamicKinematicMotionDatasetEval(globals()[dataset_name+"Dataset"])

