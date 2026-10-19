from .utils import get_adj_matrix
from .h36m import H36MKinematic
from .amass import AMASSKinematic
from .nymeria import NymeriaKinematic
from .base import Kinematic

def get_kinematic_objclass(dataset_name):
    dataset_type = {'h36m': 'H36M', 'amass': 'AMASS', 'nymeria': 'Nymeria', 'moyo': 'AMASS'}[dataset_name.lower()]
    return globals()[dataset_type+"Kinematic"], dataset_type