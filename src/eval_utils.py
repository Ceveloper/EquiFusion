import torch
import os
from torch.utils.data import DataLoader

from .data import create_skeleton


from .data import create_skeleton, custom_collate_for_mmgt, get_DatasetClass_for_evaluation

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"


def prepare_eval_dataset(config, split, shuffle=False, augmentation=0, da_mirroring=0, da_rotations=0, drop_last=False, num_workers=None, batch_size=None ,dataset=None, stats_mode="probabilistic", **kwargs):
    data_loader_name = f"data_loader_{split}"

    config[data_loader_name]["shuffle"] = shuffle
    config[data_loader_name]["da_mirroring"] = da_mirroring
    config[data_loader_name]["da_rotations"] = da_rotations
    config[data_loader_name]["augmentation"] = augmentation
    config[data_loader_name]["drop_last"] = drop_last
    config[data_loader_name]["if_permute_keypoints"] = False
    if "probabilistic" in stats_mode.lower():
        config[data_loader_name]["if_load_mmgt"] = True
    else: 
        config[data_loader_name]["if_load_mmgt"] = False
    if batch_size is not None:
        config["batch_size"] = batch_size
    if num_workers is not None:
        config["num_workers"] = 0
    
    skeleton = create_skeleton(**config)   
    if dataset == None:
        split = data_loader_name.split("_")[-1] if 'eval' not in data_loader_name else "_".join(data_loader_name.split("_")[-2:])
        DatasetClass = get_DatasetClass_for_evaluation(config['dataset_name'])
        dataset = DatasetClass(split=split, skeleton=skeleton, **config)

    data_loader = DataLoader(dataset, pin_memory= True, shuffle=config[data_loader_name]["shuffle"], batch_size=config["batch_size"], 
                                num_workers=config["num_workers"], drop_last=config[data_loader_name]["drop_last"],collate_fn=custom_collate_for_mmgt)    
    assert len(data_loader) >0, "Dataset is too small and last batch was dropped"
    return data_loader, dataset, skeleton
