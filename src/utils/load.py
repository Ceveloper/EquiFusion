import torch
import os

def get_latest_model_path(ckpnt_path):
    files = [f for f in os.listdir(ckpnt_path) if f.startswith('checkpoint_')]
    epoch = [int(f.split('_val')[0].replace('checkpoint_', '').replace('.pt', '')) for f in files]
    corr = {m:f for m,f in zip(epoch, files)}
    epoch.sort()
    return os.path.join(ckpnt_path, corr[epoch[-1]])
    
def load_model_checkpoint(load_path):
    """Loads a training checkpoint (.pt) or released weights (.safetensors, returned as {'model': state_dict})."""
    if load_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return {"model": load_file(load_path)}
    return torch.load(load_path, weights_only=False)

def load_autoencoder(model, **kwargs):
    checkpoint = load_model_checkpoint(kwargs['pretrained_autoencoder_path'])

    model.load_state_dict(checkpoint["model"])  # strict: raises on missing or unexpected keys
    # freeze weights
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
