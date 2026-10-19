import os
import yaml

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS (as in the scripts)
from hydra import initialize, compose
from omegaconf import OmegaConf

import src.core as model_arch
from .utils.config import init_obj
from .utils.config import load_and_merge_autoenc_cfg

if not OmegaConf.has_resolver("eval"):  # used by the configs (registered by the scripts when run from the command line)
    OmegaConf.register_new_resolver("eval", eval)

def create_model(device, **kwargs,):
    model = init_obj(kwargs, 'arch', model_arch).to(device)
    return model


def load_model_config_exp(checkpoint_path):
    exp_folder = os.path.dirname(os.path.dirname(checkpoint_path))
    config_path = os.path.join(exp_folder, 'config.yaml')
    with open(config_path, 'r') as stream:
        cfg_diffusion = yaml.safe_load(stream)
    ae_path = cfg_diffusion['pretrained_autoencoder_path']
    if not os.path.isabs(ae_path):  # released models: path relative to the model folder
        cfg_diffusion['pretrained_autoencoder_path'] = os.path.join(exp_folder, ae_path)
    cfg = load_and_merge_autoenc_cfg(cfg_diffusion)
    return cfg, exp_folder


def load_model_config_exp_autoenc(checkpoint_path):
    exp_folder = os.path.dirname(os.path.dirname(checkpoint_path))
    config_path = os.path.join(exp_folder, 'config.yaml')
    with open(config_path, 'r') as stream:
        cfg = yaml.safe_load(stream)
    return cfg, exp_folder


def quick_cfg_for_inference_no_hydra(checkpoint_path, dataset_name, num_samples=50, split='valid'):
    with initialize(config_path="../configs/config_eval", version_base="1.3.2"):
        cfg = compose(
            config_name="config.yaml",
            overrides=[
                f"dataset={dataset_name}",
                "task=hmp",
                f"dataset_split={split}",
                "stats_mode=deterministic",
                f"num_samples={num_samples}",
                f"batch_size=1",
                f"checkpoint_path='{checkpoint_path}'",
            ]
        )
    cfg = OmegaConf.to_container(cfg, resolve=True)
    # for backwards compatibility with the code
    for subconf in ['task', 'dataset']:
        if subconf in cfg:
            cfg = {**cfg, **cfg[subconf]}
            cfg.pop(subconf)

    from .utils.config import merge_cfg, make_paths_absolute
    cfg = make_paths_absolute(cfg)
    # load config
    cfg_orig, exp_folder = load_model_config_exp(checkpoint_path)
    cfg_orig.pop("dataset", None)  # training datasets: the skeleton is the one of the evaluated dataset
    # merge original experiment config with the current evaluation config
    cfg = merge_cfg(cfg, cfg_orig)
    cfg = merge_cfg(cfg, cfg['method_specs'])
    return cfg


# ----------------------------------------------------------------------------------------------------------
# Inference API (used by the notebooks and the demo)
# ----------------------------------------------------------------------------------------------------------
import types
import numpy as np
import torch

LIMB_NAMES = {"head": 0, "left_arm": 1, "right_arm": 2, "left_leg": 3, "right_leg": 4}  # see Kinematic.idx_to_limb


def download_equifusion(repo_id: str, local_dir: str = None) -> str:
    """Downloads a released model folder from the Hugging Face Hub and returns the path of its checkpoint,
    to be given to `load_equifusion` or eval.py (checkpoint_path=...)."""
    import glob
    from huggingface_hub import snapshot_download
    folder = snapshot_download(repo_id, local_dir=local_dir)
    return sorted(glob.glob(os.path.join(folder, "checkpoints", "checkpoint_*.safetensors")))[-1]


def missing_bones(skeleton, names, generate=False):
    """Bone indices for limb names (see LIMB_NAMES) and bone names (skeleton.bone_names) of a bone direction model.
    The missing bones must not separate the remaining skeleton: every bone after a missing one in its kinematic chain
    must be missing too. To generate missing bones, their mirrored bones (left <-> right) must be observed."""
    assert skeleton.node_type == "bone", "missing limbs or bones are supported by the bone direction models; use missing_joints for the 3D joint models"
    bones = []
    for name in names:
        if name in LIMB_NAMES:
            bones += list(skeleton.limb_bones[name])
        elif name in skeleton.bone_names:
            bones.append(skeleton.bone_names.index(name))
        else:
            raise ValueError(f"Unknown limb or bone '{name}'. Limbs: {list(LIMB_NAMES)}; bones: {skeleton.bone_names}")
    bones = sorted(set(bones))
    children = {i: [j for j, b in enumerate(skeleton.bones) if b[0] == skeleton.bones[i][1]] for i in range(len(skeleton.bones))}
    for b in bones:
        stack, below = list(children[b]), []
        while stack:
            c = stack.pop(); below.append(c); stack += children[c]
        if set(below) - set(bones):
            raise ValueError(f"Missing bone '{skeleton.bone_names[b]}' would disconnect the bones after it in the chain; "
                             f"add {[skeleton.bone_names[c] for c in sorted(set(below) - set(bones))]} to the missing ones")
    if generate:
        mirror = {int(a): int(b) for a, b in skeleton.bone_symmetry} | {int(b): int(a) for a, b in skeleton.bone_symmetry}
        for b in bones:
            if b not in mirror or mirror[b] in bones:
                raise ValueError(f"Bone '{skeleton.bone_names[b]}' can be generated only if its mirrored bone is observed")
    return bones


def load_equifusion(checkpoint_path: str, dataset_name: str, missing_limbs=None, generate_missing_limbs: bool = False,
                    num_samples: int = 50):
    """Loads a released EquiFusion model (autoencoder + latent diffusion) for the skeleton of `dataset_name`.

    missing_limbs: for the bone direction models, names of limbs (LIMB_NAMES) and/or bones (skeleton.bone_names) that are
        not observed; they must not separate the remaining skeleton (see missing_bones). With generate_missing_limbs=True
        their motion is generated (the latents of the mirrored limb are used as conditioning), otherwise the missing
        bones are removed from the graph.
    Returns (model, skeleton, config) to be used with `predict`.
    """
    from .data import create_skeleton
    from .eval_prepare_model import prepare_model
    cfg = quick_cfg_for_inference_no_hydra(checkpoint_path, dataset_name, num_samples=num_samples)
    if missing_limbs and all(l in LIMB_NAMES for l in missing_limbs):
        cfg.update(if_eval_drop_limb=True, drop_limb_ids=[LIMB_NAMES[l] for l in missing_limbs],
                   if_eval_generate_dropped_limbs=generate_missing_limbs)
    elif missing_limbs:  # bone names (and limb names): explicit list of bones
        cfg.update(if_eval_drop_limb=True, drop_limb_ids=[], if_eval_generate_dropped_limbs=generate_missing_limbs,
                   drop_bones=missing_bones(create_skeleton(**cfg), missing_limbs, generate_missing_limbs))
    skeleton = create_skeleton(**cfg)
    model, _, _ = prepare_model(cfg, skeleton, **{**cfg, "silent": True})
    return model, skeleton, cfg


def _graph_adjacency(skeleton):
    return (skeleton.bone_adj_matrix() if skeleton.node_type == "bone" else skeleton.adj_matrix).clone()


def _pad_nodes(x: torch.Tensor, num_nodes: int):
    missing = num_nodes - x.shape[-2]
    return torch.cat([x, x.new_zeros(*x.shape[:-2], missing, x.shape[-1])], dim=-2) if missing > 0 else x


def _pad_adj(adj: torch.Tensor, num_nodes: int):
    out = adj.new_zeros(*adj.shape[:-2], num_nodes, num_nodes)
    out[..., : adj.shape[-2], : adj.shape[-1]] = adj
    return out


@torch.no_grad()
def predict(model, skeleton, config, obs, num_samples: int = 50, seed: int = None, missing_joints=None):
    """Predicts `num_samples` futures of 2 s from 0.5 s of observed motion.

    obs: [T_obs, J, 3] or [B, T_obs, J, 3] joint positions in meters (including the root joint), at the frame rate of
        the dataset given to `load_equifusion` (e.g. 30 frames for AMASS/Nymeria at 60 fps, 25 for H36M at 50 fps).
    missing_joints: for the 3D joint models, names or indices of joints that are not observed (any joint except the
        root); they are removed from the graph and returned as NaN.
    Returns predictions [B, num_samples, T_pred, J-1, 3] in meters, relative to the root joint (the root is not
    returned), in the joint order of the dataset; joints that are missing (not generated) are NaN.
    """
    from .eval_prepare_model import get_prediction, process_evaluation_pair
    device = next(model[0].parameters()).device
    obs = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
    squeeze = obs.dim() == 3
    obs = obs.unsqueeze(0) if squeeze else obs
    if seed is not None:
        from .utils.reproducibility import set_seed
        set_seed(seed)  # seeds and deterministic algorithms, as in the evaluation

    # to model input space (bone directions or rescaled joints relative to the root)
    x = skeleton.tranform_to_input_space(obs)
    adj = _graph_adjacency(skeleton)
    extra = {}
    kept = None
    if getattr(skeleton, "if_eval_drop_limb", False):  # missing limbs (bone direction models)
        x = skeleton.drop_limbs(x)
        if skeleton.if_eval_generate_dropped_limbs:
            extra["adj_orig"] = _pad_adj(adj, config["max_num_nodes"]).expand(x.shape[0], -1, -1).to(device)
        adj = skeleton.adj_after_dropping_nodes(adj, skeleton.drop_joints_ids, keep_size=skeleton.if_eval_generate_dropped_limbs)
    elif missing_joints:  # missing joints (3D joint models)
        assert skeleton.node_type == "joint", "arbitrary missing joints are supported by the 3D joint models"
        name2node = {name: i for i, name in skeleton.node_dict.items()}
        missing = sorted(name2node[j] if isinstance(j, str) else int(j) for j in missing_joints)
        kept = [i for i in range(x.shape[-2]) if i not in missing]
        x_kept = torch.zeros_like(x)
        x_kept[..., : len(kept), :] = x[..., kept, :]
        x = x_kept
        adj = skeleton.adj_after_dropping_nodes(adj, missing)
    x = _pad_nodes(x, config["max_num_nodes"])
    adj = _pad_adj(adj, config["max_num_nodes"]).expand(x.shape[0], -1, -1)

    # datasets not recorded at 60 fps are resampled to the 60-fps windows of the models
    ctx = types.SimpleNamespace(skeleton=skeleton, if_eval_from60_at50fps=config.get("if_eval_from60_at50fps", False),
                                if_eval_from60_at30fps=False, if_eval_from60_at20fps=False)
    if ctx.if_eval_from60_at50fps:
        obs_len = x.shape[-3]
        x = x[..., torch.from_numpy(np.floor(np.arange(0, obs_len, obs_len / 30))).long(), :, :]
        ctx.mask_out = torch.from_numpy(np.ceil(np.arange(0, 120, 60 / 50))).long()

    x, adj = x.to(device), adj.to(device)
    kwargs = {k: v for k, v in config.items() if k not in ("num_samples", "pred_length", "adj", "extra")}
    if getattr(skeleton, "if_eval_generate_dropped_limbs", False):
        kwargs.update(sym_limb_idxs=skeleton.sym_limb_idxs, limb_idxs_dropped=skeleton.drop_joints_ids)
    pred = get_prediction(x, model, num_samples=num_samples, pred_length=config["pred_length"], adj=adj, extra=extra, **kwargs)

    # back to meters
    if kept is not None:
        pred_full = torch.full_like(pred, float("nan"))
        pred_full[..., kept, :] = pred[..., : len(kept), :]
        pred, x_full = pred_full, torch.zeros_like(x)
        x_full[..., kept, :] = x[..., : len(kept), :]
        x = x_full
    T_out = len(ctx.mask_out) if ctx.if_eval_from60_at50fps else pred.shape[2]
    target = torch.zeros(pred.shape[0], T_out, *pred.shape[3:], device=device)  # no ground truth at inference
    _, pred, _, _ = process_evaluation_pair(ctx, target=target, pred_dict={"pred": pred, "obs": x}, adj=adj, extra=extra)
    if getattr(skeleton, "if_eval_drop_limb", False) and not skeleton.if_eval_generate_dropped_limbs:
        # removed limbs: joints of the reduced skeleton -> original joint order, NaN for the missing joints
        pred_full = torch.full((*pred.shape[:-2], skeleton.num_joints - 1, 3), float("nan"), device=pred.device)
        for j_orig, j_new in skeleton.map_old2new_joint.items():
            if j_orig > 0:  # the root joint is not returned
                pred_full[..., j_orig - 1, :] = pred[..., j_new - 1, :]
        pred = pred_full
    return pred[0] if squeeze else pred
