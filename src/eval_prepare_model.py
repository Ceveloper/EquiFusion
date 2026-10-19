import torch
import os
import yaml
from typing import Optional

from .core import DiffusionManager
from .core.network.layers import model_size_mb
from .inference_utils import (
    create_model,
    load_model_config_exp,
    load_model_config_exp_autoenc,
)
from .utils.load import load_model_checkpoint


def get_eval_out_folder(exp_folder, checkpoint_path, data_split, cfg):
    checkpoint = os.path.splitext(checkpoint_path.split("/")[-1])[0].split("_val")[0].split("_")[-1]
    stats_folder = os.path.join(
        exp_folder,
        f"eval_{cfg['dataset_name']}_{cfg['batch_size']}",
        data_split,
        f"{torch.cuda.get_device_name(0).replace(' ', '_')}_seed{cfg['seed']}",
        f"checkpoint{checkpoint}",
    )
    os.makedirs(stats_folder, exist_ok=True)
    return stats_folder


def prepare_autoencoder(config, skeleton, silent=False):
    for i in range(torch.cuda.device_count()):
        if not silent:
            print(f"> GPU {i} ready: {torch.cuda.get_device_name(i)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    # build model architecture
    config["device"] = device
    model = create_model(num_nodes=config["max_num_nodes"], **config)

    # Load models
    checkpoint_path = (
        config["pretrained_autoencoder_path"]
        if "pretrained_autoencoder_path" in config
        else config["checkpoint_path"]
    )
    if not silent:
        print("Loading Autoencoder checkpoint: {} ...".format(checkpoint_path))

    checkpoint_model = load_model_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint_model["model"])  # strict: raises on missing or unexpected keys

    if "n_gpu" in config and config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    return model, device


def prepare_model(config, skeleton, silent=False, **kwargs):
    """
    This function must take as input a config_file and at least a second arg (skeleton) and return a model and a device (cpu or cuda)
    model, device = prepare_model(config, skeleton,  **kwargs)
    """

    model, device = prepare_autoencoder(config, skeleton, silent=silent)
    print("Model size Autoencoder (MB): ", model_size_mb(model))

    # build model architecture
    diffusionmanager = DiffusionManager(
        skeletons={config["dataset_name"]: skeleton},
        num_nodes=config["max_num_nodes"],
        **kwargs,
    )
    diffusion = diffusionmanager.get_diffusion().to(device)
    print("Model size Diffusion (MB): ", model_size_mb(diffusion))

    diffusion.set_eval_dataidx(config["dataset_name"], device=device)
    # Load models
    if not silent:
        print("Loading Diffusion checkpoint: {} ...".format(config["checkpoint_path"]))

    checkpoint_diffusion = load_model_checkpoint(config["checkpoint_path"])

    diffusion.load_state_dict(checkpoint_diffusion["model"])  # strict: raises on missing or unexpected keys

    if "n_gpu" in config and config["n_gpu"] > 1:
        diffusion = torch.nn.DataParallel(diffusion)

    diffusion = diffusion.to(device)
    diffusion.eval()

    return (model, diffusion), device, diffusionmanager


def get_diffusion_latent_codes(
    obs,
    model,
    num_samples=50,
    adj: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    if_eval_generate_dropped_limbs: bool = False,
    if_eval_drop_limb: bool = False,
    **kwargs,
):
    model, diffusion = model
    bs, obs_length, j, f = obs.shape
    sampler_kwargs = {} if "sampler_kwargs" not in kwargs else kwargs["sampler_kwargs"]

    _, (z_past_motion, z_past_bone), _ = model.get_past_embedding(
        obs, adj=adj, mask=mask
    )
    z_past = diffusion.model.get_cond(
        z_past_motion=z_past_motion,
        z_past_bone=z_past_bone,
        adj=adj,
        mask=mask,
    )  # B, N, L where dimension N is repeated
    if if_eval_generate_dropped_limbs and if_eval_drop_limb:
        # we should take the original adj and the mask computed from it.
        sym_limb_idxs = kwargs.get("sym_limb_idxs", None)
        limb_idxs_dropped = kwargs.get("limb_idxs_dropped", None)
        adj = kwargs["extra"]["adj_orig"]
        mask = (adj.sum(-1) != 0).unsqueeze(-1)
        z_past[:, limb_idxs_dropped, :] = z_past[:, sym_limb_idxs, :]
        # z_past[..., num_limbs_pred:num_limbs_pred+len(sym_limb_idxs), :] = z_past[..., sym_limb_idxs, :]

    if kwargs["diffusion_conditioning"]:
        latent_pred, noise = diffusion.sample(
            batch_size=bs,
            num_samples=num_samples,
            x_cond=z_past.unsqueeze(dim=0),
            adj=adj,
            mask=mask,
            **sampler_kwargs,
        )

    else:
        # out_steps has shape -> [N, Joints, Feat]
        latent_pred, _ = diffusion.sample(
            batch_size=bs, num_samples=num_samples, adj=adj, mask=mask, **sampler_kwargs
        )


    return latent_pred, (z_past_motion, z_past_bone)  # , noise


def decode_latent_pred(
    obs,
    latent_pred,
    z_past,
    model,
    num_samples=50,
    pred_length=100,
    adj: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    if_eval_generate_dropped_limbs: bool = False,
    if_eval_drop_limb: bool = False,
    **kwargs,
):
    model, diffusion = model
    bs, obs_length, j, f = obs.shape
    obs = obs.expand(num_samples, -1, -1, -1, -1)
    z_past_motion, z_past_bone = z_past

    z_past_bone = (
        z_past_bone.expand(latent_pred.shape[0], -1, -1, -1)
        if len(z_past_bone.shape) == 3
        else z_past_bone
    )
    z = model.aggregate_latent(z_motion=latent_pred, z_bone=z_past_bone)
    z_past_motion = (
        z_past_motion.expand(latent_pred.shape[0], -1, -1, -1)
        if len(z_past_motion.shape) == 3
        else z_past_motion
    )
    z_past = (z_past_motion, z_past_bone)

    if if_eval_generate_dropped_limbs and if_eval_drop_limb:
        # we should take the original adj and the mask computed from it.
        adj = kwargs["extra"]["adj_orig"]
        mask = (adj.sum(-1) != 0).unsqueeze(-1)

    pred, _ = model.decode(
        obs, z, z_past, ph=pred_length, adj=adj, mask=mask
    )  # [N, t, Joints, 3]
    pred = pred.transpose(0, 1)  # (batch, num_samples, T, J, 3)
    latent_pred = latent_pred.transpose(0, 1)
    pred = pred.view(bs, num_samples, pred_length, j, f)

    return pred


def get_prediction(obs, model, num_samples=50, pred_length=100, **kwargs):
    adj = kwargs["adj"]
    mask = (adj.sum(-1) != 0).unsqueeze(-1)
    lat_pred, z_past = get_diffusion_latent_codes(
        obs,
        model,
        num_samples=num_samples,
        mask=mask,
        **kwargs,
    )
    pred = decode_latent_pred(
        obs,
        lat_pred,
        z_past,
        model,
        num_samples=num_samples,
        pred_length=pred_length,
        mask=mask,
        **kwargs,
    )
    return pred


def process_evaluation_pair(dataset, target, pred_dict, adj=None, extra=None):
    pred, mm_gt, obs = (
        pred_dict["pred"],
        pred_dict["mm_gt"] if "mm_gt" in pred_dict else None,
        pred_dict["obs"],
    )
    skeleton = dataset.skeleton

    if skeleton.node_type == "bone":
        target = target[..., : skeleton.num_nozero_joints, :]
        pred = pred[..., : skeleton.num_nozero_joints, :]
        limb_length_y = torch.linalg.norm(target, dim=-1).unsqueeze(-1)
        obs = obs[..., : skeleton.num_nozero_joints, :]
        limb_length_x = torch.linalg.norm(obs, dim=-1).mean(-2).unsqueeze(-1)

        target = skeleton.transform_to_metric_space(target, limb_length_y)

        if "adj_orig" in extra:
            adj_full = extra["adj_orig"]
            mask_full = (adj_full.sum(-1) != 0).unsqueeze(-1)
            num_joints_past = (adj.sum(-1) != 0).sum(-1).max().item()
            obs[..., skeleton.drop_joints_ids, :] = (
                1.0  # or 0. if we want to have nana
            )
            # obs[..., num_joints_past : skeleton.num_nozero_joints, :] = 1.0

        obs = skeleton.transform_to_metric_space(obs, limb_length_x)
        if "adj_orig" in extra:

            default_bone_length = [
                skeleton.bone_mean[i] for i in skeleton.drop_joints_ids
            ]
            default_bone_length = (
                torch.tensor(default_bone_length, device=pred.device)
                .unsqueeze(0)
                .unsqueeze(-1)
            )  # 1,1, num_dropped, 1
            limb_length_occluded = default_bone_length.expand(
                obs.shape[0], -1, -1
            )  # batch,1,num
            limb_length_x[..., skeleton.drop_joints_ids, :] = limb_length_occluded
        pred = skeleton.transform_to_metric_space(pred, limb_length_x.unsqueeze(1))

        if mm_gt is not None:
            ...  # do nothing, we already centered it previously
    else:
        target = skeleton.transform_to_metric_space(target)
        pred = skeleton.transform_to_metric_space(pred)
        obs = skeleton.transform_to_metric_space(obs)
        target = target[..., : skeleton.num_nozero_joints, :]
        pred = pred[..., : skeleton.num_nozero_joints, :]
        obs = obs[..., : skeleton.num_nozero_joints, :]
        mm_gt = (
            [skeleton.transform_to_metric_space(gt) for gt in mm_gt]
            if mm_gt is not None
            else None
        )
    # if if_eval_from60_at50fps
    # downsample output to 50fps
    if (
        dataset.if_eval_from60_at50fps
        or dataset.if_eval_from60_at30fps
        or dataset.if_eval_from60_at20fps
    ):
        pred = pred[..., dataset.mask_out, :, :]
    if getattr(dataset.skeleton, "if_eval_drop_limb", False):  # skeleton with missing limbs (inference)
        pred = dataset.skeleton.out_postprocess_droplimb(pred)
    batch_size, n_samples, seq_length, num_joints, features = pred.shape
    # batch_size, n_samples, seq_length, num_joints, features = pred.shape
    assert features == 3 and list(target.shape) == [
        batch_size,
        seq_length,
        num_joints,
        features,
    ]
    return target, pred, mm_gt, obs
