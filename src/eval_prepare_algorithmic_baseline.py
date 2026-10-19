import torch
import os


class ZeroVelocityBaseline(torch.nn.Module):
    def __init__(self, **kwargs):
        super(ZeroVelocityBaseline, self).__init__(**kwargs)

    def forward(self, x: torch.Tensor, ph: int = 1) -> torch.Tensor:
        last_frame = x[..., -1, :, :].unsqueeze(-3).clone()
        B, T, J, D = last_frame.shape
        last_frame = last_frame.broadcast_to((B, ph, J, D))
        return last_frame


def load_model_config_exp(checkpoint_path):
    """
    Load the config file and the experiment folder from the checkpoint_path
    exp_folder: is a absolute or relative path
    cfg: dict of configuration
    """
    exp_folder = checkpoint_path
    return {}, exp_folder


def get_eval_out_folder(exp_folder, checkpoint_path, data_split, cfg):
    """
    Create folder where to store the evaluation results
    """
    stats_folder = os.path.join(
        exp_folder, f"eval_seed{cfg['seed']}", data_split)
    os.makedirs(stats_folder, exist_ok=True)
    return stats_folder


def prepare_model(config, skeleton, silent=False, **kwargs):
    """
    This function must take as input a config_file and at least a second arg (skeleton) and return a laoded model on a device (cpu or cuda)
    model, device = prepare_model(config, skeleton,  **kwargs)
    """
    for i in range(torch.cuda.device_count()):
        if not silent:
            print(f"> GPU {i} ready: {torch.cuda.get_device_name(i)}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.set_grad_enabled(False)

    baseline_name = config['method_name']
    alg_baseline = globals()[baseline_name]()
    alg_baseline = alg_baseline.to(device)
    alg_baseline.eval()

    return alg_baseline, device


def get_prediction(obs, model, num_samples=50, **kwargs):
    """
    Generate num_samples predictions from model and input obs.
    return a dict or anything that will be used by process_evaluation_pair
    """
    pred = model(obs, ph=kwargs['pred_length'])
    pred = pred.unsqueeze(1)  # [batch, 1, t, n_joints, 3]
    return pred


def process_evaluation_pair(dataset, target, pred_dict, **kwargs):
    """
    Process the target and the prediction and return them in the right format for metrics computation
    """
    skeleton = dataset.skeleton
    pred, mm_gt, obs = pred_dict['pred'], pred_dict['mm_gt'], pred_dict['obs']
    batch_size, n_samples, seq_length, num_joints, features = pred.shape
    target = skeleton.transform_to_metric_space(target)
    pred = skeleton.transform_to_metric_space(pred)
    obs = skeleton.transform_to_metric_space(obs)
    mm_gt = [skeleton.transform_to_metric_space(
        gt) for gt in mm_gt] if mm_gt is not None else None
    # batch_size, n_samples, n_diffusion_steps, seq_length, num_joints, features = pred.shape
    assert features == 3 and list(target.shape) == [
        batch_size, seq_length, num_joints, features]
    # return dummy altent prediction
    return target, pred, mm_gt, obs
