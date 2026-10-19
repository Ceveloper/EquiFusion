import torch


def loss(pred, y, type="mse", reduction="mean", feature_reduction='sum', mask=None, masking=True):
    if type == "mse":
        out = torch.nn.MSELoss(reduction="none")(pred, y)
    elif type in ["l1", "L1"]:
        out = torch.nn.L1Loss(reduction="none")(pred, y)
    else:
        assert 0, "Not implemnted"

    loss = out.sum(-1) if feature_reduction == 'sum' else out.mean(-1)  # spatial size [batch, time, joints]
    loss = (loss  # spatial size [32, 13, 25, 5, 128],
            .mean(-2)  # timesteps
            )
    if masking:
        if mask is None:
            mask = loss != 0.
        else:
            mask = mask.squeeze(-1)
        num_joints = mask.sum(-1)
        loss = loss.sum(-1)/num_joints  # keypoints
    else:
        loss = loss.mean(-1)  # keypoints

    if reduction == "mean":
        return loss.mean()
    elif reduction == "none":
        return loss
    else:
        assert 0, "Not implemnted"
