
from scipy import linalg
import numpy as np
import os
import sys
import torch

def _calculate_activation_statistics(activations): # [b, 48]
        mu = np.mean(activations, axis=0)  # [48,]
        sigma = np.cov(activations, rowvar=False)  # [48, 48]

        return mu, sigma

def _calculate_fid_helper(statistics_1, statistics_2):
    return _calculate_frechet_distance(statistics_1[0], statistics_1[1],
                                        statistics_2[0], statistics_2[1])

def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def fid(act1, act2):
    statistics_1 = _calculate_activation_statistics(act1)
    statistics_2 = _calculate_activation_statistics(act2)
    return _calculate_fid_helper(statistics_1, statistics_2)


from .fid_classifier import ClassifierForFID
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

def get_classifier(classifier_path, device, **kwargs):
    assert not kwargs['if_consider_hip'], "The skeleton should not consider hip to use the classifier."
    classifier_for_fid = ClassifierForFID(input_size=48, hidden_size=128, hidden_layer=2,
                                                output_size=15, device=device, use_noise=None).to(device)
                                                
    classifier_path = os.path.join(classifier_path, "h36m_classifier.pth")
    classifier_state = torch.load(classifier_path, map_location=device)
    classifier_for_fid.load_state_dict(classifier_state["model"])
    classifier_for_fid.eval()
    return classifier_for_fid

class MetricStorerFID(Metric):

    def __init__(self, output_transform=lambda x: x,  classifier_path='', device='cuda', **kwargs):
        self.all_gt_activations = []
        self.all_pred_activations = []

        assert os.path.exists(classifier_path), f"Cannot find checkpoint of classifier in {classifier_path}"
        self.classifier = get_classifier(classifier_path, device, **kwargs)
        super().__init__(output_transform=output_transform)

    def reset(self):
        self.all_gt_activations = []
        self.all_pred_activations = []
        super().reset()

    def update(self, output):
        # pred, class_idxs, step = self._output_transform(output) # This will be called by engine if we use it
        pred, target = output
        # ----------------------------- Computing features for FID -----------------------------
        # pred -> [batch_size, samples, seq_length, n_joints, n_features])
        pred_ = pred.reshape(*pred.shape[:-2], -1) # [batch_size, samples, seq_length, n_features])
        pred_ = pred_.reshape(-1, *pred_.shape[2:]) # [batch_size * samples, seq_length, n_features])
        pred_ = pred_.permute(0, 2, 1) # [batch_size * samples, n_features, seq_length])

        # same for target
        target_ = target.reshape(*target.shape[:-2],-1) # [batch_size, seq_length, n_features])
        target_ = target_.permute(0, 2, 1) # [batch_size, n_features, seq_length])

        pred_activations = self.classifier.get_fid_features(motion_sequence=pred_.float()).cpu().data.numpy()
        gt_activations = self.classifier.get_fid_features(motion_sequence=target_.float()).cpu().data.numpy()

        self.all_gt_activations.append(gt_activations)
        self.all_pred_activations.append(pred_activations)

    def compute(self):
        if len(self.all_gt_activations) == 0 or len(self.all_pred_activations) == 0:
            raise NotComputableError('MetricStorer must have at least one example before it can be computed.')
        tot = fid(np.concatenate(self.all_gt_activations, axis=0), np.concatenate(self.all_pred_activations, axis=0))
        return tot
