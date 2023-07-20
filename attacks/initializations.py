import sys
sys.path.append("..")
import torch
import numpy as np
from .inversion_losses import _weighted_CS_SE_loss, _gradient_norm_weighted_CS_SE_loss, _squared_error_loss, \
    _cosine_similarity_loss
from torch.nn.functional import conv1d


def _uniform_initialization(x_true, dataset=None, device=None):
    """
    All features are initialized independently and uniformly on the interval [-1, 1].

    :param x_true: (torch.tensor) The true datapoint that we are trying to win back.
    :param dataset: (BaseDataset) The dataset with respect to which the reconstruction is happening.
    :param device: (str) The device on which the tensors are stored.
    :return: (torch.tensor) The initial guess for our reconstruction.
    """
    if device is None:
        device = x_true.device
    x_init = (torch.rand(x_true.shape, device=device) - 0.5) * 2
    return x_init


def _gaussian_initialization(x_true, dataset, device=None):
    """
    All features are initialized independently according to a Gaussian with the same mean and variance as the feature.

    :param x_true: (torch.tensor) The true datapoint that we are trying to win back.
    :param dataset: (BaseDataset) The dataset with respect to which the reconstruction is happening.
    :param device: (str) The device on which the tensors are stored.
    :return: (torch.tensor) The initial guess for our reconstruction.
    """
    if device is None:
        device = x_true.device
    x_init = torch.randn_like(x_true, device=device)
    # if the dataset is standardized, we can leave these sample as is, if not, we perform reparametrization
    if not dataset.standardized:
        mean = dataset.mean
        std = dataset.std
        x_init *= torch.reshape(std, (1, -1))
        x_init += torch.reshape(mean, (1, -1))
    return x_init


def _mean_initialization(x_true, dataset, device=None):
    """
    All features are initialized to their mean values.

    :param x_true: (torch.tensor) The true datapoint that we are trying to win back.
    :param dataset: (BaseDataset) The dataset with respect to which the reconstruction is happening.
    :param device: (str) The device on which the tensors are stored.
    :return: (torch.tensor) The initial guess for our reconstruction.
    """
    if device is None:
        device = x_true.device
    if dataset.standardized:
        x_init = torch.zeros_like(x_true, device=device)
    else:
        x_init = torch.ones_like(x_true, device=device)
        mean = dataset.mean
        x_init *= torch.reshape(mean, (1, -1))
    return x_init


def _dataset_sample_initialization(x_true, dataset, device=None):
    """
    The initial seed is a sample from the dataset.

    :param x_true: (torch.tensor) The true datapoint that we are trying to win back.
    :param dataset: (BaseDataset) The dataset with respect to which the reconstruction is happening.
    :param device: (str) The device on which the tensors are stored.
    :return: (torch.tensor) The initial guess for our reconstruction.
    """
    if device is None:
        device = x_true.device
    Xtrain = dataset.get_Xtrain()
    batch_size = x_true.size()[0]
    batchindices = torch.tensor(np.random.randint(Xtrain.size()[0], size=batch_size)).to(device)
    x_init = Xtrain[batchindices].clone().detach()
    return x_init


def _likelihood_prior_sample_initialization(x_true, dataset, device=None):
    """
    The initial seed is a sample from the feature marginals for each feature independently.

    :param x_true: (torch.tensor) The true datapoint that we are trying to win back.
    :param dataset: (BaseDataset) The dataset with respect to which the reconstruction is happening.
    :param device: (str) The device on which the tensors are stored.
    :return: (torch.tensor) The initial guess for our reconstruction.
    """
    if device is None:
        device = x_true.device
    batch_size = x_true.size()[0]
    x_init = np.zeros((batch_size, len(dataset.train_features)), dtype='object')
    for i, (feature_name, feature_values) in enumerate(dataset.train_features.items()):
        if feature_values is None:
            lower, upper = dataset.continuous_bounds[feature_name]
            cont_histogram = dataset.cont_histograms[feature_name]
            if len(cont_histogram) < 100:
                feature_range = np.arange(lower, upper + 1)
            else:
                delta = (upper - lower) / 100
                feature_range = np.array([lower + j * delta for j in range(100)])
            x_init[:, i] = np.random.choice(feature_range, batch_size, p=dataset.cont_histograms[feature_name])
        else:
            p = dataset.categorical_histograms[feature_name]
            x_init[:, i] = np.random.choice(feature_values, batch_size, p=p)
    x_init = dataset.encode_batch(x_init, standardize=dataset.standardized)
    x_init.to(device)
    return x_init


def _mixed_initialization(x_true, dataset, device=None):
    """
    The categorical features are initialized uniformly whereas the continuous features are initialized according to
    their marginals.

    :param x_true: (torch.tensor) The true datapoint that we are trying to win back.
    :param dataset: (BaseDataset) The dataset with respect to which the reconstruction is happening.
    :param device: (str) The device on which the tensors are stored.
    :return: (torch.tensor) The initial guess for our reconstruction.
    """
    if device is None:
        device = x_true.device

    # create feature masks
    index_map = dataset.train_feature_index_map
    cat_mask = torch.ones_like(x_true)
    for feature_type, feature_index in index_map.values():
        if feature_type == 'cont':
            cat_mask[:, feature_index] = 0.
    cont_mask = torch.ones_like(x_true) - cat_mask

    cat_unif_init = _uniform_initialization(x_true, dataset, device)
    cont_likelihood_init = _likelihood_prior_sample_initialization(x_true, dataset, device)

    return cat_mask * cat_unif_init + cont_mask + cont_likelihood_init


def _best_sample_initialization(x_true, dataset, true_gradient, net, criterion, true_labels,
                                reconstruction_loss='cosine_sim', n_samples=1000, averaging_steps=2, weights=None,
                                alpha=1e-5, device=None):
    """

    :param x_true:
    :param dataset:
    :param true_gradient:
    :param net:
    :param criterion:
    :param true_labels:
    :param reconstruction_loss:
    :param n_samples:
    :param averaging_steps:
    :param weights:
    :param alpha:
    :param device:
    :return:
    """

    if device is None:
        device = x_true.device

    rec_loss_function = {
        'squared_error': _squared_error_loss,
        'cosine_sim': _cosine_similarity_loss,
        'weighted_combined': _weighted_CS_SE_loss,
        'norm_weighted_combined': _gradient_norm_weighted_CS_SE_loss
    }

    best_sample = None
    best_score = None
    for _ in range(n_samples):
        # get the current candidate for the initialization
        current_candidate = _likelihood_prior_sample_initialization(x_true, dataset, device)
        # get its gradient and check how well it fits
        candidate_loss = criterion(net(current_candidate), true_labels)
        candidate_gradient = torch.autograd.grad(candidate_loss, net.parameters())
        candidate_gradient = [grad.detach() for grad in candidate_gradient]
        candidate_reconstruction_loss = rec_loss_function[reconstruction_loss](candidate_gradient, true_gradient, device, weights, alpha).item()
        # check if this loss is better than our current best, if yes replace it and the current datapoint
        if best_sample is None or candidate_reconstruction_loss < best_score:
            best_sample = current_candidate.detach().clone()
            best_score = candidate_reconstruction_loss

    # smoothen out the categorical features a bit -- helps optimization later
    weight = torch.tensor([1/10, 1/10, 6/10, 1/10, 1/10]).unsqueeze(0).unsqueeze(1).float()
    for feature_type, feature_index in dataset.train_feature_index_map.items():
        if feature_type == 'cat':
            if len(feature_index) == 1:
                # just add a tiny bit of noise to the binary features
                best_sample[:, feature_index] += 0.2 * torch.rand(best_sample.size()[0]) - 0.1
            else:
                # add some noise
                best_sample[:, feature_index] += 0.3 * torch.rand(best_sample.size()[0]) - 0.15
                for _ in range(averaging_steps):
                    best_sample[:, feature_index] = conv1d(best_sample[:, feature_index].unsqueeze(1), weight, padding=2).squeeze(1)

    return best_sample
