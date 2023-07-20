from torch.nn.functional import gumbel_softmax, relu, softmax
import torch


def categorical_gumbel_softmax_sampling(x, dataset, tau=1., dim=-1, apply_to='all'):
    """
    Applies the gumbel-softmax sampling trick to enhance the performance of continuous optimization techniques on
    categorical optimization objectives.

    :param x: (torch.tensor) The batch of mixed type data in numerical encoding.
    :param dataset: (BaseDataset) The instantiated dataset we are working with.
    :param tau: (float) Temperature parameter for the softmax distribution.
    :param dim: (int) Dimension along which to apply the softmax.
    :param apply_to: (list) The list of categorical features to which to apply the softmax trick.
    :return: (torch.tensor) The resampled x.
    """
    mean, std = dataset.mean, dataset.std
    # de-standardize to have the real distributions
    if dataset.standardized:
        x = std * x + mean
    # if we apply the softmax trick to all categorical features then create a list with all of them
    if apply_to == 'all':
        apply_to = [key for key, item in dataset.train_feature_index_map.items() if item[0] == 'cat']
    # for each categorical feature, resample the feature vector from the gumbel-softmax distribution
    for feature_name, (feature_type, feature_index) in dataset.train_feature_index_map.items():
        if feature_type == 'cat' and not len(feature_index) == 1 and feature_name in apply_to:
            # x[:, feature_index] = relu(x[:, feature_index])
            # x[:, feature_index] += 0.001 * torch.rand(x[:, feature_index].size())
            x[:, feature_index] = gumbel_softmax(torch.log(x[:, feature_index]), tau=tau, dim=dim)
    # re-standardize
    if dataset.standardized:
        x = (x - mean) / std
    return x


def categorical_softmax(x, dataset, tau=1., dim=-1, apply_to='all'):
    """
    Applies a softmax to the categorical one-hots to approximate an argmax effect.

    :param x: (torch.tensor) The batch of mixed type data in numerical encoding.
    :param dataset: (BaseDataset) The instantiated dataset we are working with.
    :param tau: (float) Temperature parameter for the softmax distribution.
    :param dim: (int) Dimension along which to apply the softmax.
    :param apply_to: (list) The list of categorical features to which to apply the softmax trick.
    :return: (torch.tensor) The rescaled x.
    """
    mean, std = dataset.mean, dataset.std
    # de-standardize to have the real distributions
    if dataset.standardized:
        x = std * x + mean
    else:
        x = x * 1.
    # if we apply the softmax trick to all categorical features then create a list with all of them
    if apply_to == 'all':
        apply_to = [key for key, item in dataset.train_feature_index_map.items() if item[0] == 'cat']
    # for each categorical feature, resample the feature vector from the gumbel-softmax distribution
    for feature_name, (feature_type, feature_index) in dataset.train_feature_index_map.items():
        if feature_type == 'cat' and not len(feature_index) == 1 and feature_name in apply_to:
            x[:, feature_index] = softmax(x[:, feature_index]/tau, dim=dim)
    # re-standardize
    if dataset.standardized:
        x = (x - mean) / std
    return x
