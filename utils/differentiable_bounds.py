import torch


def sigmoid_bound(x, lb, ub, T=1.0):
    """
    A simple sigmoid enforcing a lower and an upper bound on the input.

    :param x: (torch.tensor) The input data.
    :param lb: (float) Lower bound.
    :param ub: (float) Upper bound.
    :param T: (float) Optional temperature.
    :return: (torch.tensor) The converted data point.
    """
    x_out = (ub - lb) / (1 + torch.exp(-x / T)) + lb
    return x_out


def continuous_sigmoid_bound(x, dataset, T=1.0):
    """
    A sigmoid enforcing the bounds on the continuous features.

    :param x: (torch.tensor) The input mixed-type data.
    :param dataset: (BaseDataset) The instantiated dataset.
    :param T: (float) An optional temperature parameter.
    :return: (torch.tensor) The converted data point.
    """
    bounds = dataset.standardized_continuous_bounds if dataset.standardized else dataset.continuous_bounds
    for feature_name, (feature_type, feature_index) in dataset.train_feature_index_map.items():
        if feature_type == 'cont':
            x[:, feature_index] = sigmoid_bound(x[:, feature_index], bounds[feature_name][0], bounds[feature_name][1], T)
    return x
