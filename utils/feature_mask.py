import torch


def create_feature_mask(x, dataset, feature_names):
    """
    Creates a binary mask over the input data. This allows us to for example restrict optimization to just a subset of
    the features in x by masking its gradient.

    :param x: (torch.tensor) The datapoint in which shape we make the mask.
    :param dataset: (datasets.BaseDataset) The dataset wrt. which we are inverting.
    :param feature_names: (list) The names of the features we wish to keep by the masking.
    :return: (torch.tensor) The mask that can be used to highlight the features.
    """
    device = x.device
    mask = torch.zeros_like(x, device=device)
    for feature_name in feature_names:
        feature_indexes = dataset.train_feature_index_map[feature_name][1]
        mask[:, feature_indexes] = torch.ones((x.size()[0], len(feature_indexes)), device=device)
    return mask
