import torch
import sys
sys.path.append("..")
from utils import match_reconstruction_ground_truth
from utils import categorical_softmax


def pooled_ensemble(reconstructions, match_to_batch=None, dataset=None, match_based_on='all', pooling='hard_avg',
                    already_reordered=False, return_std=False):
    """
    Given a sequence of reconstructions this function returns a pooled reconstruction sample. To reorder the data points
    in the individual batches, we match according to the feature group 'match_based_on' to the sample in the
    reconstruction sequence identified by the index match_to_index.

    :param reconstructions: (list of torch.tensor) A list/sequence of reconstructions from which to pool the result.
    :param match_to_batch: (torch.tensor) Match all members to this batch.
    :param dataset: (BaseDataset) The instantiated dataset.
    :param match_based_on: (str) Match by the feature group defined here. Available are 'all', 'cat', 'cont'.
    :param pooling: (str) The type of pooling to apply. Available are 'soft_avg', 'hard_avg', 'median'.
    :param already_reordered: (bool) If True, the matching pre-step will not be performed. Note that in this case the
        given reconstruction also have to be projected already.
    :param return_std: (bool) Toggle to return the standard-deviation of the continuous features.
    :return: (torch.tensor) The pooled result.
    """

    pooling_schemes = ['soft_avg', 'soft_avg+softmax', 'hard_avg', 'median', 'median+softmax']

    if already_reordered:
        reordered_reconstructions = reconstructions
    else:
        reconstructions_decoded = [dataset.decode_batch(rec.detach().clone()) for rec in reconstructions]
        tolerance_map = dataset.create_tolerance_map()
        all_indices_match = []
        for reconstruction in reconstructions_decoded:
            _, _, _, _, _, indices = match_reconstruction_ground_truth(dataset.decode_batch(match_to_batch.detach().clone()),
                                                                       reconstruction, tolerance_map=tolerance_map,
                                                                       return_indices=True, match_based_on=match_based_on)
            all_indices_match.append(indices)
        reordered_reconstructions = torch.stack([rec[idx].detach().clone() for rec, idx in zip(reconstructions, all_indices_match)])

    # continuous feature indices
    cont_feature_indices = [index[0] for feature_type, index in dataset.train_feature_index_map.values() if feature_type == 'cont']

    if pooling == 'soft_avg':
        resulting_reconstruction = reordered_reconstructions.mean(dim=0)
        cont_stds = reordered_reconstructions.std(dim=0)[:, cont_feature_indices]
    elif pooling == 'soft_avg+softmax':
        reordered_reconstructions = torch.stack([categorical_softmax(x, dataset) for x in reordered_reconstructions])
        resulting_reconstruction = reordered_reconstructions.mean(dim=0)
        cont_stds = reordered_reconstructions.std(dim=0)[:, cont_feature_indices]
    elif pooling == 'hard_avg':
        categorized_reordered_reconstructions = [dataset.decode_batch(rec.detach().clone()) for rec in reordered_reconstructions]
        projected_reordered_reconstructions = torch.stack([dataset.encode_batch(rec) for rec in categorized_reordered_reconstructions])
        resulting_reconstruction = projected_reordered_reconstructions.mean(dim=0)
        cont_stds = projected_reordered_reconstructions.std(dim=0)[:, cont_feature_indices]
    elif pooling == 'median':
        resulting_reconstruction = reordered_reconstructions.median(dim=0).values
        cont_stds = reordered_reconstructions.std(dim=0)[:, cont_feature_indices]
    elif pooling == 'median+softmax':
        reordered_reconstructions = torch.stack([categorical_softmax(x, dataset) for x in reordered_reconstructions])
        resulting_reconstruction = reordered_reconstructions.median(dim=0).values
        cont_stds = reordered_reconstructions.std(dim=0)[:, cont_feature_indices]
    elif pooling == 'mixed+softmax':
        reordered_reconstructions = torch.stack([categorical_softmax(x, dataset) for x in reordered_reconstructions])
        resulting_reconstruction = torch.zeros_like(reordered_reconstructions[0])
        for feature_type, feature_index in dataset.train_feature_index_map.values():
            if feature_type == 'cont':
                resulting_reconstruction[:, feature_index] = reordered_reconstructions[:, :, feature_index].median(dim=0).values
            else:
                resulting_reconstruction[:, feature_index] = reordered_reconstructions[:, :, feature_index].mean(dim=0)
        cont_stds = reordered_reconstructions.std(dim=0)[:, cont_feature_indices]
    else:
        raise ValueError(f'Choose a pooling strategy from the schemes {pooling_schemes}')

    if return_std:
        return resulting_reconstruction, cont_stds
    else:
        return resulting_reconstruction
