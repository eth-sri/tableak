from .eval_metrics import feature_wise_accuracy_score
import numpy as np


def calculate_entropy_heat_map(reconstructed_batch, true_batch, cont_stds, dataset, tolerance_map=None, return_error_map=True):
    """
    Given the fuzzy reconstruction from an ensemble of reconstructions, this function calculates the feature-wise
    entropies. For the categorical features we have the easy case of discrete entropy, which for comparability we
    normalize by the log of the support. The continuous features are assumed to be gaussian with variance sigma^2, and
    their entropy is the calculated as the closed form Gaussian differential entropy.

    :param reconstructed_batch: (torch.tensor) The fuzzy reconstruction stemming from an ensemble.
    :param true_batch: (torch.tensor) The ground truth data.
    :param cont_stds: (torch.tensor) The standard deviations of the continuous features at ensembling.
    :param dataset: (BaseDataset) The instantiated dataset used.
    :param tolerance_map: (list) Tolerance map for the error estimation.
    :param return_error_map: (bool) Toggle to return also the error heat map.
    :return: (tuple of np.array or np.array) Either both the entropy heat map and the error heat map
        (if return_error_map) or just the entropy heat map.
    """
    if tolerance_map is None:
        tolerance_map = dataset.create_tolerance_map()

    true_batch_decoded = dataset.decode_batch(true_batch, standardized=dataset.standardized)
    reconstructed_batch_decoded = dataset.decode_batch(reconstructed_batch, standardized=dataset.standardized)
    reconstructed_batch = dataset.de_standardize(reconstructed_batch) if dataset.standardized else reconstructed_batch

    heat_map_ground_truth = []
    heat_map_entropy_based = []
    for line_recon_decoded, line_recon, true_line, std_line in zip(reconstructed_batch_decoded,
                                                                   reconstructed_batch,
                                                                   true_batch_decoded, cont_stds):
        heat_map_ground_truth_line = []
        heat_map_entropy_based_line = []
        scores = feature_wise_accuracy_score(true_line, line_recon_decoded, tolerance_map, dataset.train_features)
        cont_index = 0
        for feature_name, (feature_type, feature_index) in dataset.train_feature_index_map.items():
            if feature_type == 'cat':
                normalized_line = line_recon[feature_index] / line_recon[feature_index].sum().item()
                entropy = normalized_line.T @ (-np.log(normalized_line + 1e-7))
                heat_map_ground_truth_line.append(scores[feature_name])
                heat_map_entropy_based_line.append(entropy/np.log(len(feature_index)))
            else:
                heat_map_ground_truth_line.append(scores[feature_name])
                heat_map_entropy_based_line.append(0.5 + 0.5 * np.log(2 * np.pi * (std_line[cont_index].item() ** 2)))
                cont_index += 1
        heat_map_ground_truth.append(heat_map_ground_truth_line)
        heat_map_entropy_based.append(heat_map_entropy_based_line)

    if return_error_map:
        return np.array(heat_map_entropy_based), np.array(heat_map_ground_truth)
    else:
        return np.array(heat_map_entropy_based)
