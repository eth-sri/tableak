from scipy.optimize import linear_sum_assignment
from .eval_metrics import categorical_accuracy_continuous_tolerance_score
import numpy as np


def match_reconstruction_ground_truth(target_batch, reconstructed_batch, tolerance_map, return_indices=False,
                                      match_based_on='all'):
    """
    For a reconstructed batch of which we do not know the order of datapoints reconstructed, as the loss and hence the
    gradient of the loss is permutation invariant with respect to the input batch, this function calculates the optimal
    reordering i.e. matching to the ground truth batch to get the minimal reconstruction error. It uses the
    reconstruction score 'categorical_accuracy_continuous_tolerance_score'.

    :param target_batch: (np.ndarray) The target batch in mixed representation.
    :param reconstructed_batch: (np.ndarray) The reconstructed batch in mixed representation.
    :param tolerance_map: (list) The tolerance map required to calculate the reconstruction score.
    :param return_indices: (bool) Trigger to return the matching index map as well.
    :param match_based_on: (str) Select based on which feature type to match. Available are 'all', 'cat', 'cont'.
    :return: reordered_reconstructed_batch (np.ndarray), batch_cost_all (np.ndarray), batch_cost_cat (np.ndarray),
        batch_cost_cont (np.ndarray): The correctly reordered reconstructed batch, the minimal cost vectors of all
        feature costs, only categorical feature costs, only continuous feature costs.
    """
    assert match_based_on in ['all', 'cat', 'cont'], 'Please select a valid matching ground from all, cat, cont'

    if len(target_batch.shape) != 2 or len(reconstructed_batch.shape) != 2:
        target_batch = np.reshape(target_batch, (-1, len(tolerance_map)))
        reconstructed_batch = np.reshape(reconstructed_batch, (-1, len(tolerance_map)))

    batch_size = target_batch.shape[0]

    # create the cost matrix for matching the reconstruction with the true data to calculate the score
    cost_matrix_all, cost_matrix_cat, cost_matrix_cont = [np.zeros((batch_size, batch_size)) for _ in range(3)]
    for k, target_data_point in enumerate(target_batch):
        for l, recon_data_point in enumerate(reconstructed_batch):
            cost_all, cost_cat, cost_cont = categorical_accuracy_continuous_tolerance_score(
                target_data_point, recon_data_point, tolerance_map, True)
            cost_matrix_all[k, l], cost_matrix_cat[k, l], cost_matrix_cont[k, l] = cost_all, cost_cat, cost_cont

    # perform the Hungarian algorithm to match the reconstruction
    if match_based_on == 'all':
        row_ind, col_ind = linear_sum_assignment(cost_matrix_all)
    elif match_based_on == 'cat':
        row_ind, col_ind = linear_sum_assignment(cost_matrix_cat)
    else:
        row_ind, col_ind = linear_sum_assignment(cost_matrix_cont)

    # create the "real" cost vectors and reorder the reconstructed batch according to the matching
    reordered_reconstructed_batch = reconstructed_batch[col_ind]
    batch_cost_all, batch_cost_cat, batch_cost_cont = cost_matrix_all[row_ind, col_ind], \
                                                      cost_matrix_cat[row_ind, col_ind], \
                                                      cost_matrix_cont[row_ind, col_ind]

    if return_indices:
        return reordered_reconstructed_batch, batch_cost_all, batch_cost_cat, batch_cost_cont, row_ind, col_ind
    else:
        return reordered_reconstructed_batch, batch_cost_all, batch_cost_cat, batch_cost_cont
