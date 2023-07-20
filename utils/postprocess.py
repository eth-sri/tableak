import torch
import numpy as np
from attacks.inversion_losses import _weighted_CS_SE_loss, _gradient_norm_weighted_CS_SE_loss, _squared_error_loss, _cosine_similarity_loss


def post_process_binaries(x_reconstruct, net, dataset, true_grad, true_labels, criterion, reconstruction_loss,
                          device=None, weights=None, alpha=1e-5):
    """
    A function to post-process the binary features of a reconstructed sample. It proceeds per sample in the given batch
    and tries out all combinations for the layout of the binary features. Finally, it replaces the given sample with
    the binary layout that produced the lowest loss. Note that this function assumes two things:
        1. A lower loss corresponds to a better reconstruction
        2. Samples can be viewed independently to reduce the loss

    :param x_reconstruct: (torch.tensor) The candidate reconstruction data point.
    :param net: (nn.Module) The torch model with respect to which we are trying to invert.
    :param dataset: (BaseDataset) The dataset subject to the inversion.
    :param true_grad: (torch.tensor) The true gradient that the inversion process receives as the input, i.e. the
        gradient sent by the clients.
    :param true_labels: (torch.tensor) The true labels corresponding to the samples.
    :param criterion: (torch.nn) The loss function respect to which the received gradient was calculated.
    :param reconstruction_loss: (str) The name of the loss function that measures the alignment of the guessed gradient
         and the true gradient. Available loss functions are: 'squared_error', 'cosine_sim', 'weighted_combined',
         'norm_weighted_combined'.
    :param device: (str) The name of the device on which the tensors are stored.
    :param weights: (list) Optional argument, controls the weighting of the reconstruction loss function per layer.
    :param alpha: (float) Optional argument, controls the linear combination weight in combined reconstruction losses.
    :return: (torch.tensor) The post-processed reconstruction guess.
    """
    if device is None:
        device = x_reconstruct.device

    rec_loss_function = {
        'squared_error': _squared_error_loss,
        'cosine_sim': _cosine_similarity_loss,
        'weighted_combined': _weighted_CS_SE_loss,
        'norm_weighted_combined': _gradient_norm_weighted_CS_SE_loss
    }

    loss = rec_loss_function[reconstruction_loss]

    mean, std = dataset.mean.detach().clone(), dataset.std.detach().clone()
    batch_size = x_reconstruct.size()[0]

    # extract the binary features' positions
    binary_index_map = {}
    for feature_name, (feature_type, feature_indices) in dataset.train_feature_index_map.items():
        if feature_type == 'cat' and len(feature_indices) == 1:
            binary_index_map[feature_name] = feature_indices

    # calculate all combinations
    n_binary_features = len(binary_index_map)
    combinations = np.zeros((2**n_binary_features, n_binary_features))
    for l in range(combinations.shape[0]):
        binary = [int(s) for s in '{0:b}'.format(l)]
        leading_zeros = [0 for _ in range(n_binary_features - len(binary))]
        combinations[l] = leading_zeros + binary

    # test the switching datapoint-wise
    for i in range(batch_size):
        errors = []
        for combination in combinations:
            current_candidate = x_reconstruct[i].detach().clone()
            for j, binary_feature_index in enumerate(binary_index_map.values()):
                current_candidate[binary_feature_index] = (combination[j] - mean[binary_feature_index]) / std[binary_feature_index] if dataset.standardized else combination[j]
            # calculate the gradient
            dummy_batch = x_reconstruct.detach().clone()
            dummy_batch[i] = current_candidate.detach().clone()
            pred_loss = criterion(net(dummy_batch), true_labels)
            dummy_gradient = torch.autograd.grad(pred_loss, net.parameters())
            dummy_gradient = [grad.detach() for grad in dummy_gradient]
            errors.append(loss(dummy_gradient, true_grad, device, weights, alpha).item())
        # get the minimal error and enter that into the batch
        min_err_index = np.argmin(errors).flatten().item()
        for j, binary_feature_index in enumerate(binary_index_map.values()):
            x_reconstruct[i, binary_feature_index] = (combinations[min_err_index][j] - mean[binary_feature_index]) / std[binary_feature_index] if dataset.standardized else combinations[min_err_index][j]

    return x_reconstruct


def post_process_continuous(x_reconstruct, dataset):
    """
    Takes a batch and clamps its continuous components into their respective valid intervals.

    :param x_reconstruct: (torch.tensor) The unclamped reconstructed datapoint.
    :param dataset: (BaseDataset) The dataset with respect to which we are inverting.
    :return: (torch.tensor) The clamped reconstructed batch.
    """
    with torch.no_grad():
        for feature_name, (feature_type, feature_indices) in dataset.train_feature_index_map.items():
            if feature_type == 'cont':
                if dataset.standardized:
                    x_reconstruct[:, feature_indices] = torch.clamp(x_reconstruct[:, feature_indices],
                                                                    min=dataset.standardized_continuous_bounds[feature_name][0],
                                                                    max=dataset.standardized_continuous_bounds[feature_name][1])
                else:
                    x_reconstruct[:, feature_indices] = torch.clamp(x_reconstruct[:, feature_indices],
                                                                    min=dataset.continuous_bounds[feature_name][0],
                                                                    max=dataset.continuous_bounds[feature_name][1])
            else:
                continue
    return x_reconstruct
