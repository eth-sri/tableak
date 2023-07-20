import sys
sys.path.append("..")
import torch
from torch.nn import functional as F
from utils import gmm_pdf_batch, categorical_softmax


def _joint_gmm_prior(x_reconstruct, dataset, labels=None, softmax_trick=None, T=None, device=None):
    """
    Returns the average negative log-likelihood of a batch's continuous components that is produced by a fitted joint
    GMM.

    :param x_reconstruct: (torch.tensor) The batch being reconstructed.
    :param dataset: (BaseDataset) The dataset with respect to which the reconstruction is happening.
    :param labels: (torch.tensor) The labels belonging to the batch.
    :param softmax_trick: (bool) Set to True if the softmax trick is already applied to the datapoint.
    :param T: (float) A temperature parameter to control any softmaxes involved in the calculation.
    :param device: (str) The device on which the tensors are stored.
    :return: (torch.tensor) The calculated prior cost to be added onto the loss.
    """
    gmm_weights = dataset.gmm_parameters['all'][0].clone().detach().float()
    gmm_means = dataset.gmm_parameters['all'][1].clone().detach().float()
    gmm_covariances = dataset.gmm_parameters['all'][2].clone().detach().float()

    cont_indices = [value[1][0] for value in dataset.train_feature_index_map.values() if value[0] == 'cont']

    average_nll = gmm_pdf_batch(x_reconstruct[:, cont_indices], means=gmm_means, covariances=gmm_covariances,
                                mixture_weights=gmm_weights, nll=True)
    return average_nll


def _mean_field_gmm_prior(x_reconstruct, dataset, labels=None, softmax_trick=None, T=None, device=None):
    """
    Returns the average negative log-likelihood of a batch's continuous components that is produced by a mean-field GMM,
    i.e. fitting a separate GMM to each of the continuous components.

    :param x_reconstruct: (torch.tensor) The batch being reconstructed.
    :param dataset: (BaseDataset) The dataset with respect to which the reconstruction is happening.
    :param labels: (torch.tensor) The labels belonging to the batch.
    :param softmax_trick: (bool) Set to True if the softmax trick is already applied to the datapoint.
    :param T: (float) A temperature parameter to control any softmaxes involved in the calculation.
    :param device: (str) The device on which the tensors are stored.
    :return: (torch.tensor) The calculated prior cost to be added onto the loss.
    """
    if device is None:
        device = x_reconstruct.device

    prior_cost = torch.as_tensor([0.0], device=device)

    for feature_name, (feature_type, feature_index) in dataset.train_feature_index_map.items():
        if feature_type == 'cont':
            gmm_weights = dataset.gmm_parameters[feature_name][0].clone().detach().float()
            gmm_means = dataset.gmm_parameters[feature_name][1].clone().detach().float()
            gmm_covariances = dataset.gmm_parameters[feature_name][2].clone().detach().float()
            prior_cost += gmm_pdf_batch(torch.reshape(x_reconstruct[:, feature_index[0]], (-1, 1)), means=gmm_means,
                                        covariances=gmm_covariances, mixture_weights=gmm_weights, nll=True)
    return prior_cost


def _categorical_prior(x_reconstruct, dataset, labels=None, softmax_trick=None, T=1., device=None):
    """
    Returns the average negative log-likelihood with respect to each categorical feature's individual estimated marginal
    distribution in the dataset. The selection necessary for calculating the categorical distribution is continuously
    estimated by a softmax with temperature T.

    :param x_reconstruct: (torch.tensor) The batch being reconstructed.
    :param dataset: (BaseDataset) The dataset with respect to which the reconstruction is happening.
    :param labels: (torch.tensor) The labels belonging to the batch.
    :param softmax_trick: (bool) Set to True if the softmax trick is already applied to the datapoint.
    :param T: (float) A temperature parameter to control any softmaxes involved in the calculation.
    :param device: (str) The device on which the tensors are stored.
    :return: (torch.tensor) The calculated prior cost to be added onto the loss.
    """
    if device is None:
        device = x_reconstruct.device

    batch_size = x_reconstruct.size()[0]
    prior_cost = torch.as_tensor([0.0], device=device)

    for key, (feature_type, feature_indices) in dataset.train_feature_index_map.items():
        if feature_type == 'cont':
            continue
        else:
            neg_logp = -torch.log(torch.tensor(dataset.categorical_histograms[key] + 1e-8, device=device)).float()
            if len(feature_indices) == 1:
                # non-rounded thresholding to label 1 for binary features
                sig_to_1 = torch.sigmoid(x_reconstruct[:, feature_indices])
                expanded_feature = torch.cat([1. - sig_to_1, sig_to_1], dim=1)
                hard_rounded = F.softmax(expanded_feature / T, dim=1).float()
            else:
                hard_rounded = F.softmax(x_reconstruct[:, feature_indices] / T, dim=1).float()
            prior_cost += torch.matmul(hard_rounded, neg_logp).sum() / batch_size

    return prior_cost


def _categorical_l2_prior(x_reconstruct, dataset, labels=None, softmax_trick=None, T=None, device=None):
    """
    We know that the categorical features have to be reconstructed as one-hot vectors, which have norm 1. Thereby, we
    try to enforce norm 1 for these features by calculating the square error between their current norm and 1. For now,
    this prior does not affect binary features.

    :param x_reconstruct: (torch.tensor) The batch being reconstructed.
    :param dataset: (BaseDataset) The dataset with respect to which the reconstruction is happening.
    :param labels: (torch.tensor) The labels belonging to the batch.
    :param softmax_trick: (bool) Set to True if the softmax trick is already applied to the datapoint.
    :param T: (float) A temperature parameter to control any softmaxes involved in the calculation.
    :param device: (str) The device on which the tensors are stored.
    :return: (torch.tensor) The calculated prior cost to be added onto the loss.
    """
    if device is None:
        device = x_reconstruct.device
    batch_size = x_reconstruct.size()[0]
    prior_cost = torch.as_tensor([0.0], device=device)
    mean, std = dataset.mean, dataset.std
    for key, (feature_type, feature_indices) in dataset.train_feature_index_map.items():
        if feature_type == 'cont' or len(feature_indices) < 2:
            continue
        else:
            cat_features = x_reconstruct[:, feature_indices]
            if dataset.standardized:
                cat_features *= std[feature_indices]
                cat_features += mean[feature_indices]
            norm = cat_features.pow(2).sum(dim=1).sqrt()
            prior_cost += (1. - norm).pow(2).sum() / batch_size
    return prior_cost


def _categorical_l1_prior(x_reconstruct, dataset, labels=None, softmax_trick=None, T=None, device=None):
    """
    We know that the categorical features have to be reconstructed as one-hot vectors, which have l1 norm 1. Thereby, we
    try to enforce l1 norm 1 for these features by calculating the square error between their current norm and 1. For
    now, this prior does not affect binary features.

    :param x_reconstruct: (torch.tensor) The batch being reconstructed.
    :param dataset: (BaseDataset) The dataset with respect to which the reconstruction is happening.
    :param labels: (torch.tensor) The labels belonging to the batch.
    :param softmax_trick: (bool) Set to True if the softmax trick is already applied to the datapoint.
    :param T: (float) A temperature parameter to control any softmaxes involved in the calculation.
    :param device: (str) The device on which the tensors are stored.
    :return: (torch.tensor) The calculated prior cost to be added onto the loss.
    """
    if device is None:
        device = x_reconstruct.device
    batch_size = x_reconstruct.size()[0]
    prior_cost = torch.as_tensor([0.0], device=device)
    mean, std = dataset.mean, dataset.std
    for key, (feature_type, feature_indices) in dataset.train_feature_index_map.items():
        if feature_type == 'cont' or len(feature_indices) < 2:
            continue
        else:
            cat_features = x_reconstruct[:, feature_indices]
            if dataset.standardized:
                cat_features *= std[feature_indices]
                cat_features += mean[feature_indices]
            norm = cat_features.abs().sum(dim=1)
            prior_cost += (1. - norm).pow(2).sum() / batch_size
    return prior_cost


# ------------------------  ADDITIONS FOR SEE THROUGH GRADIENTS EXPERIMENTS ------------------------ #

def _all_line_l2_prior(x_reconstruct, dataset, labels=None, softmax_trick=None, T=None, device=None):
    """
    Prior that simply penalizes the whole batch not to have too high of an L2 norm. This makes sense for 
    normalized data, and for data that does not conatin outliers.

    :param x_reconstruct: (torch.tensor) The batch being reconstructed.
    :param dataset: (BaseDataset) The dataset with respect to which the reconstruction is happening.
    :param labels: (torch.tensor) The labels belonging to the batch.
    :param softmax_trick: (bool) Set to True if the softmax trick is already applied to the datapoint.
    :param T: (float) A temperature parameter to control any softmaxes involved in the calculation.
    :param device: (str) The device on which the tensors are stored.
    :return: (torch.tensor) The calculated prior cost to be added onto the loss.
    """
    if device is None:
        device = x_reconstruct.device
    l2_norm_line = x_reconstruct.pow(2).sum()
    return l2_norm_line


def _batch_norm_prior(x_reconstruct, dataset, true_bn_stats, measured_bn_stats, labels=None, 
                      softmax_trick=None, T=None, device=None):
    """
    Prior that regularizes based on the knowledge of the batchnorm statistics.

    :param x_reconstruct: (torch.tensor) The batch being reconstructed.
    :param dataset: (BaseDataset) The dataset with respect to which the reconstruction is happening.
    :param true_bn_stats: (list[tuple]) The true BN stats corresponding to the true batch.
    :param measured_bn_stats: (list[tuple]) The BN stats measured on the current reconstruction.
    :param labels: (torch.tensor) The labels belonging to the batch.
    :param softmax_trick: (bool) Set to True if the softmax trick is already applied to the datapoint.
    :param T: (float) A temperature parameter to control any softmaxes involved in the calculation.
    :param device: (str) The device on which the tensors are stored.
    :return: (torch.tensor) The calculated prior cost to be added onto the loss.
    """
    if device is None:
        device = x_reconstruct.device
    regularizer = torch.tensor([0.], device=device)
    for (true_mean, true_var), (measured_mean, measured_var) in zip(true_bn_stats, measured_bn_stats):
        regularizer += (true_mean - measured_mean).pow(2).sum() + (true_var - measured_var).pow(2).sum()
    return regularizer
    
# ---------------------  END ADDITIONS FOR SEE THROUGH GRADIENTS EXPERIMENTS END --------------------- #


def _continuous_uniform_prior(x_reconstruct, dataset, labels=None, softmax_trick=None, T=None, device=None):
    """
    As the continuous features are restricted to an interval, this prior punishes the violation of this interval
    linearly by enclosing the interval in a pair of relus.

    :param x_reconstruct: (torch.tensor) The batch being reconstructed.
    :param dataset: (BaseDataset) The dataset with respect to which the reconstruction is happening.
    :param labels: (torch.tensor) The labels belonging to the batch.
    :param softmax_trick: (bool) Set to True if the softmax trick is already applied to the datapoint.
    :param T: (float) A temperature parameter to control any softmaxes involved in the calculation.
    :param device: (str) The device on which the tensors are stored.
    :return: (torch.tensor) The calculated prior cost to be added onto the loss.
    """
    if device is None:
        device = x_reconstruct.device

    batch_size = x_reconstruct.size()[0]
    prior_cost = torch.as_tensor([0.0], device=device)

    for key, (feature_type, feature_indices) in dataset.train_feature_index_map.items():
        if feature_type == 'cont':
            if dataset.standardized:
                lb, rb = float(dataset.standardized_continuous_bounds[key][0]), \
                         float(dataset.standardized_continuous_bounds[key][1])
            else:
                lb, rb = float(dataset.continuous_bounds[key][0]), float(dataset.continuous_bounds[key][1])
            prior_cost += F.relu(lb - x_reconstruct[:, feature_indices].flatten()).sum() / batch_size
            prior_cost += F.relu(x_reconstruct[:, feature_indices].flatten() - rb).sum() / batch_size
        else:
            continue

    return prior_cost


def _categorical_mean_field_jensen_shannon_prior(x_reconstruct, dataset, labels=None, softmax_trick=None, T=None, device=None):
    """
    Knowing the mean-field distribution of the categorical features in the dataset (estimated from the dataset samples),
    this prior assigns a cost by the Jensen-Shannon divergence between the mean-field categorical distributions of the
    reconstructed batch and the true one. To estimate the distribution of the batch, on each line we use a softmax of
    temperature T.

    :param x_reconstruct: (torch.tensor) The batch being reconstructed.
    :param dataset: (BaseDataset) The dataset with respect to which the reconstruction is happening.
    :param labels: (torch.tensor) The labels belonging to the batch.
    :param softmax_trick: (bool) Set to True if the softmax trick is already applied to the datapoint.
    :param T: (float) A temperature parameter to control any softmaxes involved in the calculation.
    :param device: (str) The device on which the tensors are stored.
    :return: (torch.tensor) The calculated prior cost to be added onto the loss.
    """
    if device is None:
        device = x_reconstruct.device
    if T is None:
        T = 0.01
    prior_cost = torch.as_tensor([0.0], device=device)
    for key, (feature_type, feature_indices) in dataset.train_feature_index_map.items():
        if feature_type == 'cont':
            continue
        else:
            true_probabilities = torch.tensor(dataset.categorical_histograms[key] + 1e-8, device=device,
                                              dtype=torch.float32)
            if len(feature_indices) == 1:
                # non-rounded thresholding to label 1 for binary features
                sig_to_1 = torch.sigmoid(x_reconstruct[:, feature_indices])
                expanded_feature = torch.cat([1. - sig_to_1, sig_to_1], dim=1)
                rec_normalized_probabilities = F.softmax(expanded_feature / T, dim=1).float()
            else:
                rec_normalized_probabilities = F.softmax(x_reconstruct[:, feature_indices] / T, dim=1).float()
            rec_normalized_probabilities = torch.true_divide(rec_normalized_probabilities.sum(dim=0),
                                                             rec_normalized_probabilities.sum()).flatten()
            M = 0.5 * (true_probabilities + rec_normalized_probabilities)
            log_true_M = torch.log(torch.true_divide(true_probabilities, M))
            log_rec_M = torch.log(torch.true_divide(rec_normalized_probabilities, M))
            prior_cost += 0.5 * (true_probabilities.T @ log_true_M + rec_normalized_probabilities.T @ log_rec_M)

    return prior_cost


def _theoretical_optimal_prior(x_reconstruct, dataset, labels, softmax_trick, T=1., device=None):
    """
    Calculates the theoretical prior for the bayesian optimal adversary. Works only if the dataset has a closed form
    implementation of the theoretical nll, i.e. only synthetic datasets are supported.

    :param x_reconstruct: (torch.tensor) The batch being reconstructed.
    :param dataset: (BaseDataset) The dataset with respect to which the reconstruction is happening.
    :param labels: (torch.tensor) The labels belonging to the batch.
    :param softmax_trick: (bool) Set to True if the softmax trick is already applied to the datapoint.
    :param T: (float) A temperature parameter to control any softmaxes involved in the calculation.
    :param device: (str) The device on which the tensors are stored.
    :return: (torch.tensor) The negative log likelihood cost of the batch.
    """
    if device is None:
        device = x_reconstruct.device
    mean, std = dataset.mean, dataset.std

    # apply softmax to the categoricals to get something that resembles one-hots if not already applied
    if not softmax_trick:
        x_rec = categorical_softmax(x_reconstruct, dataset, T)
    else:
        x_rec = x_reconstruct * 1
    # de-standardize to have the real distributions
    if dataset.standardized:
        x_r= std * x_rec + mean
    else:
        x_r = x_rec * 1

    nll = dataset.calculate_nll_of_batch(x_r, labels)
    return nll


def _theoretical_typicality_prior(x_reconstruct, dataset, labels, softmax_trick, T=1., device=None):
    """
    Implements a prior relying on the squared error between the negative log likelihood of a batch and the entropy of
    the dataset. The implementation relies on the assumption that the NLL and the entropy can be calculated exactly
    according to the true distribution and hence this function is only suitable for use for synthetic datasets.

     :param x_reconstruct: (torch.tensor) The batch being reconstructed.
    :param dataset: (BaseDataset) The dataset with respect to which the reconstruction is happening.
    :param labels: (torch.tensor) The labels belonging to the batch.
    :param softmax_trick: (bool) Set to True if the softmax trick is already applied to the datapoint.
    :param T: (float) A temperature parameter to control any softmaxes involved in the calculation.
    :param device: (str) The device on which the tensors are stored.
    :return: (torch.tensor) The square error between the NLL of a batch and the entropy of the dataset.
    """
    if device is None:
        device = x_reconstruct.device
    mean, std = dataset.mean, dataset.std

    # apply softmax to the categoricals to get something that resembles one-hots if not already applied
    if not softmax_trick:
        x_rec = categorical_softmax(x_reconstruct, dataset, T)
    else:
        x_rec = x_reconstruct * 1
    # de-standardize to have the real distributions
    if dataset.standardized:
        x_r= std * x_rec + mean
    else:
        x_r = x_rec * 1

    prior_cost = torch.as_tensor([0.0], device=device)

    # get the negative log likelihood of the batch and the entropy of the dataset
    nll = dataset.calculate_nll_of_batch(x_r, labels)
    entropy_0, entropy_1 = dataset.entropy_0.clone().detach(), dataset.entropy_1.clone().detach()

    # reweight the entropy per label
    zero_label_n = len((labels == 0).nonzero(as_tuple=True)[0])
    one_label_n = len((labels == 1).nonzero(as_tuple=True)[0])
    batch_size = x_reconstruct.size()[0]
    weighted_entropy = (zero_label_n * entropy_0 + one_label_n * entropy_1) / batch_size

    # calculate the squared deviation from the global entropy
    prior_cost += (weighted_entropy - nll).pow(2)

    return prior_cost


def _theoretical_marginal_prior(x_reconstruct, dataset, labels, softmax_trick, T=1., device=None):
    """
    Calculates the theoretical marginal prior for the mean field bayesian optimal adversary. Works only if the dataset
    has a closed form implementation of the theoretical nll, i.e. only synthetic datasets are supported.

    :param x_reconstruct: (torch.tensor) The batch being reconstructed.
    :param dataset: (BaseDataset) The dataset with respect to which the reconstruction is happening.
    :param labels: (torch.tensor) The labels belonging to the batch.
    :param softmax_trick: (bool) Set to True if the softmax trick is already applied to the datapoint.
    :param T: (float) A temperature parameter to control any softmaxes involved in the calculation.
    :param device: (str) The device on which the tensors are stored.
    :return: (torch.tensor) The negative log likelihood cost of the batch.
    """
    if device is None:
        device = x_reconstruct.device
    mean, std = dataset.mean, dataset.std

    # apply softmax to the categoricals to get something that resembles one-hots if not already applied
    if not softmax_trick:
        x_rec = categorical_softmax(x_reconstruct, dataset, T)
    else:
        x_rec = x_reconstruct * 1
    # de-standardize to have the real distributions
    if dataset.standardized:
        x_r= std * x_rec + mean
    else:
        x_r = x_rec * 1

    nll = dataset.calculate_marginal_nll_of_batch(x_r, labels)
    return nll


def _theoretical_marginal_typicality_prior(x_reconstruct, dataset, labels, softmax_trick, T=1., device=None):
    """
    Implements a prior relying on the squared error between the mean field negative log likelihood of a batch and the
    mean field entropy of the dataset. The implementation relies on the assumption that the marginal NLL and the
    marginal entropy can be calculated exactly according to the true distribution and hence this function is only
    suitable for use for synthetic datasets.

    :param x_reconstruct: (torch.tensor) The batch being reconstructed.
    :param dataset: (BaseDataset) The dataset with respect to which the reconstruction is happening.
    :param labels: (torch.tensor) The labels belonging to the batch.
    :param softmax_trick: (bool) Set to True if the softmax trick is already applied to the datapoint.
    :param T: (float) A temperature parameter to control any softmaxes involved in the calculation.
    :param device: (str) The device on which the tensors are stored.
    :return: (torch.tensor) The square error between the NLL of a batch and the entropy of the dataset.
    """
    if device is None:
        device = x_reconstruct.device
    mean, std = dataset.mean, dataset.std

    # apply softmax to the categoricals to get something that resembles one-hots if not already applied
    if not softmax_trick:
        x_rec = categorical_softmax(x_reconstruct, dataset, T)
    else:
        x_rec = x_reconstruct * 1
    # de-standardize to have the real distributions
    if dataset.standardized:
        x_r= std * x_rec + mean
    else:
        x_r = x_rec * 1

    prior_cost = torch.as_tensor([0.0], device=device)

    # get the negative log likelihood of the batch and the entropy of the dataset
    nll = dataset.calculate_marginal_nll_of_batch(x_r, labels)
    entropy_0, entropy_1 = dataset.marginal_entropy_0.clone().detach(), dataset.marginal_entropy_1.clone().detach()

    # reweight the entropy per label
    zero_label_n = len((labels == 0).nonzero(as_tuple=True)[0])
    one_label_n = len((labels == 1).nonzero(as_tuple=True)[0])
    batch_size = x_reconstruct.size()[0]
    weighted_entropy = (zero_label_n * entropy_0 + one_label_n * entropy_1) / batch_size

    # calculate the squared deviation from the global entropy
    prior_cost += (weighted_entropy - nll).pow(2)

    return prior_cost
