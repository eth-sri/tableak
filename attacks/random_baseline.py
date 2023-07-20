import sys
sys.path.append("..")
import numpy as np
from utils import batch_feature_wise_accuracy_score, match_reconstruction_ground_truth
import torch


def calculate_random_baseline(dataset, recover_batch_sizes, tolerance_map, n_samples=10, mode='uniform', device='cpu'):
    """
    Calculates random baselines for out batch inversion experiments by simply guessing a batch according to a
    distribution without taking a look at the gradient. We have currently three modes of operation available:
        1. 'uniform': Each feature is samples according to a uniform distribution over its support. For categorical
            features their support is the span of all categories, for continuous features we define the support as the
            continuous interval between the minimum value and the maximum value of the given feature.
        2. 'cat_empirical': The continuous features are sampled as in 'uniform' mode, but the categorical features are
            sampled according to their empirical relative frequency in the dataset.
        3. 'all_empirical': All features are sampled according to their empirical relative frequency in the dataset.

    :param dataset: (datasets.BaseDataset) An instantiated child of the datasets.BaseDataset object.
    :param recover_batch_sizes: (list) A list of all batch sizes we want to estimate the random recovery error for.
    :param tolerance_map: (list) The tolerance map required to calculate the error between the guessed and the true
        batch.
    :param n_samples: (int) The number of monte carlo samples to estimate the mean and the standard deviation of the
        random reconstruction error.
    :param mode: (str) The mode/set of assumptions for the sampling process. For details, see the main body of the
        documentation.
    :param device: (str) The device on which the tensors in the dataset are located. Not used for now.
    :return: (np.ndarray) The mean and the standard deviation reconstruction error for the randomly guessed batches for
        each batch size in 'recover_batch_sizes'. The dimensions are (len(recover_batch_sizes), 3, 2); where the middle
        dimension contains the error data for 0: complete batch error, 1: categorical feature error, 2: continuous
        feature error, each as (mean, std).
    """
    assert mode in ['uniform', 'cat_empirical', 'all_empirical']
    Xtrain, Xtest = dataset.get_Xtrain(), dataset.get_Xtest()
    X = torch.cat([Xtrain, Xtest], dim=0)
    X = dataset.decode_batch(X, standardized=dataset.standardized)
    random_reconstruction_error = np.zeros((len(recover_batch_sizes), 3 + len(dataset.train_features), 5))

    for j, recover_batch_size in enumerate(recover_batch_sizes):
        print(recover_batch_size, end='\r')
        recon_score_all = []
        recon_score_cat = []
        recon_score_cont = []
        per_feature_recon_scores = []
        for sample in range(n_samples):
            target_batch_cat = X[np.random.randint(0, X.shape[0], recover_batch_size)]  # sample a batch to reconstruct
            random_batch = np.zeros((recover_batch_size, X.shape[1]), dtype='object')
            for i, (data_col, (key, feature)) in enumerate(zip(X.T, dataset.train_features.items())):
                if feature is None:
                    if mode != 'all_empirical':
                        lower, upper = min(data_col.astype(np.float32)), max(data_col.astype(np.float32))
                        random_batch[:, i] = np.random.randint(lower, upper + 1, recover_batch_size)
                    else:
                        lower, upper = dataset.continuous_bounds[key]
                        cont_histogram = dataset.cont_histograms[key]
                        if len(cont_histogram) < 100 and not str(dataset).startswith('Law'):
                            feature_range = np.arange(lower, upper+1)
                        else:
                            delta = (upper - lower) / len(cont_histogram)
                            feature_range = np.array([lower + i*delta for i in range(len(cont_histogram))])
                        random_batch[:, i] = np.random.choice(feature_range, 1, p=dataset.cont_histograms[key]).item()
                else:
                    p = None if mode == 'uniform' else dataset.categorical_histograms[key]
                    random_batch[:, i] = np.random.choice(feature, recover_batch_size, p=p)

            # perform the Hungarian algorithm to align the reconstructed batch with the ground truth and calculate the
            # mean reconstruction score
            batch_recon_cat, batch_cost_all, batch_cost_cat, batch_cost_cont = match_reconstruction_ground_truth(
                target_batch_cat, random_batch, tolerance_map)
            recon_score_all.append(np.mean(batch_cost_all))
            recon_score_cat.append(np.mean(batch_cost_cat))
            recon_score_cont.append(np.mean(batch_cost_cont))

            # calculate the reconstruction accuracy also per feature
            per_feature_recon_scores.append(
                batch_feature_wise_accuracy_score(target_batch_cat, batch_recon_cat, tolerance_map,
                                                  dataset.train_features))

        random_reconstruction_error[j, 0] = np.mean(recon_score_all), np.std(recon_score_all), \
                                            np.median(recon_score_all), np.min(recon_score_all), np.max(recon_score_all)
        random_reconstruction_error[j, 1] = np.mean(recon_score_cat), np.std(recon_score_cat), \
                                            np.median(recon_score_cat), np.min(recon_score_cat), np.max(recon_score_cat)
        random_reconstruction_error[j, 2] = np.mean(recon_score_cont), np.std(recon_score_cont), \
                                            np.median(recon_score_cont), np.min(recon_score_cont), np.max(recon_score_cont)

        # aggregate and add the feature-wise data as well
        for k, feature_name in enumerate(dataset.train_features.keys()):
            curr_feature_errors = [feature_errors[feature_name] for feature_errors in per_feature_recon_scores]
            random_reconstruction_error[j, 3 + k] = np.mean(curr_feature_errors), np.std(curr_feature_errors), np.median(
                curr_feature_errors), np.min(curr_feature_errors), np.max(curr_feature_errors)

    return random_reconstruction_error


def calculate_synthetic_random_baseline(reconstruction_batch_sizes, tolerance_map, dataset, n_samples):
    """
    A function that calculates the true random baseline for synthetic datasets.

    :param reconstruction_batch_sizes: (list) A list of all batch sizes we want to estimate the random recovery error
        for.
    :param tolerance_map: (list) The tolerance map required to calculate the error between the guessed and the true
        batch.
    :param dataset: (datasets.BaseDataset) An instantiated child of the datasets.BaseDataset object.
    :param n_samples: (int) The number of monte carlo samples to estimate the mean and the standard deviation of the
        random reconstruction error.
    :return: (np.ndarray) The mean and the standard deviation reconstruction error for the randomly guessed batches for
        each batch size in 'recover_batch_sizes'. The dimensions are (len(recover_batch_sizes), 3, 2); where the middle
        dimension contains the error data for 0: complete batch error, 1: categorical feature error, 2: continuous
        feature error, each as (mean, std).
    """
    Xtrain, Xtest = dataset.get_Xtrain(), dataset.get_Xtest()
    ytrain, ytest = dataset.get_ytrain(), dataset.get_ytest()
    X = torch.cat([Xtrain, Xtest], dim=0)
    y = torch.cat([ytrain, ytest], dim=0).numpy()
    X = dataset.decode_batch(X, standardized=dataset.standardized)

    original_class_nums = dataset.class_nums

    random_reconstruction_error = np.zeros((len(reconstruction_batch_sizes), 3 + len(dataset.train_features), 5))

    for i, reconstruction_batch_size in enumerate(reconstruction_batch_sizes):
        recon_score_all = []
        recon_score_cat = []
        recon_score_cont = []
        per_feature_recon_scores = []

        for _ in range(n_samples):
            # sample a random batch
            random_indices = np.random.randint(0, len(X), reconstruction_batch_size)
            true_batch, true_batch_labels = X[random_indices], y[random_indices]
            class_nums = [reconstruction_batch_size - sum(true_batch_labels), sum(true_batch_labels)]
            dataset.class_nums = class_nums
            random_batch, _ = dataset.build_dataset()

            # perform the Hungarian algorithm to align the reconstructed batch with the ground truth and calculate the
            # mean reconstruction score
            random_batch, batch_cost_all, batch_cost_cat, batch_cost_cont = match_reconstruction_ground_truth(
                true_batch, random_batch, tolerance_map)
            recon_score_all.append(np.mean(batch_cost_all))
            recon_score_cat.append(np.mean(batch_cost_cat))
            recon_score_cont.append(np.mean(batch_cost_cont))

            # calculate the reconstruction accuracy also per feature
            per_feature_recon_scores.append(
                batch_feature_wise_accuracy_score(true_batch, random_batch, tolerance_map, dataset.train_features))

        random_reconstruction_error[i, 0] = np.mean(recon_score_all), np.std(recon_score_all), \
                                            np.median(recon_score_all), np.min(recon_score_all), np.max(recon_score_all)
        random_reconstruction_error[i, 1] = np.mean(recon_score_cat), np.std(recon_score_cat), \
                                            np.median(recon_score_cat), np.min(recon_score_cat), np.max(recon_score_cat)
        random_reconstruction_error[i, 2] = np.mean(recon_score_cont), np.std(recon_score_cont), \
                                            np.median(recon_score_cont), np.min(recon_score_cont), np.max(
            recon_score_cont)

        # aggregate and add the feature-wise data as well
        for k, feature_name in enumerate(dataset.train_features.keys()):
            curr_feature_errors = [feature_errors[feature_name] for feature_errors in per_feature_recon_scores]
            random_reconstruction_error[i, 3 + k] = np.mean(curr_feature_errors), np.std(
                curr_feature_errors), np.median(
                curr_feature_errors), np.min(curr_feature_errors), np.max(curr_feature_errors)

    dataset.class_nums = original_class_nums

    return random_reconstruction_error
