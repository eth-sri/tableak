import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import numpy as np


def get_acc_and_bac(network, X, y):
    """
    Returns the accuracy and the balanced accuracy score of a given neural network on the dataset X, y.

    :param network: (nn.Module) The torch model of which we wish to measure the accuracy of.
    :param X: (torch.tensor) The input features.
    :param y: (torch.tensor) The true labels corresponding to the input features.
    :return: (tuple) The accuracy score and the balanced accuracy score.
    """
    with torch.no_grad():
        _, all_pred = torch.max(network(X).data, 1)
        acc = accuracy_score(np.array(y.cpu()), np.array(all_pred.cpu()))
        bac = balanced_accuracy_score(np.array(y.cpu()), np.array(all_pred.cpu()))
    return acc, bac


def feature_wise_accuracy_score(true_data, reconstructed_data, tolerance_map, train_features):
    """
    Calculates the categorical accuracy and in-tolerance-interval accuracy for continuous features per feature.

    :param true_data: (np.ndarray) The true/reference mixed-type feature vector.
    :param reconstructed_data: (np.ndarray) The reconstructed mixed-type feature vector.
    :param tolerance_map: (list or np.ndarray) A list with the same length as a single datapoint. Each entry in the list
        corresponding to a numerical feature in the data should contain a floating point value marking the
        reconstruction tolerance for the given feature. At each position corresponding to a categorical feature the list
        has to contain the entry 'cat'.
    :param train_features: (dict) A dictionary of the feature names per column.
    :return: (dict) A dictionary with the features and their corresponding error.
    """
    feature_errors = {}
    for feature_name, true_feature, reconstructed_feature, tol in zip(train_features.keys(), true_data, reconstructed_data, tolerance_map):
        if tol == 'cat':
            feature_errors[feature_name] = 0 if str(true_feature) == str(reconstructed_feature) else 1
        else:
            feature_errors[feature_name] = 0 if (float(true_feature) - tol <= float(reconstructed_feature) <= float(true_feature) + tol) else 1
    return feature_errors


def batch_feature_wise_accuracy_score(true_data, reconstructed_data, tolerance_map, train_features):
    """

    :param true_data: (np.ndarray) The true/reference mixed-type feature matrix.
    :param reconstructed_data:
    :param tolerance_map: (list or np.ndarray) A list with the same length as a single datapoint. Each entry in the list
        corresponding to a numerical feature in the data should contain a floating point value marking the
        reconstruction tolerance for the given feature. At each position corresponding to a categorical feature the list
        has to contain the entry 'cat'.
    :param train_features: (dict) A dictionary of the feature names per column.
    :return: (dict) A dictionary with the features and their corresponding error.
    """

    assert len(true_data.shape) == 2, 'This function requires a batch of data'

    batch_size = true_data.shape[0]
    feature_errors = {feature_name: 0 for feature_name in train_features.keys()}
    for true_data_line, reconstructed_data_line in zip(true_data, reconstructed_data):
        line_feature_errors = feature_wise_accuracy_score(true_data_line, reconstructed_data_line, tolerance_map, train_features)
        for feature_name in feature_errors.keys():
            feature_errors[feature_name] += 1/batch_size * line_feature_errors[feature_name]
    return feature_errors


def _categorical_accuracy_continuous_tolerance_score(true_data, reconstructed_data, tolerance_map, detailed=False):
    """

    :param true_data: (np.ndarray) The true/reference mixed-type feature vector.
    :param reconstructed_data:
    :param tolerance_map: (list or np.ndarray) A list with the same length as a single datapoint. Each entry in the list
        corresponding to a numerical feature in the data should contain a floating point value marking the
        reconstruction tolerance for the given feature. At each position corresponding to a categorical feature the list
        has to contain the entry 'cat'.
    :param detailed: (bool) Set to True if you want additionally to calculate the error rate induced by categorical
        features and by continuous features separately.
    :return: (float or tuple of floats) The accuracy score with respect to the given tolerance of the reconstruction.
        If the flag 'detailed' is set to True the reconstruction errors of the categorical and the continuous features
        are returned separately.
    """
    cat_score = 0
    cont_score = 0
    num_cats = 0
    num_conts = 0

    for true_feature, reconstructed_feature, tol in zip(true_data, reconstructed_data, tolerance_map):
        if tol == 'cat':
            cat_score += 0 if str(true_feature) == str(reconstructed_feature) else 1
            num_cats += 1
        elif not isinstance(tol, str):
            cont_score += 0 if (float(true_feature) - tol <= float(reconstructed_feature) <= float(true_feature) + tol) else 1
            num_conts += 1
        else:
            raise TypeError('The tolerance map has to either contain numerical values to define tolerance intervals or '
                            'the string >cat< to mark the position of a categorical feature.')
    if detailed:
        if num_cats < 1:
            num_cats = 1
        if num_conts < 1:
            num_conts = 1
        return (cat_score + cont_score)/(num_cats + num_conts), cat_score/num_cats, cont_score/num_conts
    else:
        return (cat_score + cont_score)/(num_cats + num_conts)


def categorical_accuracy_continuous_tolerance_score(true_data, reconstructed_data, tolerance_map, detailed=False):
    """
    Calculates an error score between the true mixed-type datapoint and a reconstructed mixed-type datapoint. For each
    categorical feature we count a 0-1 error by the rule of the category being reconstructed correctly. For each
    continuous feature we count a 0-1 error by the rule of the continuous variable being reconstructed within a
    symmetric tolerance interval around the true value. The tolerance parameters are set by 'tolerance_map'.

    :param true_data: (np.ndarray) The true/reference mixed-type feature vector or matrix if comprising more than
        datapoint.
    :param reconstructed_data: (np.ndarray) The reconstructed mixed-type feature vector/matrix.
    :param tolerance_map: (list or np.ndarray) A list with the same length as a single datapoint. Each entry in the list
        corresponding to a numerical feature in the data should contain a floating point value marking the
        reconstruction tolerance for the given feature. At each position corresponding to a categorical feature the list
        has to contain the entry 'cat'.
    :param detailed: (bool) Set to True if you want additionally to calculate the error rate induced by categorical
        features and by continuous features separately.
    :return: (float or tuple of floats) The accuracy score with respect to the given tolerance of the reconstruction.
        If a batch of data is given, then the average accuracy of the batch is returned. Additionally, if the flag
        'detailed' is set to True the reconstruction errors of the categorical and the continuous features are returned
        separately.
    """
    assert true_data.shape == reconstructed_data.shape
    score = 0
    cat_score = 0
    cont_score = 0
    if len(true_data.shape) > 1:
        for true_data_line, reconstructed_data_line in zip(true_data, reconstructed_data):
            assert len(true_data_line) == len(tolerance_map)
            if detailed:
                scores = _categorical_accuracy_continuous_tolerance_score(true_data_line, reconstructed_data_line,
                                                                          tolerance_map, True)
                score += 1/true_data.shape[0] * scores[0]
                cat_score += 1 / true_data.shape[0] * scores[1]
                cont_score += 1 / true_data.shape[0] * scores[2]
            else:
                score += 1/true_data.shape[0] * _categorical_accuracy_continuous_tolerance_score(true_data_line,
                                                                                                 reconstructed_data_line,
                                                                                                 tolerance_map)
    else:
        assert len(true_data) == len(tolerance_map)
        if detailed:
            scores = _categorical_accuracy_continuous_tolerance_score(true_data, reconstructed_data, tolerance_map, True)
            score += scores[0]
            cat_score += scores[1]
            cont_score += scores[2]
        else:
            score = _categorical_accuracy_continuous_tolerance_score(true_data, reconstructed_data, tolerance_map)

    if detailed:
        return score, cat_score, cont_score
    else:
        return score
