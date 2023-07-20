import numpy as np


def to_numeric(data: np.ndarray, features: dict, label: str = '', single_bit_binary: bool = False) -> np.ndarray:
    """
    Takes an array of categorical and continuous mixed type data and encodes it in numeric data. Categorical features of
    more than 2 categories are turned into a one-hot vector and continuous features are kept standing. The description
    of each feature has to be provided in the dictionary 'features'. The implementation assumes python 3.7 or higher as
    it requires the dictionary to be ordered.

    :param data: (np.ndarray) The mixed type input vector or matrix of more datapoints.
    :param features: (dict) A dictionary containing each feature of data's description. The dictionary's items have to
        be of the form: ('name of the feature', ['list', 'of', 'categories'] if the feature is categorical else None).
        The dictionary has to be ordered in the same manner as the features appear in the input array.
    :param label: (str) The name of the feature storing the label.
    :param single_bit_binary: (bool) Toggle to encode binary features in a single bit instead of a 2-component 1-hot.
    :return: (np.ndarray) The fully numeric data encoding.
    """
    num_columns = []
    n_features = 0
    for i, key in enumerate(list(features.keys())):
        if features[key] is None:
            num_columns.append(np.reshape(data[:, i], (-1, 1)))
        elif len(features[key]) == 2 and (single_bit_binary or key == label):
            num_columns.append(np.reshape(np.array([int(str(val) == str(features[key][-1])) for val in data[:, i]]), (-1, 1)))
        else:
            sub_matrix = np.zeros((data.shape[0], len(features[key])))
            col_one_place = [np.argwhere(np.array(features[key]) == str(val)) for val in data[:, i]]
            for row, one_place in zip(sub_matrix, col_one_place):
                row[one_place] = 1
            num_columns.append(sub_matrix)
        n_features += num_columns[-1].shape[-1]
    pointer = 0
    num_data = np.zeros((data.shape[0], n_features))
    for column in num_columns:
        end = pointer + column.shape[1]
        num_data[:, pointer:end] = column
        pointer += column.shape[1]
    return num_data.astype(np.float32)


def to_categorical(data: np.ndarray, features: dict, label: str = '', single_bit_binary=False, nearest_int=True) -> np.ndarray:
    """
    Takes an array of matrix of more datapoints in numerical form and turns it back into mixed type representation.
    The requirement for a successful reconstruction is that the numerical data was generated following the same feature
    ordering as provided here in the dictionary 'features'.

    :param data: (np.ndarray) The numerical data to be converted into mixed-type.
    :param features: (dict) A dictionary containing each feature of data's description. The dictionary's items have to
        be of the form: ('name of the feature', ['list', 'of', 'categories'] if the feature is categorical else None).
        The dictionary has to be ordered in the same manner as the features appear in the input array.
    :param label: (str) The name of the feature storing the label.
    :param single_bit_binary: (bool) Toggle if the binary features have been encoded in a single bit instead of a
        2-component 1-hot.
    :param nearest_int: (bool) Toggle to round to nearest integer.
    :return: (np.ndarray) The resulting mixed type data array.
    """
    cat_columns = []
    pointer = 0
    for key in list(features.keys()):
        if features[key] is None:
            if nearest_int:
                cat_columns.append(np.floor(data[:, pointer] + 0.5))
            else:
                cat_columns.append(data[:, pointer])
            pointer += 1
        elif len(features[key]) == 2 and (single_bit_binary or key == label):
            cat_columns.append([features[key][max(min(int(val + 0.5), 1), 0)] for val in data[:, pointer]])
            pointer += 1
        else:
            start = pointer
            end = pointer + len(features[key])
            hot_args = np.argmax(data[:, start:end], axis=1)
            cat_columns.append([features[key][arg] for arg in hot_args])
            pointer = end
    cat_array = None
    for cat_column in cat_columns:
        if cat_array is None:
            cat_array = np.reshape(np.array(cat_column), (data.shape[0], -1))
        else:
            cat_array = np.concatenate((cat_array, np.reshape(np.array(cat_column), (data.shape[0], -1))), axis=1)
    return cat_array
