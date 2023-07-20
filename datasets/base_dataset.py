import sys
sys.path.append("..")
from abc import ABC, abstractmethod
import torch
from utils import to_categorical, to_numeric
import numpy as np


class BaseDataset(ABC):

    @abstractmethod
    def __init__(self, name, device, random_state=42):
        self.name = name
        self.device = device
        self.random_state = random_state
        self.split_status = 'train'
        self.standardized = False

        # To be assigned by the concrete dataset
        self.mean, self.std = None, None
        self.Xtrain, self.ytrain = None, None
        self.Xtest, self.ytest = None, None
        self.feature_data, self.labels = self.Xtrain, self.ytrain
        self.num_features = None
        self.features, self.train_features = None, None
        self.index_maps_created = False
        self.histograms_and_continuous_bounds_calculated = False
        self.gmm_parameters_loaded = False
        self.label = ''
        self.single_bit_binary = False

    def __str__(self):
        return self.name + f' Dataset: {self.split_status}'

    def __getitem__(self, item):
        return self.feature_data[item], self.labels[item]

    def __len__(self):
        return self.labels.size()[0]

    def _assign_split(self, split):
        """
        Private method to load data into 'self.features' and 'self.labels' if the object is desired to be used directly.

        :param split: (str) Which split to assign to 'self.features' and 'self.labels'. The available splits are
            ['train', 'test'], meaning we can either assign the training set or the testing set there.
        :return: None
        """
        self.split_status = split
        if split == 'train':
            self.feature_data, self.labels = self.Xtrain, self.ytrain
        elif split == 'test':
            self.feature_data, self.labels = self.Xtest, self.ytest
        else:
            raise ValueError('Unsupported split')

    def train(self):
        self._assign_split('train')

    def test(self):
        self._assign_split('test')

    def shuffle(self):
        """
        Reshuffles the splits.

        :return: None
        """
        train_shuffle_indices = torch.randperm(self.Xtrain.size()[0]).to(self.device)
        test_shuffle_indices = torch.randperm(self.Xtest.size()[0]).to(self.device)
        self.Xtrain, self.ytrain = self.Xtrain[train_shuffle_indices], self.ytrain[train_shuffle_indices]
        self.Xtest, self.ytest = self.Xtest[test_shuffle_indices], self.ytest[test_shuffle_indices]

    def get_Xtrain(self):
        """
        Returns a detached copy of the training dataset.

        :return: (torch.tensor)
        """
        return self.Xtrain.clone().detach()

    def get_ytrain(self):
        """
        Returns a detached copy of the training labels.

        :return: (torch.tensor)
        """
        return self.ytrain.clone().detach()

    def get_Xtest(self):
        """
        Returns a detached copy of the test dataset.

        :return: (torch.tensor)
        """
        return self.Xtest.clone().detach()

    def get_ytest(self):
        """
        Returns a detached copy of the test labels.

        :return: (torch.tensor)
        """
        return self.ytest.clone().detach()

    def _calculate_mean_std(self):
        """
        Private method to calculate the mean and the standard deviation of the underlying dataset.

        :return: None
        """
        if not self.standardized:  # we do not want to lose the original standardization parameters by overwriting them
            joint_data = torch.cat([self.Xtrain, self.Xtest])
            self.mean = torch.mean(joint_data, dim=0)
            self.std = torch.std(joint_data, dim=0)
            # to avoid divisions by zero and exploding features in constant or nearly constant columns we set standard
            # deviations of zero to one
            zero_stds = torch.nonzero(self.std == 0, as_tuple=False).flatten()
            self.std[zero_stds] = 1.0

    def standardize(self, batch=None, mode='both'):
        """
        Standardizes the given data (0 mean and 1 variance). It works in three modes: 'batch', 'split', and 'both'. In
        case of 'batch' we standardize a given batch of data by the global statistics of the dataset. In case of 'both'
        we simply standardize the whole underlying dataset, i.e. self.Xtrain and self.Xtest will be standardized. In
        case of 'split' we only standardize the data currently loaded into self.features.

        :param batch:
        :param mode:
        :return: None
        """
        if batch is not None:
            mode = 'batch'
        if mode == 'split':
            self.feature_data = (self.features - self.mean) / self.std
        elif mode == 'both':
            if not self.standardized:
                self.standardized = True
                self.Xtrain, self.Xtest = (self.Xtrain - self.mean) / self.std, (self.Xtest - self.mean) / self.std
                self._assign_split(self.split_status)
        elif mode == 'batch':
            return (batch - self.mean) / self.std
        else:
            raise ValueError('Unsupported mode')

    def de_standardize(self, batch=None, mode='both'):
        if batch is not None:
            mode = 'batch'
        if mode == 'split':
            self.feature_data = self.features * self.std + self.std
        elif mode == 'both':
            if self.standardized:
                self.standardized = False
                self.Xtrain, self.Xtest = self.Xtrain * self.std + self.mean, self.Xtest * self.std + self.mean
                self._assign_split(self.split_status)
        elif mode == 'batch':
            return batch * self.std + self.mean
        else:
            raise ValueError('Unsupported mode')

    def positive_prevalence(self):
        """
        In case of a binary classification task this function calculates the prevalence of the positive label (1). This
        data is useful when assessing the degree of class imbalance.

        :return: (tuple) Prevalence of the positive class in the training set and in the testing set.
        """
        # TODO: for now this method only makes sense for binary classification tasks
        return torch.true_divide(self.ytrain.sum(), self.ytrain.size()[0]), torch.true_divide(self.ytest.sum(), self.ytest.size()[0])

    def decode_batch(self, batch, standardized=True):
        """
        Given a batch of numeric data, this function turns that batch back into the interpretable mixed representation.

        :param batch: (torch.tensor) A batch of data to be decoded according to the features and statistics of the
            underlying dataset.
        :param standardized: (bool) Flag if the batch had been standardized or not.
        :return: (np.ndarray) The batch decoded into mixed representation as the dataset is out of the box.
        """
        if standardized:
            batch = self.de_standardize(batch)
        return to_categorical(batch.clone().detach().cpu(), self.train_features, single_bit_binary=self.single_bit_binary)

    def encode_batch(self, batch, standardize=True):
        """
        Given a batch of mixed type data (np.ndarray on the cpu) we return a numerically encoded batch (torch tensor on
        the dataset device).

        :param batch: (np.ndarray) The mixed type data we wish to convert to numeric.
        :param standardize: (bool) Toggle if the numeric data is to be standardized or not.
        :return: (torch.tensor) The numeric encoding of the data as a torch tensor.
        """
        batch = torch.tensor(to_numeric(batch, self.train_features, label=self.label, single_bit_binary=self.single_bit_binary), device=self.device)
        if standardize:
            batch = self.standardize(batch)
        return batch

    def project_batch(self, batch, standardized=True):
        """
        Given a batch of numeric fuzzy data, this returns its projected encoded counterpart.

        :param batch: (torch.tensor) The fuzzy data to be projected.
        :param standardized: (bool) Mark if the fuzzy data is standardized or not. The data will be returned in the same
            way.
        :return: (torch.tensor) The projected data.
        """
        return self.encode_batch(self.decode_batch(batch, standardized=standardized), standardize=standardized)

    def _create_index_maps(self):
        """
        A private method that creates easy access indexing tools for other methods.

        :return: None
        """
        # check if the feature map has already been assigned
        assert self.features is not None, 'Instantiate a dataset with a feature map'

        # register the type of the feature and the positions of all numerical features corresponding to this feature
        pointer = 0
        self.feature_index_map = {}
        for val, key in zip(self.features.values(), self.features.keys()):
            if val is None or (len(val) == 2 and self.single_bit_binary):
                index_list = [pointer]
                pointer += 1
            else:
                index_list = [pointer + i for i in range(len(val))]
                pointer += len(val)
            im = 'cont' if val is None else 'cat'
            self.feature_index_map[key] = (im, index_list) if key != self.label else (im, index_list[0])  # binary labels

        # for ease of use, create the one just for the X part of the data
        self.train_feature_index_map = {key: self.feature_index_map[key] for key in self.train_features.keys()}

        # for ease of use make the non-numerically encoded feature positions also accessible by type
        index_map = np.array(['cont' if val is None else 'cat' for val in self.features.values()])
        self.cont_indices = np.argwhere(index_map == 'cont').flatten()
        self.cat_indices = np.argwhere(index_map == 'cat').flatten()
        # this makes sense only for classification tasks
        self.train_cont_indices = self.cont_indices
        self.train_cat_indices = self.cat_indices[:-1]
        self.index_maps_created = True

    def _calculate_categorical_feature_distributions_and_continuous_bounds(self):
        """
        A private method to calculate the feature distributions and feature bounds that are needed to understand the
        statistical properties of the dataset.

        :return: None
        """
        # if we do not have the index maps yet then we should create that
        if not self.index_maps_created:
            self._create_index_maps()

        # copy the feature tensors and concatenate them
        X = torch.cat([self.get_Xtrain(), self.get_Xtest()], dim=0)

        # check if the dataset was standardized, if yes then destandardize X
        if self.standardized:
            X = self.de_standardize(X)

        # now run through X and create the necessary items
        X = X.detach().clone().cpu().numpy()
        n_samples = X.shape[0]
        self.categorical_histograms = {}
        self.cont_histograms = {}
        self.continuous_bounds = {}
        self.standardized_continuous_bounds = {}

        for key, (feature_type, index_map) in self.train_feature_index_map.items():
            if feature_type == 'cont':
                # calculate the bounds
                lb = min(X[:, index_map[0]])
                ub = max(X[:, index_map[0]])
                self.continuous_bounds[key] = (lb, ub)
                self.standardized_continuous_bounds[key] = ((lb - self.mean[index_map].item()) / self.std[index_map].item(),
                                                            (ub - self.mean[index_map].item()) / self.std[index_map].item())
                # calculate histograms
                value_range = np.arange(lb, ub+1)
                hist, _ = np.histogram(X[:, index_map[0]], bins=min(100, len(value_range)))
                self.cont_histograms[key] = hist / n_samples
            elif feature_type == 'cat':
                # calculate the histograms
                hist = np.sum(X[:, index_map], axis=0) / n_samples
                # extend the histogram to two entries for binary features (Bernoulli dist)
                if len(hist) == 1:
                    hist = np.array([1-hist[0], hist[0]])
                self.categorical_histograms[key] = hist
            else:
                raise ValueError('Invalid feature index map')
        self.histograms_and_continuous_bounds_calculated = True

    def create_tolerance_map(self, tol=0.319):
        """
        Given a tolerance value for multiplying the standard deviation, this method calculates a tolerance map that is
        required for the error calculation between a guessed/reconstructed batch and a true batch of data.

        :param tol: (float) Tolerance value. The tolerance interval for each continuous feature will be calculated as:
            [true - tol, true + tol].
        :return: (list) The tolerance map required for the error calculation.
        """
        x_std = self.std.clone().detach().cpu().numpy()
        cont_indices = [idxs[0] for nature, idxs in self.train_feature_index_map.values() if nature == 'cont']
        numeric_stds = x_std[cont_indices]
        tolerance_map = []
        pointer = 0

        for value in self.train_features.values():
            to_append = tol * numeric_stds[pointer] if value is None else 'cat'
            pointer += 1 if value is None else 0
            tolerance_map.append(to_append)

        return tolerance_map
