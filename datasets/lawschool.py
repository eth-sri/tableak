import torch
import numpy as np
import pandas as pd
from .base_dataset import BaseDataset
import sys

sys.path.append("..")
from utils import to_numeric, to_categorical
from sklearn.model_selection import train_test_split


class Lawschool(BaseDataset):

    def __init__(self, name='Lawschool', train_test_ratio=0.2, single_bit_binary=False, device='cpu', random_state=42):
        super(Lawschool, self).__init__(name=name, device=device, random_state=random_state)

        self.train_test_ratio = train_test_ratio

        self.features = {
            'lsat': None,
            'gpa': None,
            'race': ['Asian', 'Black', 'Hispanic', 'White'],
            'resident': ['0.0', '1.0'],
            'college': ['Arizona', 'Arizona State', 'Baltimore', 'Buffalo', 'Cincinnati', 'Cleveland State',
                        'George Mason', 'Hawaii', 'Houston',
                        'Idaho', 'Louisiana State', 'Michigan', 'Minnesota', 'Missouri at Columbia',
                        'Missouri at Kansas City', 'Nevada Las Vegas',
                        'North Carolina', 'Northern Illinois', 'Ohio State', 'Virginia', 'Washington', 'West Virginia',
                        'William and Mary', 'Wyoming'],
            'year': ['2005', '2006', '2007', '2008', '200x'],
            'gender': ['0.0', '1.0'],
            'admit': ['0.0', '1.0']
        }

        self.single_bit_binary = single_bit_binary
        self.label = 'admit'

        self.train_features = {key: self.features[key] for key in self.features.keys() if key != self.label}

        # load the data
        data_df = pd.read_stata('datasets/Lawschool/lawschs1_1.dta')

        # preprocess
        data_df.drop(['enroll', 'asian', 'black', 'hispanic', 'white', 'missingrace', 'urm'], axis=1, inplace=True)
        data_df.dropna(axis=0, inplace=True, subset=['admit'])
        data_df.replace(to_replace='', value=np.nan, inplace=True)
        data_df.dropna(axis=0, inplace=True)

        # convert to numeric
        data = data_df.to_numpy()
        data_num = (to_numeric(data, self.features, label=self.label, single_bit_binary=self.single_bit_binary)).astype(np.float32)

        # split labels and features
        X, y = data_num[:, :-1], data_num[:, -1]
        self.num_features = X.shape[1]

        # create a train and test split and shuffle
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=self.train_test_ratio,
                                                        random_state=self.random_state, shuffle=True)

        # convert to torch
        self.Xtrain, self.Xtest = torch.tensor(Xtrain).to(self.device), torch.tensor(Xtest).to(self.device)
        self.ytrain, self.ytest = torch.tensor(ytrain, dtype=torch.long).to(self.device), torch.tensor(ytest, dtype=torch.long).to(self.device)

        # set to train mode as base
        self.train()

        # calculate the standardization statistics
        self._calculate_mean_std()

        # calculate the histograms and feature bounds
        self._calculate_categorical_feature_distributions_and_continuous_bounds()

    def repeat_split(self, split_ratio=None, random_state=None):
        """
        As the dataset does not come with a standard train-test split, we assign this split manually during the
        initialization. To allow for independent experiments without much of a hassle, we allow through this method for
        a reassignment of the split.

        :param split_ratio: (float) The desired ratio of test_data/all_data.
        :param random_state: (int) The random state according to which we do the assignment,
        :return: None
        """
        if random_state is None:
            random_state = self.random_state
        if split_ratio is None:
            split_ratio = self.train_test_ratio
        X = torch.cat([self.Xtrain, self.Xtest], dim=0).detach().cpu().numpy()
        y = torch.cat([self.ytrain, self.ytest], dim=0).detach().cpu().numpy()
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=split_ratio, random_state=random_state,
                                                        shuffle=True)
        # convert to torch
        self.Xtrain, self.Xtest = torch.tensor(Xtrain).to(self.device), torch.tensor(Xtest).to(self.device)
        self.ytrain, self.ytest = torch.tensor(ytrain, dtype=torch.long).to(self.device), torch.tensor(ytest, dtype=torch.long).to(self.device)
        # update the split status as well
        self._assign_split(self.split_status)

    def decode_batch(self, batch, standardized=True):
        """
        Given a batch of numeric data, this function turns that batch back into the interpretable mixed representation.
        We overwrite this base method in this dataset due to the prevalence of non integer features.

        :param batch: (torch.tensor) A batch of data to be decoded according to the features and statistics of the
            underlying dataset.
        :param standardized: (bool) Flag if the batch had been standardized or not.
        :return: (np.ndarray) The batch decoded into mixed representation as the dataset is out of the box.
        """
        if standardized:
            batch = self.de_standardize(batch)
        return to_categorical(batch.clone().detach().cpu(), self.train_features,
                              single_bit_binary=self.single_bit_binary, nearest_int=False)

    def _calculate_categorical_feature_distributions_and_continuous_bounds(self):
        """
        A private method to calculate the feature distributions and feature bounds that are needed to understand the
        statistical properties of the dataset.
        We overwrite this base method in this dataset due to the prevalence of non integer features.

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
                if key == 'gpa':
                    hist, _ = np.histogram(X[:, index_map[0]], bins=30)
                else:
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
