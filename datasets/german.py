import torch
import numpy as np
import pandas as pd
from .base_dataset import BaseDataset
import sys

sys.path.append("..")
from utils import to_numeric
from sklearn.model_selection import train_test_split


class German(BaseDataset):

    def __init__(self, name='German', train_test_ratio=0.2, single_bit_binary=False, device='cpu', random_state=42):
        super(German, self).__init__(name=name, device=device, random_state=random_state)

        self.train_test_ratio = train_test_ratio

        self.features = {
            'A1': ['A1' + str(i) for i in range(1, 5)],
            'A2': None,
            'A3': ['A3' + str(i) for i in range(0, 5)],
            'A4': ['A4' + str(i) for i in range(0, 11)],
            'A5': None,
            'A6': ['A6' + str(i) for i in range(1, 6)],
            'A7': ['A7' + str(i) for i in range(1, 6)],
            'A8': None,
            'A9': ['A9' + str(i) for i in range(1, 6)],
            'A10': ['A10' + str(i) for i in range(1, 4)],
            'A11': None,
            'A12': ['A12' + str(i) for i in range(1, 5)],
            'A13': None,
            'A14': ['A14' + str(i) for i in range(1, 4)],
            'A15': ['A15' + str(i) for i in range(1, 4)],
            'A16': None,
            'A17': ['A17' + str(i) for i in range(1, 5)],
            'A18': None,
            'A19': ['A19' + str(i) for i in range(1, 3)],
            'A20': ['A20' + str(i) for i in range(1, 3)],
            'class': [1, 2]
        }

        self.single_bit_binary = single_bit_binary
        self.label = 'class'

        self.train_features = {key: self.features[key] for key in self.features.keys() if key != self.label}

        # load the data
        data_df = pd.read_csv('datasets/German/german.data', delimiter=' ', names=list(self.features.keys()), engine='python')

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
