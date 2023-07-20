import torch.nn as nn
import torch
import numpy as np
from utils.eval_metrics import get_acc_and_bac


class LinReLU(nn.Module):
    """
    A linear layer followed by a ReLU activation layer.
    """

    def __init__(self, in_size, out_size):
        super(LinReLU, self).__init__()

        linear = nn.Linear(in_size, out_size)
        ReLU = nn.ReLU()
        # self.Dropout = nn.Dropout(0.25)
        self.layers = nn.Sequential(linear, ReLU)

    def reset_parameters(self):
        self.layers[0].reset_parameters()
        return self

    def forward(self, x):
        x = self.layers(x)
        return x


class FullyConnected(nn.Module):
    """
    A simple fully connected neural network with ReLU activations.
    """

    def __init__(self, input_size, layout):

        super(FullyConnected, self).__init__()
        layers = [nn.Flatten()]  # does not play any role, but makes the code neater
        prev_fc_size = input_size
        for i, fc_size in enumerate(layout):
            if i + 1 < len(layout):
                layers += [LinReLU(prev_fc_size, fc_size)]
            else:
                layers += [nn.Linear(prev_fc_size, fc_size)]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class FullyConnectedTrainer:
    """
    An object to wrap the training process of a fully connected neural network.
    """

    def __init__(self, data_x, data_y, optimizer, criterion, device='cpu', verbose=False):
        """
        :param data_x: (torch.tensor) Training features.
        :param data_y: (torch.tensor) Training labels.
        :param optimizer: Instantiated torch optimizer to train the network with the parameters of the network assigned
            to it.
        :param criterion: Instantiated torch loss function. Will be used as the training loss.
        :param device: (str) The device on which the training shall be executed. Note that this device has to match for
            all given device sensitive objects, i.e. for the network and for the data.
        :param verbose: (bool) Toggle to print the progress of the training process.
        """
        self.data_x = data_x
        self.data_y = data_y
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.verbose = verbose

    # TODO: I do not think it was very smart from my side to put the optimizer into the object with assigned weights but
    #  then have the network only here. This workflow needs some revision (its might worth it to just simply turn it
    #  into a lone-standing function).
    def train(self, net, n_epochs, batch_size, reset=True, shuffle=True, testx=None, testy=None):
        """
        Method to train a given neural network for a given number of epochs at a given batch size. The progress of the
        network's performance on a given held-out set can be recorded if such a dataset is given.

        :param net: (nn.Module) The neural network to be trained.
        :param n_epochs: (int) The number of epochs for which the network is to be trained.
        :param batch_size: (int) The size of the data batches we feed into the network to estimate its gradient at each
            iteration.
        :param reset: (bool) Toggle if you want to reinitialize the network.
        :param shuffle: (bool) Toggle if you want to reshuffle the dataset.
        :param testx: (torch.tensor, optional) If given also labels have to be given (testy). If present, in the
            beginning of each epoch the performance of the neural network on this given dataset is calculated and
            recorded, later returned at the end of the training.
        :param testy: (torch.tensor, optional) Labels for the held out in-process testing dataset. For details see the
            description of 'testx'.
        :return: (None or tuple) If 'testx' and 'testy' are given, we return the accuracy and the balanced accuracy of
            the network at the beginning of each epoch.
        """
        # get rid of any previous gradients on the data points
        self.data_x, self.data_y = self.data_x.detach(), self.data_y.detach()

        # reset the network parameters if required
        if reset:
            for layer in net.layers:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        # shuffle the data if required
        if shuffle:
            train_shuffler = torch.randperm(self.data_x.size()[0]).to(self.device)
            self.data_x, self.data_y = self.data_x[train_shuffler], self.data_y[train_shuffler]

        accs, baccs = [], []
        net.train()
        for epoch in range(n_epochs):
            running_loss = []
            for i in range(int(np.ceil(self.data_x.size()[0] / batch_size))):
                bottom_line = i * batch_size
                upper_line = min((i + 1) * batch_size, self.data_x.size()[0])
                inputs = self.data_x[bottom_line:upper_line]
                labels = self.data_y[bottom_line:upper_line]

                self.optimizer.zero_grad()
                outputs = net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += [loss.item()]
                if i % 100 == 99 and self.verbose:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, np.mean(running_loss)), end='\r')
                    running_loss = []
            if testx is not None and testy is not None:
                acc, bac = get_acc_and_bac(net, testx, testy)
                accs.append(acc)
                baccs.append(bac)
        if self.verbose:
            print('Finished Training')
        if testx is not None and testy is not None:
            return accs, baccs

