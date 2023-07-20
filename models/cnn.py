import torch
import torch.nn as nn


class LinReLU(nn.Module):
    """
    A linear layer followed by a ReLU activation layer.
    """

    def __init__(self, in_size, out_size):
        super(LinReLU, self).__init__()

        linear = nn.Linear(in_size, out_size)
        ReLU = nn.ReLU()
        self.layers = nn.Sequential(linear, ReLU)

    def reset_parameters(self):
        self.layers[0].reset_parameters()
        return self

    def forward(self, x):
        x = self.layers(x)
        return x


class CNN(nn.Module):
    """
    A simple CNN with ReLU activations and a 1D convolutional layer.
    """

    def __init__(self, input_size, layout, kernel_size=3, batch_norm=True):

        super(CNN, self).__init__()
        conv_out_channels = layout[0]
        prev_fc_size = (input_size - (kernel_size-1)) * conv_out_channels

        # Add 1D convolutional layer with proper input and output channels
        conv1d = nn.Conv1d(1, conv_out_channels, kernel_size, bias=False)
        bn = nn.BatchNorm1d(conv_out_channels)
        layers = [conv1d, bn, nn.ReLU(), nn.Flatten()] if batch_norm else [conv1d, nn.ReLU(), nn.Flatten()]

        for i, fc_size in enumerate(layout[1:]):
            if i + 1 < len(layout) - 1:
                layers += [LinReLU(prev_fc_size, fc_size)]
            else:
                layers += [nn.Linear(prev_fc_size, fc_size)]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # Add unsqueeze operation to match the input dimensions to Conv1D
        x = x.unsqueeze(1)
        x = self.layers(x)
        return x


class CNN_fixed_arch(nn.Module):
    """
    A hacked CNN arch that returns the batchnorm statistics as well, needed to emulate the 
    See through gradients attack.
    """
    
    def __init__(self, input_size, channels=16, kernel_size=3):
        
        super(CNN_fixed_arch, self).__init__()
        conv_out_channels = channels
        size = (input_size - (kernel_size-1)) * conv_out_channels
        self.conv1d = nn.Conv1d(1, conv_out_channels, kernel_size, bias=False)
        self.bn = nn.BatchNorm1d(conv_out_channels)
        self.relu1 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(size, 100)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(100, 100)
        self.relu3 = nn.ReLU()
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x, return_bn_stats=False):
        # Add unsqueeze operation to match the input dimensions to Conv1D
        x = x.unsqueeze(1)
        x = self.conv1d(x)
        # calculate the batch-norm stats by hand as well
        inter_mean = x.mean(dim=(0, 2))
        inter_var = x.var(dim=(0, 2))
        x = self.bn(x)
        x = self.relu1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        x = self.relu3(x)
        x = self.fc3(x)
        bn_stats = [(inter_mean, inter_var)]
        
        if return_bn_stats:
            return x, bn_stats
        else:
            return x
