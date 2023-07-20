import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, input_size, out_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, out_size)

    def forward(self, x):
        x = self.linear(x)
        return x
