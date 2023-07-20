"""
The following code is adapted with some changes from the CTGAN library: Xu et al., Modeling Tabular Data using 
Conditional GAN, 2019, https://github.com/sdv-dev/CTGAN/blob/master/ctgan/synthesizers/ctgan.py

License:
MIT License

Copyright (c) 2019, MIT Data To AI Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch.nn as nn
import torch


class Residual(nn.Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = nn.Linear(i, o)
        self.bn = nn.BatchNorm1d(o)
        self.relu = nn.ReLU()

    def forward(self, input_, return_bn_stats=False):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        bn_stats = (out.mean(dim=0), out.var(dim=0))
        out = self.bn(out)
        out = self.relu(out)
        if return_bn_stats:
            return torch.cat([out, input_], dim=1), bn_stats
        else:
            return torch.cat([out, input_], dim=1)


class ResNet(nn.Module):

    def __init__(self, input_size, layout):
        super(ResNet, self).__init__()
        layers = [nn.Flatten()]  # does not play any role, but makes the code neater
        prev_fc_size = input_size
        for i, fc_size in enumerate(layout):
            if i + 1 < len(layout):
                layers += [Residual(prev_fc_size, fc_size)]
            else:
                layers += [nn.Linear(prev_fc_size, fc_size)]
            prev_fc_size = fc_size + prev_fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class ResNet_fixed_arch(nn.Module):

    def __init__(self, input_size):
        super(ResNet_fixed_arch, self).__init__()
        self.flatten = nn.Flatten()
        self.residual1 = Residual(input_size, 100)
        self.residual2 = Residual(100 + input_size, 100)
        self.fc = nn.Linear(200 + input_size, 2)

    def forward(self, x, return_bn_stats=False):
        x = self.flatten(x)
        if return_bn_stats:
            x, bn_1 = self.residual1(x, return_bn_stats=True)
            x, bn_2 = self.residual2(x, return_bn_stats=True)
        else:
            x = self.residual1(x, return_bn_stats=False)
            x = self.residual2(x, return_bn_stats=False)
        x = self.fc(x)

        if return_bn_stats:
            return x, [bn_1, bn_2]
        else:
            return x
