import torch
import torch.nn as nn
from .skip import skip
from .common import *
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_depth, pic_size, skip_args):
        super(Net, self).__init__()
        # Context network

        self.context = skip(**skip_args)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(input_depth, 30, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(30, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * (28 ** 2), 32), # for 128 input_size
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.final = nn.Sequential(
            nn.Conv2d(input_depth + 3, 3, kernel_size=2, padding=1),
            nn.MaxPool2d(2, stride=1),
            nn.ReLU(True)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(len(x), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        res = self.final(torch.cat([self.stn(x), self.context(x)], dim=1))
        return res
