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
        self.localization = nn.Sequential()
        self.localization.add(conv(input_depth, input_depth, 3, 2, bias=True, pad=2))
        self.localization.add(bn(input_depth))
        self.localization.add(act('LeakyReLU'))

        self.localization.add(conv(input_depth, 2, 3, bias=True, pad=2))
        self.localization.add(bn(2))
        self.localization.add(act('LeakyReLU'))

        self.localization.add(nn.Upsample(scale_factor=2, mode="nearest"))


        self.final = nn.Sequential()
        self.final.add(conv(input_depth + 3, 4, 3, 2, bias=True, pad=2, downsample_mode="stride"))
        self.final.add(bn(4))
        self.final.add(act('LeakyReLU'))

        self.final.add(conv(4, 3, 3, bias=True, pad=2))
        self.final.add(bn(3))
        self.final.add(act('LeakyReLU'))
        self.final.add(nn.Upsample(scale_factor=2, mode="nearest"))


    # Spatial transformer network forward function
    def stn(self, x):
        grid = self.localization(x)
        grid = grid.view(-1, grid.data.shape[2], grid.data.shape[3], 2)

        grid = F.tanh(grid)
        xs = F.grid_sample(x, grid)

        return xs

    def forward(self, x):
        # transform the input
        res = self.final(torch.cat([self.stn(x), self.context(x)], dim=1))
        return res
