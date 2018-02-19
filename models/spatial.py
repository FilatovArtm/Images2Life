import torch
import torch.nn as nn
from .skip import skip
from .common import *
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_depth, pic_size, skip_args_main, skip_args_grid, spatial_only=False):
        super(Net, self).__init__()
        # Context network
        self.spatial_only_ = spatial_only
        self.context = skip(**skip_args_main)


        self.pre_grid_ = skip(**skip_args_grid)
        self.grid_prior_ = torch.zeros((1, 128, 128, 2))
        self.grid_prior_[0, :, :, 0] += torch.linspace(-1, 1, 128)
        self.grid_prior_[0, :, :, 1] += torch.linspace(-1, 1, 128)[:, None]
        self.grid_prior_ = torch.autograd.Variable(self.grid_prior_.cuda())

        self.final = nn.Sequential()
        if self.spatial_only_ is False:
            self.final.add(conv(input_depth + 3, input_depth + 3, 3, 2, bias=True, pad=2, downsample_mode="stride"))
        else:
            self.final.add(conv(input_depth, input_depth + 3, 3, 2, bias=True, pad=2, downsample_mode="stride"))

        self.final.add(bn(input_depth + 3))
        self.final.add(act('LeakyReLU'))
        self.final.add(nn.Upsample(scale_factor=2, mode="nearest"))

        self.final.add(conv(input_depth + 3, 3, 3, bias=True, pad=2))
        self.final.add(nn.Sigmoid())



    # Spatial transformer network forward function
    def stn(self, x):
        grid = self.pre_grid_(x)
        grid = grid.view(-1, grid.data.shape[2], grid.data.shape[3], 2)
        xs = F.grid_sample(x, F.tanh(grid + self.grid_prior_))

        return xs

    def forward(self, x):
        # transform the input
        if self.spatial_only_ is True:
            res = self.final(self.stn(x))
        else:
            res = self.final(torch.cat([self.stn(x), self.context(x)], dim=1))
        return res
