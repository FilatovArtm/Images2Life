import torch
import torch.nn as nn
from .skip import skip
from .common import *
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_depth, pic_size, skip_args, skip_args_tail, num_spatials=3):
        super(Net, self).__init__()
        # Context network
        self.spatials_ = []
        for _ in range(num_spatials):
            self.spatials_.append(skip(**skip_args))

        self.grid_prior_ = torch.zeros((1, 128, 128, 2))
        self.grid_prior_[0, :, :, 0] += torch.linspace(-1, 1, 128)
        self.grid_prior_[0, :, :, 1] += torch.linspace(-1, 1, 128)[:, None]
        self.grid_prior_ = torch.autograd.Variable(self.grid_prior_.cuda())

        self.tail_ = skip(**skip_args_tail)


    # Spatial transformer network forward function
    def stn(self, x):
        results = []
        for spatial_net in self.spatials_:
            grid = spatial_net(x)
            grid = grid.view(-1, grid.shape[2], grid.shape[3], 2)
            results.append(F.grid_sample(x, F.tanh(grid + self.grid_prior_)))

        return torch.cat(results, dim=1)

    def forward(self, x):
        res = self.tail_(self.stn(x))
        return res
