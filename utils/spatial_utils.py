import torch
import numpy as np
from torchvision.models.vgg import vgg16
from torch.autograd import Variable
from utils.common_utils import numpyToVar, mse_loss



class SpatialMapsGenerator:

    def __init__(self, M, noise_level=0.1):
        self.spatial_variables = {"alpha": None,
                                  "beta": None, "gamma": None, "delta": None}
        self.maps_number_ = M
        for key in self.spatial_variables:
            self.spatial_variables[key] = numpyToVar(np.random.normal(
                0, noise_level, M), requires_grad=True)

    def __call__(self, start_T, end_T, k, r):
        x, y = map(lambda x: numpyToVar(x.astype(np.float32)),
                   np.meshgrid(np.arange(k), np.arange(r)))
        single_maps = x[None, :, :] * \
            self.spatial_variables["alpha"][:, None, None]
        single_maps += y[None, :, :] * \
            self.spatial_variables["beta"][:, None, None]
        single_maps += self.spatial_variables["delta"][:, None, None]

        single_maps = single_maps.expand(end_T - start_T, self.maps_number_, k, r)
        time_vars = torch.ger(Variable(torch.arange(start_T, end_T)).cuda(), self.spatial_variables["gamma"])[:, :, None, None]
        time_vars = time_vars.expand(end_T - start_T, self.maps_number_, k, r)
        result = single_maps + time_vars

        return result

class BatchGenerator:

    def __init__(self, target, maps_generator, k, r, batch_size=8):
        self.target_ = target
        self.maps_generator_ = maps_generator
        self.k = k
        self.r = r
        self.T = len(target)

        self.current_batch_ = 0
        self.batch_size_ = batch_size
        self.n_batches_ = int(len(target) / batch_size)
        self.batch_order_ = np.random.choice(self.n_batches_, size=self.n_batches_, replace=False)

        self.test_batch_ = 0

    def __call__(self, mode='train', begin=0, n=0):
        if mode == 'train':
            start = self.batch_order_[self.current_batch_] * self.batch_size_
            end = (self.batch_order_[self.current_batch_] + 1) * self.batch_size_
            start, end = int(start), int(end)

            self.current_batch_ += 1

            if self.current_batch_ == len(self.batch_order_):
                self.current_batch_ = 0
                self.batch_order_ = np.random.choice(self.n_batches_, size=self.n_batches_, replace=False)
            return self.maps_generator_(start, end, self.k, self.r), self.target_[start : end]
        else:
            start = int(begin + n * self.batch_size_)
            end = int(begin + (n + 1) * self.batch_size_)
            return self.maps_generator_(start, end, self.k, self.r)


class PerceptualLoss:

    def __init__(self, alpha=1.):
        vgg_model = vgg16(pretrained=True)
        self.loss_network_ = LossNetwork(vgg_model.cuda())
        self.loss_network_.eval()
        self.alpha_ = alpha

    def __call__(self, Y, Y_hat):
        return mse_loss(self.loss_network_(Y_hat), self.loss_network_(Y)) * self.alpha_ + \
        mse_loss(Y_hat, Y) * (1 - self.alpha_)


class LossNetwork(torch.nn.Module):

    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers_ = vgg_model.features
        self.layer_name_mapping_ = {
            '3': "relu1_2"
        }

    def forward(self, x):
        for name, module in self.vgg_layers_._modules.items():
            x = module(x)
            if name in self.layer_name_mapping_:
                return x

