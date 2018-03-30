import torch
import numpy as np
from torchvision.models.vgg import vgg16
import skvideo.io
from skimage.transform import resize
from torch.autograd import Variable
from utils.common_utils import plot_image_grid



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

    def __call__(self, mode='train'):
        if mode == 'train':
            start = self.batch_order_[self.current_batch_] * batch_size
            end = (self.batch_order_[self.current_batch_] + 1) * batch_size
            self.current_batch_ += 1

            if self.current_batch_ == len(self.batch_order_):
                self.current_batch_ = 0
                self.batch_order_ = np.random.choice(self.n_batches_, size=self.n_batches_, replace=False)
            return self.maps_generator_(start, end, self.k, self.r), self.target_[start : end]
        else:
            start = self.test_batch_ * batch_size
            end = (self.test_batch_ + 1) * batch_size
            return self.maps_generator_(len(self.target_) + start, len(self.target_) + end, self.k, self.r)

class BatchGeneratorVideoAndImage:
    def __init__(self, target, maps_generator_video, maps_generator_image, k, r):
        self.target_ = target
        self.maps_generator_video_ = maps_generator_video
        self.maps_generator_image_ = maps_generator_image
        self.k = k
        self.r = r
        self.T = len(target)

    def __call__(self, mode='train'):
        if mode == 'train':
            return torch.cat([self.maps_generator_video_(0, len(self.target_) - 1, self.k, self.r), 
                              self.maps_generator_image_(0, 1, self.k, self.r)]), self.target_
        else:
            return self.maps_generator_image_(0, 64, self.k, self.r)


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


def plotter(net_output):
    out_np = net_output[0].cpu().data.numpy()
    plot_image_grid([np.clip(out_np, 0, 1)], factor=4, nrow=1)


def mse_loss(input, target):
    return torch.sum((input - target) ** 2) / input.data.nelement()


def numpyToVar(x, requires_grad=False):
    xs = torch.FloatTensor(x)
    xs = xs.cuda()
    return Variable(xs, requires_grad=requires_grad)

def prepareWriting(x):
    return np.clip(np.transpose(x.cpu().data.numpy(), (0, 2, 3, 1)), 0, 1)

def preprocessTarget(video, T, pic_size):
    data = video / np.max(video)
    data = np.transpose(data[:T], [0, 3, 1, 2])
    data = np.array(list(map(lambda x: resize(x, output_shape=(
        3, pic_size, pic_size), mode='constant'), data)))
    return numpyToVar(data)
