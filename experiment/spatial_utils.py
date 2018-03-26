import torch
import numpy as np


class SpatialMapsGenerator:

    def __init__(self, M, noise_level=0.1):
        self.spatial_variables = {"alpha": None,
                                  "beta": None, "gamma": None, "delta": None}
        self.maps_number_ = M
        for key in spatial_variables:
            spatial_variables[key] = numpyToVar(np.random.normal(
                0, noise_level, M), requires_grad=True)

    def __call__(self, T, k, r):
        x, y = map(lambda x: numpyToVar(x.astype(np.float32)),
                   np.meshgrid(np.arange(k), np.arange(r)))
        single_maps = x[None, :, :] * \
            self.spatial_variables["alpha"][:, None, None]
        single_maps += y[None, :, :] * \
            self.spatial_variables["beta"][:, None, None]
        single_maps += self.spatial_variables["delta"][:, None, None]

        single_maps = single_maps.expand(T, self.maps_number_, k, r)
        single_maps += torch.ger(Variable(torch.arange(0, T)
                                          ).cuda(), self.spatial_variables["gamma"])[:, :, None, None]

        return single_maps


class BatchGenerator:

    def __init__(self, target, maps_generator, k, r):
        self.target_ = target
        self.maps_generator_ = maps_generator
        self.k = k
        self.r = r
        self.T = len(target)

    def __call__(self):
        return self.maps_generator_(self.T, self.k, self.r), self.target_


class SpatialLoss:

    def __init__(self):
        vgg_model = vgg16(pretrained=True)
        self.loss_network_ = LossNetwork(vgg_model.cuda())
        self.loss_network_.eval()

    def __call__(self, Y, Y_hat):
        return mse_loss(Y_hat, Y) + mse_loss(self.loss_network_(Y_hat), self.loss_network_(Y))


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
    plot_image_grid([np.clip(out_np, 0, 1)], factor=figsize, nrow=1)


def mse_loss(input, target):
    return torch.sum((input - target) ** 2) / input.data.nelement()


def numpyToVar(x, requires_grad=False):
    xs = torch.FloatTensor(x)
    xs = xs.cuda()
    return Variable(xs, requires_grad=requires_grad)


def preprocessTarget(video, T, pic_size):
    data = video / np.max(video)
    data = np.transpose(data[:T], [0, 3, 1, 2])
    data = np.array(list(map(lambda x: resize(x, output_shape=(
        3, pic_size, pic_size), mode='constant'), data)))
    return numpyToVar(data)
