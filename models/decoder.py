import torch
import torch.nn as nn
from .common import *
import numpy as np


class Unflatten(nn.Module):
    def forward(self, X):
        return X.view(len(X), 1, int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1])))

def decoder(
        input_size=16, output_size=128, num_output_channels=3, 
        num_channels_up=[16, 32, 64, 128, 128],
        filter_size_up=3,
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU'):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """ 
    model = nn.Sequential(nn.Linear(input_size, 16 * 16),
                          Unflatten())

    num_channels_up = [1] + num_channels_up

    stride_layers = int(np.log2(output_size / 16))
    for i in range(len(num_channels_up) - 1):

        if i < stride_layers:
            stride = 1
        else:
            stride = 2
        model.add(conv(num_channels_up[i], num_channels_up[i + 1], filter_size_up, stride=stride, bias=need_bias, pad=pad))
        model.add(bn(num_channels_up[i + 1]))
        model.add(act(act_fun))

        model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))


    model.add(conv(num_channels_up[-1], num_output_channels, 1, bias=need_bias, pad=pad))
    model.add(nn.Sigmoid())

    return model
