import functools

import torchlib

from torch import nn


def _get_norm_layer_2d(norm):
    if norm == 'none':
        return torchlib.Identity
    elif norm == 'batch_norm':
        return nn.BatchNorm2d
    elif norm == 'instance_norm':
        return functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm == 'layer_norm':
        return lambda num_features: nn.GroupNorm(1, num_features)
    else:
        raise NotImplementedError


class ConvGenerator(nn.Module):

    def __init__(self,
                 input_dim=128,
                 output_channels=3,
                 dim=64,
                 n_upsamplings=4,
                 norm='batch_norm'):
        super().__init__()

        Norm = _get_norm_layer_2d(norm)

        def dconv_norm_relu(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=False or Norm == torchlib.Identity),
                Norm(out_dim),
                nn.ReLU()
            )

        layers = []

        # 1: 1x1 -> 4x4
        d = min(dim * 2 ** (n_upsamplings - 1), dim * 8)
        layers.append(dconv_norm_relu(input_dim, d, kernel_size=4, stride=1, padding=0))

        # 2: upsamplings, 4x4 -> 8x8 -> 16x16 -> ...
        for i in range(n_upsamplings - 1):
            d_last = d
            d = min(dim * 2 ** (n_upsamplings - 2 - i), dim * 8)
            layers.append(dconv_norm_relu(d_last, d, kernel_size=4, stride=2, padding=1))

        layers.append(nn.ConvTranspose2d(d, output_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, z):
        x = self.net(z)
        return x


class ConvDiscriminator(nn.Module):

    def __init__(self,
                 input_channels=3,
                 dim=64,
                 n_downsamplings=4,
                 norm='batch_norm'):
        super().__init__()

        Norm = _get_norm_layer_2d(norm)

        def conv_norm_lrelu(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=False or Norm == torchlib.Identity),
                Norm(out_dim),
                nn.LeakyReLU(0.2)
            )

        layers = []

        # 1: downsamplings, ... -> 16x16 -> 8x8 -> 4x4
        d = dim
        layers.append(nn.Conv2d(input_channels, d, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2))

        for i in range(n_downsamplings - 1):
            d_last = d
            d = min(dim * 2 ** (i + 1), dim * 8)
            layers.append(conv_norm_lrelu(d_last, d, kernel_size=4, stride=2, padding=1))

        # 2: logit
        layers.append(nn.Conv2d(d, 1, kernel_size=4, stride=1, padding=0))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return y
