import torch
import torch.nn as nn
from torch import Tensor
from torchsummary import summary


def init_weights(layer, method='xavier_uniform'):
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        if method == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(layer.weight)
        elif method == 'xavier_normal':
            torch.nn.init.xavier_normal_(layer.weight)
        elif method == 'kaiming_uniform':
            torch.nn.init.kaiming_uniform_(layer.weight)
        elif method == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(layer.weight)
        else:
            raise ValueError(f'Unknown weight initialization method: {method}')
        if layer.bias is not None:
            layer.bias.data.zero_()


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation="relu", weight_init=None, transpose=False):
        super(ConvBlock, self).__init__()

        conv_layer = nn.ConvTranspose2d if transpose else nn.Conv2d
        conv_kwargs = {'output_padding': (stride - 1)} if transpose else {}

        act_dict = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "lrelu": nn.LeakyReLU(negative_slope=0.2)
        }

        self.block = nn.Sequential(
            conv_layer(in_channels, out_channels, kernel_size, stride, padding, **conv_kwargs),
            nn.BatchNorm2d(out_channels),
            act_dict[activation]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, weight_init=None):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=3, stride=2, padding=1, activation="lrelu"),
            # ConvBlock(64, 128, kernel_size=3, stride=2, padding=1, activation="lrelu"),
            # ConvBlock(128, 256, kernel_size=3, stride=2, padding=1, activation="lrelu"),
            # ConvBlock(256, 128, kernel_size=3, stride=2, padding=1, transpose=True, activation="lrelu"),
            # ConvBlock(128, 64, kernel_size=3, stride=2, padding=1, transpose=True, activation="lrelu"),
            ConvBlock(64, 32, kernel_size=3, stride=2, padding=1, transpose=True, activation="lrelu"),
            ConvBlock(32, out_channels, kernel_size=5, stride=1, padding=2, activation="sigmoid", transpose=True)
        )

        if weight_init:
            self.net.apply(lambda layer: init_weights(layer, method=weight_init))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3, weight_init=None):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=3, stride=2, padding=0, activation="lrelu"),
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=0, activation="lrelu"),
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=0, activation="lrelu"),
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=0, activation="lrelu"),
            nn.Flatten(),
            nn.Linear(256*15*15, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        if weight_init:
            self.net.apply(lambda layer: init_weights(layer, method=weight_init))

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


if __name__ == "__main__":
    print(f"Generator:")
    gen = Generator(in_channels=3, out_channels=3)
    summary(gen, (3, 256, 256))

    print(f"Discriminator:")
    disc = Discriminator(in_channels=3)
    summary(disc, (3, 256, 256))

