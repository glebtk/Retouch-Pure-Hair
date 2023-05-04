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


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation="identity", batch_norm=False, transpose=False):
        super(ConvolutionalBlock, self).__init__()
        params = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding
        }

        if transpose:
            params["output_padding"] = (stride - 1)

        act = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "lrelu": nn.LeakyReLU(negative_slope=0.2),
            "identity": nn.Identity()
        }

        self.block = nn.Sequential(
            nn.ConvTranspose2d(**params) if transpose else nn.Conv2d(**params),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            act[activation]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Model(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_layers=5, num_features=64, weight_init=None):
        super(Model, self).__init__()

        residual_layers = []
        for _ in range(num_layers - 2):
            residual_layers.append(
                ConvolutionalBlock(num_features, num_features, kernel_size=3, stride=1, padding=1, activation="lrelu")
            )

        self.net = nn.Sequential(
            ConvolutionalBlock(in_channels, num_features, kernel_size=3, stride=1, padding=1, activation="lrelu"),
            *residual_layers,
            ConvolutionalBlock(num_features, out_channels, kernel_size=3, stride=1, padding=1, activation="tanh")
        )

        if weight_init:
            self.net.apply(lambda layer: init_weights(layer, method=weight_init))

    def forward(self, x):
        residual = x
        x = self.net(x)
        return x + residual


if __name__ == "__main__":
    print(f"Model:")
    model = Model(in_channels=3, out_channels=3, num_layers=5, num_features=32)
    summary(model, (3, 256, 256))
