import torch
import torch.nn as nn
import torch.nn.functional as F


def cbr_module(in_channel, out_channel, kernel, stride, padding):
    cbr = nn.Sequential(
        nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel,
            stride=stride,
            padding=padding
        ),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )
    return cbr


class ResBlock(nn.Module):
    def __init__(self, channels, inner_channels):
        super().__init__()
        self.conv1 = cbr_module(channels, inner_channels, 1, 1, 0)
        self.conv2 = nn.Conv2d(inner_channels, channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(out)
        out += x
        return F.relu(out)


class ResSppNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.features = nn.Sequential(
            cbr_module(input_channels, 64, 3, 1, 1),
            cbr_module(64, 128, 3, 2, 1),

            ResBlock(128, 64),
            cbr_module(128, 256, 3, 2, 1),

            ResBlock(256, 256),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_channels),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    net = ResSppNet(1, 10)
    x = torch.rand((5,1,28,28))
    y = net.forward(x)
    print(y.shape)
