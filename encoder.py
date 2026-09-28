import torch.nn as nn
from double_conv import DoubleConv


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()

        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip_connection = self.conv(x)
        out = self.pool(skip_connection)

        return out, skip_connection