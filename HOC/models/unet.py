import torch
import torch.nn as nn

from HOC.utils import MODELS


class ConvGnReLu2x(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvGnReLu2x, self).__init__()
        self.conv_gn_relu_2x = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(16, ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(16, ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_gn_relu_2x(x)


class UpConvGnReLu(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConvGnReLu, self).__init__()
        self.up_conv_gn_relu = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(16, ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up_conv_gn_relu(x)


@MODELS.register_module()
class UNetGn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetGn, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvGnReLu2x(in_channels, 64)
        self.Conv2 = ConvGnReLu2x(64, 128)
        self.Conv3 = ConvGnReLu2x(128, 256)
        self.Conv4 = ConvGnReLu2x(256, 512)
        self.Conv5 = ConvGnReLu2x(512, 1024)

        self.Up5 = UpConvGnReLu(1024, 512)
        self.Up4 = UpConvGnReLu(512, 256)
        self.Up3 = UpConvGnReLu(256, 128)
        self.Up2 = UpConvGnReLu(128, 64)

        self.Up5_Conv4 = ConvGnReLu2x(1024, 512)
        self.Up4_Conv3 = ConvGnReLu2x(512, 256)
        self.Up3_Conv2 = ConvGnReLu2x(256, 128)
        self.Up2_Conv1 = ConvGnReLu2x(128, 64)

        self.Conv1x1 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoder
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # Decoder+Concat
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up5_Conv4(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up4_Conv3(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up3_Conv2(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up2_Conv1(d2)

        return self.Conv1x1(d2)
