import torch.nn as nn

from HOC.utils import MODELS

def conv3x3_bn_relu(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 1, 1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


def downsample2x(in_channel, out_channel, downsample):
    if downsample:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True))


def repeat_block(block_channel, n):
    layers = [
        nn.Sequential(
            conv3x3_bn_relu(block_channel, block_channel),
        )
        for _ in range(n)]
    return nn.Sequential(*layers)

@MODELS.register_module()
class FreeNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size):
        super(FreeNetEncoder, self).__init__()
        if patch_size in (3, 5):
            d = (True, False, False)
        elif patch_size in (7, 9, 11):
            d = (True, True, False)
        elif patch_size in (13, 15, 17):
            d = (True, True, True)
        else:
            raise NotImplemented
        self.stem = conv3x3_bn_relu(in_channels, 64)
        self.block1 = repeat_block(64, 1)
        self.downsample1 = downsample2x(64, 128, d[0])
        self.block2 = repeat_block(128, 1)
        self.downsample2 = downsample2x(128, 192, d[1])
        self.block3 = repeat_block(192, 1)
        self.downsample3 = downsample2x(192, 256, d[2])
        self.block4 = repeat_block(256, 1)
        self.avgpool = nn.AvgPool2d(2)
        self.fc = nn.Linear(256, out_channels)

    def forward(self, input):
        x = self.stem(input)
        x = self.block1(x)
        x = self.downsample1(x)
        x = self.block2(x)
        x = self.downsample2(x)
        x = self.block3(x)
        x = self.downsample3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = self.fc(x.view(x.shape[0], -1))
        return x
