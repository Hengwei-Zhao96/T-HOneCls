import torch
import torch.nn as nn
import torch.nn.functional as F
from torchprofile import profile_macs


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=planes)
        self.conv2 = conv3x3(planes, planes)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.gn3 = nn.GroupNorm(num_groups=4, num_channels=planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.gn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=1):
        super(segmenthead, self).__init__()
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=inplanes)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=interplanes)

        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)

        self.relu = nn.ReLU(inplace=False)
        self.scale_factor = scale_factor

    def forward(self, x):
        height = x.shape[-2] * self.scale_factor
        width = x.shape[-1] * self.scale_factor

        x = self.conv1(self.relu(self.gn1(x)))
        out = self.conv2(self.relu(self.gn2(x)))
        out = F.interpolate(out, size=[height, width], mode='bilinear', align_corners=True)

        return out


class OCNet(nn.Module):

    def __init__(self, in_channels, num_classes,
                 block=BasicBlock,
                 layers=(2, 2, 2, 2),
                 planes=16,
                 head_planes=16):
        super(OCNet, self).__init__()

        highres_planes = int(planes * 2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, planes, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=planes),
            nn.ReLU(inplace=True))

        self.relu = nn.ReLU(inplace=False)

        self.layer1 = self._make_layer(block, planes, planes, layers[0])
        self.layer2 = self._make_layer(block, planes, planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes * 2, planes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes * 4, planes * 8, layers[3], stride=2)
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)

        self.layer3_ = self._make_layer(block, planes * 2, highres_planes, 2)
        self.layer4_ = self._make_layer(block, highres_planes, highres_planes, 2)
        self.layer5_ = self._make_layer(Bottleneck, highres_planes, highres_planes, 1)

        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, highres_planes, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=highres_planes))

        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, highres_planes, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=highres_planes))

        self.down3 = nn.Sequential(
            nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=planes * 4))
        self.down4 = nn.Sequential(
            nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=planes * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=planes * 8))

        self.final_layer = segmenthead(planes * 4, head_planes, num_classes, scale_factor=8)

        self.layer5_compression = nn.Sequential(
            nn.GroupNorm(num_groups=4, num_channels=planes * 16),
            nn.ReLU(),
            nn.Conv2d(planes * 16, planes * 4, kernel_size=1),
        )

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=4, num_channels=planes * block.expansion))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def forward(self, x):

        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        layers = []

        x = self.conv1(x)
        x = self.layer1(x)
        layers.append(x)

        x = self.layer2(self.relu(x))
        layers.append(x)

        x = self.layer3(self.relu(x))
        layers.append(x)
        x_ = self.layer3_(self.relu(layers[1]))

        x = x + self.down3(self.relu(x_))
        x_ = x_ + F.interpolate(
            self.compression3(self.relu(layers[2])),
            size=[height_output, width_output],
            mode='bilinear',
            align_corners=True)

        x = self.layer4(self.relu(x))
        layers.append(x)
        x_ = self.layer4_(self.relu(x_))

        x = x + self.down4(self.relu(x_))
        x_ = x_ + F.interpolate(
            self.compression4(self.relu(layers[3])),
            size=[height_output, width_output],
            mode='bilinear',
            align_corners=True)

        x_ = self.layer5_(self.relu(x_))
        x = F.interpolate(
            self.layer5_compression(self.layer5(self.relu(x))),
            size=[height_output, width_output],
            mode='bilinear',
            align_corners=True)

        x_ = self.final_layer(x + x_)

        return x_


def get_ocnet(config):
    return OCNet(in_channels=config['in_channels'], num_classes=config['num_classes'])


if __name__ == "__main__":
    model = OCNet(in_channels=270, num_classes=1)
    input = torch.randn((1, 270, 800, 800))
    macs = profile_macs(model, input)
    print('Gmacs: {:.2f} G'.format(macs / 1e9))
    # from torchstat import stat
    #
    # stat(model, (270, 800, 800))
