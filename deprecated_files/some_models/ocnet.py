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


class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()

        self.scale0 = nn.Sequential(
            nn.GroupNorm(num_groups=4, num_channels=inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(num_groups=4, num_channels=inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            nn.GroupNorm(num_groups=4, num_channels=inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            nn.GroupNorm(num_groups=4, num_channels=inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.GroupNorm(num_groups=4, num_channels=inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))

        self.process1 = nn.Sequential(
            nn.GroupNorm(num_groups=4, num_channels=branch_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False))
        self.process2 = nn.Sequential(
            nn.GroupNorm(num_groups=4, num_channels=branch_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False))
        self.process3 = nn.Sequential(
            nn.GroupNorm(num_groups=4, num_channels=branch_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False))
        self.process4 = nn.Sequential(
            nn.GroupNorm(num_groups=4, num_channels=branch_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False))

        self.compression = nn.Sequential(
            nn.GroupNorm(num_groups=4, num_channels=branch_planes * 5),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False))

        self.shortcut = nn.Sequential(
            nn.GroupNorm(num_groups=4, num_channels=inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False))

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                                                   size=[height, width],
                                                   mode='bilinear',
                                                   align_corners=True) + x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                                                    size=[height, width],
                                                    mode='bilinear',
                                                    align_corners=True) + x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                                                   size=[height, width],
                                                   mode='bilinear',
                                                   align_corners=True) + x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                                                   size=[height, width],
                                                   mode='bilinear',
                                                   align_corners=True) + x_list[3])))

        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


class segmenthead(nn.Module):

    def __init__(self, inplanes, stemplanes, interplanes, outplanes, scale_factor=1):
        super(segmenthead, self).__init__()
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=inplanes)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.gn1_1 = nn.GroupNorm(num_groups=4, num_channels=interplanes)

        self.conv2 = nn.Conv2d(stemplanes, interplanes, kernel_size=1, stride=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=interplanes)

        self.calibration = nn.Conv2d(interplanes, interplanes, kernel_size=3, padding=1, stride=1)
        self.gn3 = nn.GroupNorm(num_groups=4, num_channels=interplanes)

        self.conv3 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)

        self.relu = nn.ReLU(inplace=False)
        self.scale_factor = scale_factor

    def forward(self, x, x0):
        height = x.shape[-2] * self.scale_factor
        width = x.shape[-1] * self.scale_factor

        x = self.gn1_1(self.conv1(self.relu(self.gn1(x))))
        x = self.relu(
            F.interpolate(x, size=[height, width], mode='bilinear', align_corners=True) + self.gn2(self.conv2(x0)))

        out = self.conv3(self.relu(self.gn3(self.calibration(x))))

        return out


class OCNet(nn.Module):

    def __init__(self, in_channels, num_classes,
                 block=BasicBlock,
                 layers=(2, 2, 2, 2),
                 planes=16,
                 spp_planes=128,
                 head_planes=16,
                 augment=False):
        super(OCNet, self).__init__()

        stem_planes = int(planes / 2)
        highres_planes = int(planes * 2)
        self.augment = augment

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_planes, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=stem_planes),
            nn.ReLU(inplace=True),
        )

        # self.cbam_stem_1 = CBAM(gate_channels=stem_planes, reduction_ratio=4)

        self.conv1 = nn.Sequential(
            nn.Conv2d(stem_planes, planes, kernel_size=3, stride=2, padding=1),
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

        # self.stem_compression3 = nn.Sequential(
        #     nn.Conv2d(highres_planes, stem_planes, kernel_size=1, bias=False),
        #     nn.GroupNorm(num_groups=4, num_channels=stem_planes)
        # )
        # self.stem_fusion3 = nn.Sequential(
        #     nn.Conv2d(stem_planes, stem_planes, kernel_size=3, stride=1, padding=1),
        #     nn.GroupNorm(num_groups=4, num_channels=stem_planes),
        #     nn.ReLU(inplace=True),
        # )

        # self.stem_compression4 = nn.Sequential(
        #     nn.Conv2d(highres_planes, stem_planes, kernel_size=1, bias=False),
        #     nn.GroupNorm(num_groups=4, num_channels=stem_planes)
        # )
        # self.stem_fusion4 = nn.Sequential(
        #     nn.Conv2d(stem_planes, stem_planes, kernel_size=3, stride=1, padding=1),
        #     nn.GroupNorm(num_groups=4, num_channels=stem_planes),
        #     nn.ReLU(inplace=True),
        # )

        # self.cbam_stem_2 = CBAM(gate_channels=stem_planes, reduction_ratio=4)

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

        self.spp = DAPPM(planes * 16, spp_planes, highres_planes * 2)

        if self.augment:
            self.seghead_extra = segmenthead(highres_planes, stem_planes, head_planes, num_classes, scale_factor=8)

        self.final_layer = segmenthead(highres_planes * 2, stem_planes, head_planes, num_classes, scale_factor=8)

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

        x = self.stem(x)
        # x = self.cbam_stem_1(x)
        layers.append(x)

        x = self.conv1(x)
        x = self.layer1(x)

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

        # layers[0] = layers[0]+F.interpolate(
        #     self.stem_compression3(self.relu(x_)),
        #     size=[height_output*8,width_output*8],
        #     mode='bilinear',
        #     align_corners=True
        # )
        # layers[0] = self.stem_fusion3(self.relu(layers[0]))

        if self.augment:
            temp = x_

        x = self.layer4(self.relu(x))
        layers.append(x)
        x_ = self.layer4_(self.relu(x_))

        x = x + self.down4(self.relu(x_))
        x_ = x_ + F.interpolate(
            self.compression4(self.relu(layers[3])),
            size=[height_output, width_output],
            mode='bilinear',
            align_corners=True)
        # layers[0] = layers[0] + F.interpolate(
        #     self.stem_compression4(self.relu(x_)),
        #     size=[height_output * 8, width_output * 8],
        #     mode='bilinear',
        #     align_corners=True
        # )
        # layers[0] = self.cbam_stem_2(self.stem_fusion4(self.relu(layers[0])))

        x_ = self.layer5_(self.relu(x_))
        x = F.interpolate(
            self.spp(self.layer5(self.relu(x))),
            size=[height_output, width_output],
            mode='bilinear',
            align_corners=True)

        x_ = self.final_layer(x + x_, layers[0])

        if self.augment:
            x_extra = self.seghead_extra(temp, layers[0])
            return [x_, x_extra]
        else:
            return x_


def get_ocnet(config):
    return OCNet(in_channels=config['in_channels'], num_classes=config['num_classes'])


if __name__ == "__main__":
    model = OCNet(in_channels=274, num_classes=1)
    input = torch.randn((1, 274, 1232, 304))
    macs = profile_macs(model, input)
    print('Gmacs: {:.2f} G'.format(macs / 1e9))
    # from torchstat import stat
    #
    # stat(model, (270, 800, 800))
