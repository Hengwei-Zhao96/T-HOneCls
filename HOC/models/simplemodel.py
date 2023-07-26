import torch.nn as nn
import torch.nn.functional as F

from HOC.utils.registry import MODELS


def conv3x3_gn_relu(in_channel, out_channel, num_group):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 1, 1),
        nn.GroupNorm(num_group, out_channel),
        nn.ReLU(inplace=True)
    )


def downsample2x(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 2, 1),
        nn.ReLU(inplace=True)
    )


@MODELS.register_module()
class SimpleNet(nn.Module):
    def __init__(self, in_channels=274, num_classes=1):
        super(SimpleNet, self).__init__()
        self.layer1 = conv3x3_gn_relu(in_channels, 128, 16)
        self.layer1_2 = downsample2x(128, 64)
        self.layer2 = conv3x3_gn_relu(64, 128, 16)
        self.layer3 = conv3x3_gn_relu(128, 128, 16)

        self.cls_pred_conv = nn.Conv2d(128, num_classes, 1)

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x

    def forward(self, x, y=None, w=None, **kwargs):
        layer_1 = self.layer1(x)
        layer_1_2 = self.layer1_2(layer_1)
        layer_2 = self.layer2(layer_1_2)
        final_feat = self.layer3(F.interpolate(layer_2, scale_factor=2.0, mode='nearest') + layer_1)

        logit = self.cls_pred_conv(final_feat)
        return logit