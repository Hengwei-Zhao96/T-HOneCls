from torch import nn
from torch.nn import functional as F

from HOC.models.toolkit_hrnet.encoder import HRNetV2Encoder48
from HOC.models.toolkit_hrnet.decoder import HRNetSegmentationDecoder
from HOC.utils import MODELS


class HRNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.encoder = HRNetV2Encoder48()
        self.encoder = self.encoder.change_input_channels(input_channels=input_channels)
        self.decoder = HRNetSegmentationDecoder(feature_maps=self.encoder.channels, output_channels=num_classes)

    def forward(self, x):
        enc_features = self.encoder(x)
        mask = self.decoder(enc_features)
        mask = F.interpolate(mask, size=x.size()[2:], mode="bilinear", align_corners=False)

        return mask


@MODELS.register_module()
class HRNetGn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HRNetGn, self).__init__()
        self.model = HRNet(in_channels, out_channels)

    def forward(self, x):
        return self.model(x)
