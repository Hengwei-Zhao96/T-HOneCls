import copy

import segmentation_models_pytorch as smp
import torch.nn as nn

from HOC.models.model_utils import batch_norm_to_group_norm
from HOC.utils import MODELS


class SMPModel(nn.Module):
    def __init__(self, encoder_name: str, model_network: str, in_channels: int = 4, n_class: int = 1):
        super(SMPModel, self).__init__()
        self.encoder_name = encoder_name
        self.model_network = model_network
        self.in_channels = in_channels
        self.n_class = n_class

        self.model = self._get_smp_model()

    def _get_smp_model(self):
        return getattr(smp, self.model_network)(
            encoder_name=self.encoder_name,
            encoder_weights=None,
            in_channels=self.in_channels,
            classes=self.n_class,
        )

    def forward(self, x):
        return self.model(x)


@MODELS.register_module()
class SMPModelGn(nn.Module):
    def __init__(self, encoder_name, model_network, in_channels, out_channels):
        super(SMPModelGn, self).__init__()
        self.model = SMPModel(encoder_name, model_network, in_channels, out_channels)
        self.modelgn = copy.deepcopy(batch_norm_to_group_norm(self.model))
        del self.model

    def forward(self, x):
        return self.modelgn(x)
