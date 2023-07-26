import pytorch_ood.loss

from HOC.utils.build import build_from_cfg
from HOC.utils.registry import LOSS_FUNCTIONS

LOSS_FUNCTIONS.register_module(module=pytorch_ood.loss.OutlierExposureLoss)
LOSS_FUNCTIONS.register_module(module=pytorch_ood.loss.EntropicOpenSetLoss)
LOSS_FUNCTIONS.register_module(module=pytorch_ood.loss.EnergyRegularizedLoss)


def build_loss_function(cfg):
    return build_from_cfg(cfg, LOSS_FUNCTIONS)
