import torch
from HOC.utils.build import build_from_cfg
from HOC.utils import OPTIMIZERS

OPTIMIZERS.register_module(module=torch.optim.SGD)
OPTIMIZERS.register_module(module=torch.optim.Adam)


def build_optimizer(cfg):
    return build_from_cfg(cfg, OPTIMIZERS)
