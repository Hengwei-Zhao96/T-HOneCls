import torch
from HOC.utils.build import build_from_cfg
from HOC.utils import LR_SCHEDULER

LR_SCHEDULER.register_module(module=torch.optim.lr_scheduler.ExponentialLR)
LR_SCHEDULER.register_module(module=torch.optim.lr_scheduler.PolynomialLR)


def build_lr_scheduler(cfg):
    return build_from_cfg(cfg, LR_SCHEDULER)
