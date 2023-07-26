from HOC.utils.build import build_from_cfg
from HOC.utils.registry import LOSS_FUNCTIONS


def build_loss_function(cfg):
    return build_from_cfg(cfg, LOSS_FUNCTIONS)
