from HOC.utils.build import build_from_cfg
from HOC.utils.registry import MODELS


def build_model(cfg):
    return build_from_cfg(cfg, MODELS)
