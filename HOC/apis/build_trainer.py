from HOC.utils import TRAINER
from HOC.utils.build import build_from_cfg


def build_trainer(cfg):
    return build_from_cfg(cfg['trainer'], TRAINER)
