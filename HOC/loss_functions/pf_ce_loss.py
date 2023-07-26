import torch.nn as nn
import torch.nn.functional as F

from HOC.utils import LOSS_FUNCTIONS


@LOSS_FUNCTIONS.register_module()
class CELossPf(nn.Module):
    def __init__(self):
        super(CELossPf, self).__init__()

    def forward(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None, ignore_index=-1, reduction='none')
        v = losses.mul_(weight).sum() / weight.sum()
        return v
