import torch.nn as nn
import torch.nn.functional as F

from HOC.utils import LOSS_FUNCTIONS


@LOSS_FUNCTIONS.register_module()
class CELossPb(nn.Module):
    def __init__(self):
        super(CELossPb, self).__init__()

    def forward(self, x, y, epoch, device):
        losses = F.cross_entropy(x, y.long())
        return losses
