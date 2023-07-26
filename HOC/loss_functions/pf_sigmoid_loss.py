import torch.nn as nn

from HOC.loss_functions.loss_functions import sigmoid_loss
from HOC.utils import LOSS_FUNCTIONS


@LOSS_FUNCTIONS.register_module()
class SigmoidLossPf(nn.Module):
    def __init__(self, equal_weight=False):
        super(SigmoidLossPf, self).__init__()
        self.equal_weight = equal_weight

    def forward(self, pred, positive_mask, negative_mask, epoch, device):
        positive_mask = positive_mask.unsqueeze(dim=0).float()
        negative_mask = negative_mask.unsqueeze(dim=0).float()

        positive_loss = sigmoid_loss(pred) * positive_mask
        negative_loss = sigmoid_loss(-pred) * negative_mask

        estimated_p_loss = positive_loss.sum() / positive_mask.sum()
        estimated_n_loss = negative_loss.sum() / negative_mask.sum()

        if self.equal_weight:
            loss = (estimated_p_loss + estimated_n_loss) / 2
        else:
            loss = (positive_loss.sum() + negative_loss.sum()) / (positive_mask.sum() + negative_mask.sum())

        return loss, estimated_p_loss, estimated_n_loss
