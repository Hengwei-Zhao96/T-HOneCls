import torch
import torch.nn as nn

from HOC.utils import LOSS_FUNCTIONS


@LOSS_FUNCTIONS.register_module()
class MAELossPf(nn.Module):
    """
    Can Cross Entropy Loss Be Robust to Label Noise?
    """

    def __init__(self, equal_weight):
        super(MAELossPf, self).__init__()
        self.equal_weight = equal_weight

    def forward(self, pred, positive_mask, unlabeled_mask, epoch, device):
        positive_mask = positive_mask.unsqueeze(dim=0).float()
        unlabeled_mask = unlabeled_mask.unsqueeze(dim=0).float()

        sigmoid_pre = torch.sigmoid(pred)

        _positive_loss = (2 * (torch.ones_like(positive_mask) - sigmoid_pre) * positive_mask).sum()
        _negative_loss = (2 * sigmoid_pre * unlabeled_mask).sum()

        if self.equal_weight:
            positive_loss = _positive_loss / positive_mask.sum()
            negative_loss = _negative_loss / unlabeled_mask.sum()
        else:
            positive_loss = _positive_loss / (positive_mask.sum() + unlabeled_mask.sum())
            negative_loss = _negative_loss / (positive_mask.sum() + unlabeled_mask.sum())

        loss = positive_loss + negative_loss

        return loss, positive_loss, negative_loss
