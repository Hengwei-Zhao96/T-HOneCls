import torch
import torch.nn as nn
import torch.nn.functional as F

from HOC.utils import LOSS_FUNCTIONS


@LOSS_FUNCTIONS.register_module()
class TaylorCELossPf(nn.Module):
    """
    Can Cross Entropy Loss Be Robust to Label Noise?
    """

    def __init__(self, order, equal_weight):
        super(TaylorCELossPf, self).__init__()
        self.order = order
        self.equal_weight = equal_weight

    def forward(self, pred, positive_mask, unlabeled_mask, epoch, device):
        positive_mask = positive_mask.unsqueeze(dim=0).float()
        unlabeled_mask = unlabeled_mask.unsqueeze(dim=0).float()

        sigmoid_pred = torch.sigmoid(pred)

        positive_approx = 0
        negative_approx = 0
        for i in range(1, self.order + 1):
            positive_approx += (torch.ones_like(positive_mask) - sigmoid_pred) ** i / i
            negative_approx += sigmoid_pred ** i / i

        _positive_loss = (positive_approx * positive_mask).sum()
        _negative_loss = (negative_approx * unlabeled_mask).sum()

        if self.equal_weight:
            positive_loss = _positive_loss / positive_mask.sum()
            negative_loss = _negative_loss / unlabeled_mask.sum()
        else:
            positive_loss = _positive_loss / (positive_mask.sum() + unlabeled_mask.sum())
            negative_loss = _negative_loss / (positive_mask.sum() + unlabeled_mask.sum())

        loss = positive_loss + negative_loss

        return loss, positive_loss, negative_loss


@LOSS_FUNCTIONS.register_module()
class TaylorCEPULossPf(nn.Module):
    """
    Can Cross Entropy Loss Be Robust to Label Noise?
    """

    def __init__(self, order, equal_weight):
        super(TaylorCEPULossPf, self).__init__()
        self.order = order
        self.equal_weight = equal_weight

    def forward(self, pred, positive_mask, unlabeled_mask, epoch, device):
        positive_mask = positive_mask.unsqueeze(dim=0).float()
        unlabeled_mask = unlabeled_mask.unsqueeze(dim=0).float()

        log_sigmoid = F.logsigmoid(pred)
        _positive_loss = -1 * (log_sigmoid * positive_mask).sum()

        negative_approx = 0
        for i in range(1, self.order + 1):
            negative_approx += torch.exp(log_sigmoid) ** i / i

        _negative_loss = (negative_approx * unlabeled_mask).sum()

        if self.equal_weight:
            positive_loss = _positive_loss / positive_mask.sum()
            negative_loss = _negative_loss / unlabeled_mask.sum()
        else:
            positive_loss = _positive_loss / (positive_mask.sum() + unlabeled_mask.sum())
            negative_loss = _negative_loss / (positive_mask.sum() + unlabeled_mask.sum())

        loss = positive_loss + negative_loss

        return loss, positive_loss, negative_loss
