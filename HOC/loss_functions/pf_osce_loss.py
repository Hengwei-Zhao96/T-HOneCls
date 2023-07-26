import torch.nn
import torch.nn as nn
import torch.nn.functional as F

from HOC.utils import LOSS_FUNCTIONS


class CELossPf(nn.Module):
    def __init__(self):
        super(CELossPf, self).__init__()

    def forward(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None, ignore_index=-1, reduction='none')
        v = losses.mul_(weight).sum() / weight.sum()
        return v


class TBCE_PU(nn.Module):
    def __init__(self, order):
        super(TBCE_PU, self).__init__()
        self.order = order

    def forward(self, os_target, positive_mask, unlabeled_mask):
        sigmoid_os_target = torch.sigmoid(os_target)
        positive_loss = -1 * (torch.log(sigmoid_os_target + 1e-8) * positive_mask).sum() / (positive_mask.sum() + 1e-8)

        negative_loss_approx = 0
        for i in range(1, self.order + 1):
            negative_loss_approx += (sigmoid_os_target ** i) / i
        negative_loss = (negative_loss_approx * unlabeled_mask).sum() / (unlabeled_mask.sum() + 1e-8)

        loss = (positive_loss + negative_loss) / 2

        return loss

# class TBCE_PU(nn.Module):
#     def __init__(self, order):
#         super(TBCE_PU, self).__init__()
#         self.order = order
#
#     def forward(self, os_target, positive_mask, unlabeled_mask):
#         sigmoid_os_target = torch.sigmoid(os_target)
#
#         positive_loss_approx = 0
#         for i in range(1, 51):
#             positive_loss_approx += ((torch.ones_like(sigmoid_os_target)-sigmoid_os_target) ** i) / i
#         positive_loss = (positive_loss_approx * positive_mask).sum() / (positive_mask.sum() + 1e-8)
#
#         negative_loss_approx = 0
#         for i in range(1, self.order + 1):
#             negative_loss_approx += (sigmoid_os_target ** i) / i
#         negative_loss = (negative_loss_approx * unlabeled_mask).sum() / (unlabeled_mask.sum() + 1e-8)
#
#         loss = (positive_loss + negative_loss) / 2
#
#         return loss


@LOSS_FUNCTIONS.register_module()
class OSCELossPf(nn.Module):
    def __init__(self, order, num_classes):
        super(OSCELossPf, self).__init__()
        self.ce = CELossPf()
        self.order = order
        self.os_loss = TBCE_PU(self.order)
        self.num_classes = num_classes

    def forward(self, target, gt, mcc_mask, os_target, positive_mask, unlabeled_mask):
        mcc_loss = self.ce(target, gt, mcc_mask)
        os_loss = self.os_loss(os_target, positive_mask, unlabeled_mask)
        loss = mcc_loss + self.num_classes * os_loss
        return loss
