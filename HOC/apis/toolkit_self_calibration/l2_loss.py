import torch
import torch.nn as nn


class L2Loss(nn.Module):
    def __init__(self, equal_weight=False):
        super(L2Loss, self).__init__()
        self.equal_weight = equal_weight

    def forward(self, pred1, pred2, positive_mask, unlabeled_mask):
        positive_mask = positive_mask.unsqueeze(dim=0).float()
        unlabeled_mask = unlabeled_mask.unsqueeze(dim=0).float()

        pred1_sigmoid = torch.sigmoid(pred1)
        pred2_sigmoid = torch.sigmoid(pred2)

        loss_raw = (pred1_sigmoid - pred2_sigmoid) ** 2

        p_loss = loss_raw * positive_mask
        u_loss = loss_raw * unlabeled_mask

        if not self.equal_weight:
            loss = (p_loss.sum() + u_loss.sum()) / (positive_mask.sum() + unlabeled_mask.sum())
        else:
            loss = p_loss.sum() / positive_mask.sum() + u_loss.sum() / unlabeled_mask.sum()

        return loss
