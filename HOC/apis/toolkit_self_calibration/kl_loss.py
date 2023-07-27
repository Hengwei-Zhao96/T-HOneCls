import torch
import torch.nn as nn
from HOC.utils.logits_norm import logits_norm


class KLLoss(nn.Module):
    def __init__(self, equal_weight=False):
        super(KLLoss, self).__init__()
        self.equal_weight = equal_weight

    def forward(self, pred1, pred2, positive_mask, unlabeled_mask):
        positive_mask = positive_mask.unsqueeze(dim=0).float()
        unlabeled_mask = unlabeled_mask.unsqueeze(dim=0).float()

        pred1_normalized = logits_norm(pred1) + 1e-8
        pred2_normalized = logits_norm(pred2) + 1e-8

        loss_raw = (nn.KLDivLoss(reduction='none')(torch.log(pred1_normalized), pred2_normalized) + nn.KLDivLoss(
            reduction='none')(torch.log(pred2_normalized), pred1_normalized)).sum(axis=1, keepdim=True)

        p_loss = loss_raw * positive_mask
        u_loss = loss_raw * unlabeled_mask

        if not self.equal_weight:
            loss = (p_loss.sum() + u_loss.sum()) / (positive_mask.sum() + unlabeled_mask.sum())
        else:
            loss = p_loss.sum() / positive_mask.sum() + u_loss.sum() / unlabeled_mask.sum()

        return loss
