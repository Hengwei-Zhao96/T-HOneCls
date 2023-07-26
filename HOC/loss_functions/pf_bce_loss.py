import torch
import torch.nn as nn

from HOC.utils import LOSS_FUNCTIONS
from HOC.utils.logits_norm import logits_norm


@LOSS_FUNCTIONS.register_module()
class BCELossPf(nn.Module):
    def __init__(self, equal_weight=False, reduction=True):
        super(BCELossPf, self).__init__()
        self.equal_weight = equal_weight
        self.reduction = reduction

    def forward(self, pred, positive_mask, unlabeled_mask, epoch, device):
        positive_mask = positive_mask.unsqueeze(dim=0).float()
        unlabeled_mask = unlabeled_mask.unsqueeze(dim=0).float()

        positive_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(pred, positive_mask) * positive_mask
        unlabeled_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(pred, (1 - unlabeled_mask)) * unlabeled_mask

        if not self.equal_weight:
            estimated_p_loss = positive_loss.sum() / (positive_mask.sum() + unlabeled_mask.sum())
            estimated_n_loss = unlabeled_loss.sum() / (positive_mask.sum() + unlabeled_mask.sum())
            loss = estimated_p_loss + estimated_n_loss
        else:
            estimated_p_loss = positive_loss.sum() / positive_mask.sum()
            estimated_n_loss = unlabeled_loss.sum() / unlabeled_mask.sum()
            loss = estimated_p_loss + estimated_n_loss

        if not self.reduction:
            return loss, estimated_p_loss, estimated_n_loss, unlabeled_loss
        else:
            return loss, estimated_p_loss, estimated_n_loss


class ConKLLossPf(nn.Module):
    def __init__(self, equal_weight=False, reduction=True):
        super(ConKLLossPf, self).__init__()
        self.equal_weight = equal_weight
        self.reduction = reduction

    def forward(self, pred1, pred2, positive_mask, unlabeled_mask, epoch, device):
        positive_mask = positive_mask.unsqueeze(dim=0).float()
        unlabeled_mask = unlabeled_mask.unsqueeze(dim=0).float()

        pred1_normalized = logits_norm(pred1) + 1e-8
        pred2_normalized = logits_norm(pred2) + 1e-8

        loss_raw = (nn.KLDivLoss(reduction='none')(torch.log(pred1_normalized), pred2_normalized) + nn.KLDivLoss(
            reduction='none')(torch.log(pred2_normalized), pred1_normalized)).sum(axis=1, keepdim=True)

        positive_loss = loss_raw * positive_mask
        unlabeled_loss = loss_raw * unlabeled_mask

        if not self.equal_weight:
            estimated_p_loss = positive_loss.sum() / (positive_mask.sum() + unlabeled_mask.sum())
            estimated_n_loss = unlabeled_loss.sum() / (positive_mask.sum() + unlabeled_mask.sum())
            loss = estimated_p_loss + estimated_n_loss
        else:
            estimated_p_loss = positive_loss.sum() / positive_mask.sum()
            estimated_n_loss = unlabeled_loss.sum() / unlabeled_mask.sum()
            loss = estimated_p_loss + estimated_n_loss

        if not self.reduction:
            return loss, estimated_p_loss, estimated_n_loss, unlabeled_loss
        else:
            return loss, estimated_p_loss, estimated_n_loss


@LOSS_FUNCTIONS.register_module()
class GCELossPf(nn.Module):
    def __init__(self, q=0.7, equal_weight=False):
        super(GCELossPf, self).__init__()
        self.q = q
        self.equal_weight = equal_weight

    def forward(self, pred, positive_mask, unlabeled_mask, epoch, device):
        positive_mask = positive_mask.unsqueeze(dim=0).float()
        unlabeled_mask = unlabeled_mask.unsqueeze(dim=0).float()

        positive_sigmoid_pre = torch.sigmoid(pred)+1e-8
        negative_sigmoid_pre = (torch.ones_like(positive_mask) - positive_sigmoid_pre)+1e-8

        _positive_loss = (((torch.ones_like(
            positive_mask) - positive_sigmoid_pre ** self.q) / self.q) * positive_mask).sum()
        _negative_loss = (((torch.ones_like(
            positive_mask) - negative_sigmoid_pre ** self.q) / self.q) * unlabeled_mask).sum()

        if self.equal_weight:
            positive_loss = _positive_loss / positive_mask.sum()
            negative_loss = _negative_loss / unlabeled_mask.sum()
        else:
            positive_loss = _positive_loss / (positive_mask.sum() + unlabeled_mask.sum())
            negative_loss = _negative_loss / (positive_mask.sum() + unlabeled_mask.sum())

        loss = positive_loss + negative_loss

        return loss, positive_loss, negative_loss


@LOSS_FUNCTIONS.register_module()
class SymmetricBCELossPf(nn.Module):
    '''
    Symmetric Cross Entropy for Robust Learning With Noisy Labels
    '''

    def __init__(self, alpha=0.01, beta=1.0, A=-4, equal_weight=False):
        super(SymmetricBCELossPf, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.A = -1 * A
        self.equal_weight = equal_weight

    def forward(self, pred, positive_mask, unlabeled_mask, epoch, device):
        positive_mask = positive_mask.unsqueeze(dim=0).float()
        unlabeled_mask = unlabeled_mask.unsqueeze(dim=0).float()

        # bce
        bce_positive_loss = (torch.nn.BCEWithLogitsLoss(reduction='none')(pred, positive_mask) * positive_mask).sum()
        bce_unlabeled_loss = (
                torch.nn.BCEWithLogitsLoss(reduction='none')(pred, (1 - unlabeled_mask)) * unlabeled_mask).sum()

        # rbce
        sigmoid_pred = torch.clamp(torch.sigmoid(pred), min=1e-7, max=1.0)
        rbce_positive_loss = ((torch.ones_like(sigmoid_pred) - sigmoid_pred) * self.A * positive_mask).sum()
        rbce_unlabeled_loss = ((sigmoid_pred * self.A) * unlabeled_mask).sum()

        if not self.equal_weight:
            estimated_p_loss = (self.alpha * bce_positive_loss + self.beta * rbce_positive_loss) / (
                    positive_mask.sum() + unlabeled_mask.sum())
            estimated_n_loss = (self.alpha * bce_unlabeled_loss + self.beta * rbce_unlabeled_loss) / (
                    positive_mask.sum() + unlabeled_mask.sum())
        else:
            estimated_p_loss = (self.alpha * bce_positive_loss + self.beta * rbce_positive_loss) / positive_mask.sum()
            estimated_n_loss = (
                                       self.alpha * bce_unlabeled_loss + self.beta * rbce_unlabeled_loss) / unlabeled_mask.sum()

        loss = estimated_p_loss + estimated_n_loss
        return loss, estimated_p_loss, estimated_n_loss
