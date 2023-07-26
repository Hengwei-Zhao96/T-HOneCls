import torch
import torch.nn as nn

from HOC.loss_functions.loss_functions import sigmoid_loss
from HOC.utils import LOSS_FUNCTIONS


@LOSS_FUNCTIONS.register_module()
class UPULossPf(nn.Module):
    def __init__(self, prior):
        super(UPULossPf, self).__init__()
        self.prior = torch.tensor(prior)

    def forward(self, pred, positive_mask, unlabeled_mask, epoch, device):
        positive_mask = positive_mask.unsqueeze(dim=0).float()
        unlabeled_mask = unlabeled_mask.unsqueeze(dim=0).float()

        positive_p_loss = sigmoid_loss(pred) * positive_mask

        unlabeled_loss = sigmoid_loss(-pred)
        positive_n_loss = unlabeled_loss * positive_mask
        unlabeled_n_loss = unlabeled_loss * unlabeled_mask

        estimated_p_loss = positive_p_loss.sum() / positive_mask.sum()

        estimated_u_n_loss = unlabeled_n_loss.sum() / unlabeled_mask.sum()
        estimated_p_n_loss = positive_n_loss.sum() / positive_mask.sum()

        estimated_n_loss = (estimated_u_n_loss - self.prior.to(device) * estimated_p_n_loss) / (
                1 - self.prior.to(device))

        loss = self.prior.to(device) * estimated_p_loss + (1 - self.prior.to(device)) * estimated_n_loss

        return loss, estimated_p_loss, estimated_n_loss


@LOSS_FUNCTIONS.register_module()
class CSPULossPf(nn.Module):
    def __init__(self, prior, positive_weight, negative_weight):
        super(CSPULossPf, self).__init__()
        self.prior = torch.tensor(prior)
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

    def forward(self, pred, positive_mask, unlabeled_mask, epoch, device):
        positive_mask = positive_mask.unsqueeze(dim=0).float()
        unlabeled_mask = unlabeled_mask.unsqueeze(dim=0).float()

        positive_p_loss = sigmoid_loss(pred) * positive_mask

        unlabeled_loss = sigmoid_loss(-pred)
        positive_n_loss = unlabeled_loss * positive_mask
        unlabeled_n_loss = unlabeled_loss * unlabeled_mask

        estimated_p_loss = positive_p_loss.sum() / positive_mask.sum()

        estimated_u_n_loss = unlabeled_n_loss.sum() / unlabeled_mask.sum()
        estimated_p_n_loss = positive_n_loss.sum() / positive_mask.sum()

        estimated_n_loss = (estimated_u_n_loss - self.prior.to(device) * estimated_p_n_loss) / (
                1 - self.prior.to(device))

        loss = self.prior.to(device) * self.positive_weight * estimated_p_loss + (
                1 - self.prior.to(device)) * self.negative_weight * estimated_n_loss

        return loss, estimated_p_loss, estimated_n_loss
