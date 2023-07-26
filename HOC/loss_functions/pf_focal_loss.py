import torch
import torch.nn as nn

from HOC.loss_functions.loss_functions import bce_loss
from HOC.utils import LOSS_FUNCTIONS


@LOSS_FUNCTIONS.register_module()
class FocalLossPf(nn.Module):
    def __init__(self, class_weight=0.3, focal_weight=0.1):
        super(FocalLossPf, self).__init__()
        self.class_weight = torch.tensor(class_weight)
        self.focal_weight = focal_weight

    def forward(self, pred, positive_mask, negative_mask, epoch, device):
        positive_mask = positive_mask.unsqueeze(dim=0).float()
        negative_mask = negative_mask.unsqueeze(dim=0).float()

        p_weight = torch.pow((torch.ones_like(pred) - torch.clamp(torch.sigmoid(pred), min=0, max=0.999)),
                             self.focal_weight)
        n_weight = torch.pow(torch.clamp(torch.sigmoid(pred), min=0, max=0.999),self.focal_weight)

        positive_loss = bce_loss(pred, positive_mask) * positive_mask * p_weight
        negative_loss = bce_loss(pred, negative_mask, positive=False) * negative_mask*n_weight

        estimated_p_loss = positive_loss.sum() / positive_mask.sum()

        estimated_n_loss = negative_loss.sum() / negative_mask.sum()

        loss = self.class_weight.to(device) * estimated_p_loss + (1 - self.class_weight.to(device)) * estimated_n_loss

        return loss, estimated_p_loss, estimated_n_loss
