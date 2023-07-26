import torch
import torch.nn as nn
import torch.nn.functional as F

from HOC.utils import LOSS_FUNCTIONS


@LOSS_FUNCTIONS.register_module()
class AsyPULossPf(nn.Module):
    """
    ASYMMETRIC LOSS FOR POSITIVE-UNLABELED LEARNING
    """

    def __init__(self, asy=1):
        super(AsyPULossPf, self).__init__()
        self.asy = asy

    def forward(self, pred, positive_mask, unlabeled_mask, epoch, device):
        positive_mask = positive_mask.unsqueeze(dim=0).float()
        unlabeled_mask = unlabeled_mask.unsqueeze(dim=0).float()

        log_sigmoid = F.logsigmoid(pred)
        positive_loss = -1 * (log_sigmoid * positive_mask).sum() / positive_mask.sum()
        unlabeled_loss = -1 * (torch.log(
            (1 - torch.exp(log_sigmoid)) + torch.tensor(self.asy).to(
                device)) * unlabeled_mask).sum() / unlabeled_mask.sum()

        loss = positive_loss + unlabeled_loss

        return loss, positive_loss, unlabeled_loss
