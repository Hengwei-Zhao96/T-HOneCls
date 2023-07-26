import torch
import torch.nn as nn
import torch.nn.functional as F

from HOC.utils.registry import LOSS_FUNCTIONS


@LOSS_FUNCTIONS.register_module()
class TaylorVarPULossPf(nn.Module):

    def __init__(self, order):
        super(TaylorVarPULossPf, self).__init__()
        self.order = order

    def log_taylor_series(self, x):
        approximation = 0
        for i in range(1, self.order + 1):
            approximation -= ((1 - x) ** i) / i
        return approximation

    def forward(self, pred, positive_mask, unlabeled_mask, epoch, device):
        positive_mask = positive_mask.unsqueeze(dim=0).float()
        device = positive_mask.device
        unlabeled_mask = unlabeled_mask.unsqueeze(dim=0).float()
        log_sigmoid = F.logsigmoid(pred)
        positive_loss = -1 * (log_sigmoid * positive_mask).sum() / positive_mask.sum()

        exp_phi = (torch.exp(log_sigmoid) * unlabeled_mask).sum() / torch.sum(unlabeled_mask)
        taylor_var_unlabeled_loss = self.log_taylor_series(exp_phi)

        unlabeled_loss = taylor_var_unlabeled_loss

        loss = unlabeled_loss + positive_loss

        return loss, positive_loss, unlabeled_loss
