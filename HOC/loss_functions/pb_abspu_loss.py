import torch
import torch.nn as nn

from HOC.loss_functions.loss_functions import sigmoid_loss
from HOC.utils import LOSS_FUNCTIONS


@LOSS_FUNCTIONS.register_module()
class AbsPULossPb(nn.Module):
    def __init__(self, prior):
        super(AbsPULossPb, self).__init__()
        self.prior = torch.tensor(prior)

    def forward(self, pred, label, epoch, device):
        positive_x = (label == 1).float()
        unlabeled_x = (label == 0).float()

        positive_num = torch.max(torch.sum(positive_x), torch.tensor(1).cuda().float())
        unlabeled_num = torch.max(torch.sum(unlabeled_x), torch.tensor(1).cuda().float())

        positive_loss = torch.sum(self.prior.to(device) * positive_x * sigmoid_loss(pred).squeeze() / positive_num)
        unlabeled_loss = torch.abs(torch.sum(
            (unlabeled_x / unlabeled_num - self.prior.to(device) * positive_x / positive_num) * sigmoid_loss(
                -1 * pred).squeeze()))

        loss = positive_loss + unlabeled_loss

        return loss, positive_loss, unlabeled_loss
