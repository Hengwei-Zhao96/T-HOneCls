import torch
from torch.autograd import Function
import torch.nn as nn

from HOC.utils import LOSS_FUNCTIONS


class _nnPULoss(Function):
    @staticmethod
    def forward(ctx, pred, positive_mask, unlabeled_mask, prior, beta, gamma):
        # Save parameter for backward
        ctx.save_for_backward(pred, positive_mask, unlabeled_mask, prior, beta, gamma)
        # Loss function
        ctx.loss_func = lambda x: torch.sigmoid(-x)
        # Get the Loss positive and negative samples
        ctx.positive_x = (positive_mask == 1).float()
        ctx.unlabeled_x = (unlabeled_mask == 1).float()
        ctx.positive_num = torch.max(torch.sum(ctx.positive_x), torch.tensor(1).cuda().float())
        ctx.unlabeled_num = torch.max(torch.sum(ctx.unlabeled_x), torch.tensor(1).cuda().float())
        ctx.positive_y = ctx.loss_func(pred)
        ctx.unlabeled_y = ctx.loss_func(-pred)
        ctx.positive_loss = torch.sum(prior * ctx.positive_x / ctx.positive_num * ctx.positive_y.squeeze())
        ctx.negative_loss = torch.sum((
                                              ctx.unlabeled_x / ctx.unlabeled_num - prior * ctx.positive_x / ctx.positive_num) * ctx.unlabeled_y.squeeze())
        if ctx.negative_loss.data < -beta:
            objective = ctx.positive_loss - beta
        else:
            objective = ctx.positive_loss + ctx.negative_loss
        return objective

    @staticmethod
    def backward(ctx, grad_output):
        pred, positive_mask, unlabeled_mask, prior, beta, gamma = ctx.saved_tensors
        d_input = torch.zeros(pred.shape).cuda().float()
        d_positive_loss = -prior * ctx.positive_x / ctx.positive_num * ctx.positive_y.squeeze() * (
                1 - ctx.positive_y.squeeze())
        d_negative_loss = (
                                  ctx.unlabeled_x / ctx.unlabeled_num - prior * ctx.positive_x / ctx.positive_num) * ctx.unlabeled_y.squeeze() * (
                                  1 - ctx.unlabeled_y.squeeze())

        if ctx.negative_loss.data < -beta:
            d_input = -gamma * d_negative_loss
        else:
            d_input = d_positive_loss + d_negative_loss
        d_input = d_input.unsqueeze(1)
        d_input = d_input * grad_output
        return d_input, None, None, None, None, None


@LOSS_FUNCTIONS.register_module()
class NnPULossPf(nn.Module):
    def __init__(self, prior, beta=None, gamma=None):
        super(NnPULossPf, self).__init__()
        self.prior = torch.tensor(prior)
        if beta is None:
            self.beta = torch.tensor(0)
        else:
            self.beta = torch.tensor(beta)
        if gamma is None:
            self.gamma = torch.tensor(1)
        else:
            self.gamma = torch.tensor(gamma)

    def forward(self, pred, positive_mask, unlabeled_mask, epoch, device):
        return _nnPULoss.apply(pred, positive_mask, unlabeled_mask, self.prior.cuda(), self.beta.cuda(),
                               self.gamma.cuda()), torch.tensor(0), torch.tensor(0)
