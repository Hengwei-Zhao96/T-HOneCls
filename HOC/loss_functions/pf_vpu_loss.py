import torch
import torch.nn as nn
import torch.nn.functional as F

from HOC.utils.registry import LOSS_FUNCTIONS


@LOSS_FUNCTIONS.register_module()
class VarPULossPf(nn.Module):

    def __init__(self, alpha=1, lamda=0):
        super(VarPULossPf, self).__init__()
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, pred, positive_mask, unlabeled_mask, epoch, device):
        positive_mask = positive_mask.unsqueeze(dim=0).float()
        unlabeled_mask = unlabeled_mask.unsqueeze(dim=0).float()
        log_sigmoid = F.logsigmoid(pred)
        unlabeled_loss = torch.log(((torch.exp(log_sigmoid)) * unlabeled_mask).sum() / torch.sum(unlabeled_mask) + 1e-8)
        # print('unlabeled')
        # print(((torch.exp(log_sigmoid)) * unlabeled_mask).sum())
        # unlabeled_loss = torch.log(
        #     ((torch.exp(log_sigmoid) + torch.tensor(1).to(device)) * unlabeled_mask).sum() / torch.sum(
        #         unlabeled_mask))
        positive_loss = -1 * (log_sigmoid * positive_mask).sum() / positive_mask.sum()
        # print('positive')
        # print((torch.exp(log_sigmoid) * positive_mask).sum())

        loss = unlabeled_loss + positive_loss

        return loss, positive_loss, unlabeled_loss


@LOSS_FUNCTIONS.register_module()
class TaylorVarPULossPf(nn.Module):

    def __init__(self, order):
        super(TaylorVarPULossPf, self).__init__()
        self.order = order

    # def positive_taylor_series(self, x):
    #     return (torch.ones_like(x) - x)

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


# @LOSS_FUNCTIONS.register_module()
# class TaylorVarPULossPf(nn.Module):
#
#     def __init__(self, order, m, k):
#         super(TaylorVarPULossPf, self).__init__()
#         self.order = order
#         self.m = m
#         self.k = k
#
#     # def positive_taylor_series(self, x):
#     #     return (torch.ones_like(x) - x)
#
#     def log_taylor_series(self, x):
#         approximation = 0
#         for i in range(1, self.order + 1):
#             approximation -= ((1 - x) ** i) / i
#         return approximation
#
#     def forward(self, pred, positive_mask, unlabeled_mask, epoch, device):
#         positive_mask = positive_mask.unsqueeze(dim=0).float()
#         device = positive_mask.device
#         unlabeled_mask = unlabeled_mask.unsqueeze(dim=0).float()
#         log_sigmoid = F.logsigmoid(pred)
#         positive_loss = -1 * (log_sigmoid * positive_mask).sum() / positive_mask.sum()
#         # positive_loss += positive_loss+self.positive_taylor_series(torch.exp(log_sigmoid)).sum()/positive_mask.sum()
#
#         exp_phi = (torch.exp(log_sigmoid) * unlabeled_mask).sum() / torch.sum(unlabeled_mask)
#         taylor_var_unlabeled_loss = self.log_taylor_series(exp_phi)
#         bce_unlabeled_loss = (torch.nn.BCEWithLogitsLoss(reduction='none')(pred, (
#                 1 - unlabeled_mask)) * unlabeled_mask).sum() / unlabeled_mask.sum()
#
#         weight = torch.sigmoid(torch.tensor(self.m * (epoch - self.k))).to(device)
#
#         unlabeled_loss = (1 - weight) * bce_unlabeled_loss + weight * taylor_var_unlabeled_loss
#
#         loss = unlabeled_loss + positive_loss
#
#         return loss, positive_loss, unlabeled_loss


# @LOSS_FUNCTIONS.register_module()
# class TaylorVarPULossPfLabelEMA(nn.Module):
#
#     def __init__(self, order, m, k):
#         super(TaylorVarPULossPfLabelEMA, self).__init__()
#         self.order = order
#         self.m = m
#         self.k = k
#
#     # def positive_taylor_series(self, x):
#     #     return (torch.ones_like(x) - x)
#
#     def log_taylor_series(self, x):
#         approximation = 0
#         for i in range(1, self.order + 1):
#             approximation -= ((1 - x) ** i) / i
#         return approximation
#
#     def forward(self, pred, positive_mask, unlabeled_mask, ema_label, epoch, device):
#         positive_mask = positive_mask.unsqueeze(dim=0).float()
#         device = positive_mask.device
#         unlabeled_mask = unlabeled_mask.unsqueeze(dim=0).float()
#         ema_label = ema_label.unsqueeze(dim=0).float()
#         log_sigmoid = F.logsigmoid(pred)
#         positive_loss = -1 * (log_sigmoid * positive_mask).sum() / positive_mask.sum()
#         # positive_loss_extra = ((1 - torch.exp(log_sigmoid)) * positive_mask).sum() / positive_mask.sum()
#         # positive_loss += positive_loss+self.positive_taylor_series(torch.exp(log_sigmoid)).sum()/positive_mask.sum()
#
#         exp_phi = (torch.exp(log_sigmoid) * ema_label * unlabeled_mask).sum() / torch.sum(unlabeled_mask)
#         taylor_var_unlabeled_loss = self.log_taylor_series(exp_phi)
#         bce_unlabeled_loss = (torch.nn.BCEWithLogitsLoss(reduction='none')(pred, (
#                 1 - unlabeled_mask)) * ema_label * unlabeled_mask).sum() / unlabeled_mask.sum()
#
#         weight = torch.sigmoid(torch.tensor(self.m * (epoch - self.k))).to(device)
#         # if epoch>140:
#         #     weight = 0
#
#         unlabeled_loss = (1 - weight) * bce_unlabeled_loss + weight * taylor_var_unlabeled_loss
#
#         loss = unlabeled_loss + positive_loss
#
#         return loss, positive_loss, unlabeled_loss


@LOSS_FUNCTIONS.register_module()
class AsyVarPULossPf(nn.Module):

    def __init__(self, asy):
        super(AsyVarPULossPf, self).__init__()
        self.asy = asy

    def forward(self, pred, positive_mask, unlabeled_mask, epoch, device):
        positive_mask = positive_mask.unsqueeze(dim=0).float()
        device = positive_mask.device
        unlabeled_mask = unlabeled_mask.unsqueeze(dim=0).float()
        log_sigmoid = F.logsigmoid(pred)
        positive_loss = -1 * (log_sigmoid * positive_mask).sum() / positive_mask.sum()
        unlabeled_loss = torch.log(
            ((torch.exp(log_sigmoid) + torch.tensor(self.asy).cuda()) * unlabeled_mask).sum() / torch.sum(
                unlabeled_mask))

        loss = positive_loss + unlabeled_loss

        return loss, positive_loss, unlabeled_loss
