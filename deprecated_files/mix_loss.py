import torch.nn as nn


class MixLoss(nn.Module):
    def __init__(self, loss_1, loss_2):
        super(MixLoss, self).__init__()
        self.loss_1 = loss_1
        self.loss_2 = loss_2

    def get_epoch(self, epoch, total_epoch):
        self.epoch = epoch
        self.total_epoch = total_epoch

    def forward(self, pred, positive_mask, unlabeled_mask):
        weight = (1 / self.total_epoch) * self.epoch
        loss_1_loss_value, loss_1_estimated_p_loss, loss_1_estimated_n_loss = self.loss_1(pred, positive_mask,
                                                                                          unlabeled_mask)
        loss_2_loss_value, loss_2_estimated_p_loss, loss_2_estimated_n_loss = self.loss_2(pred, positive_mask,
                                                                                          unlabeled_mask)
        loss_value = weight * loss_1_loss_value + (1 - weight) * loss_2_loss_value
        estimated_p_loss = weight * loss_1_estimated_p_loss + (1 - weight) * loss_2_estimated_p_loss
        estimated_n_loss = weight * loss_1_estimated_n_loss + (1 - weight) * loss_2_estimated_n_loss

        return loss_value, estimated_p_loss, estimated_n_loss
