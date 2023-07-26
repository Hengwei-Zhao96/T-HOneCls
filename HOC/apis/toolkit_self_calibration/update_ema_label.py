import torch


def update_ema_labels(unlabeled_mask, ema_label, target_t, alpha, warm_up_epoch, epoch):
    if epoch > warm_up_epoch:
        new_ema_label = alpha * ema_label + (1 - alpha) * (1 - torch.sigmoid(target_t))
    else:
        new_ema_label = torch.ones_like(unlabeled_mask)
    return new_ema_label
