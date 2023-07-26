import torch


def get_small_loss_unlabeled_samples(unlabeled_train_mask, loss_mask, ratio):
    unlabeled_train_mask = unlabeled_train_mask.unsqueeze(dim=0).float()
    unlabeled_loss_mask = unlabeled_train_mask * loss_mask
    unlabeled_loss_mask = unlabeled_loss_mask[unlabeled_train_mask != 0]

    sorted, indx = unlabeled_loss_mask.sort(descending=True)

    clean_positive_number = int(unlabeled_train_mask.sum() * ratio)
    threshold = sorted[clean_positive_number]

    small_loss_unlabeled_train_mask = torch.squeeze(torch.where(loss_mask <= threshold, 1, 0) * unlabeled_train_mask,
                                                    dim=0)

    return small_loss_unlabeled_train_mask.detach()
