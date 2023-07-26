import torch


def pan_training(target_d, target_c, positive_train_mask, unlabeled_train_mask, weight, adversarial=True):
    eps = 1e-8

    prob_target_d = torch.sigmoid(target_d)
    prob_target_c = torch.sigmoid(target_c)

    # prob_target_d_ = prob_target_d.clone().detach()

    loss_1 = torch.sum(torch.log(prob_target_d + eps) * positive_train_mask) / positive_train_mask.sum() + \
             torch.sum(torch.log(torch.ones_like(
                 positive_train_mask) - prob_target_d + eps) * unlabeled_train_mask) / unlabeled_train_mask.sum()

    # u_mean = (prob_target_d_ * unlabeled_train_mask).sum() / unlabeled_train_mask.sum()

    # loss_1 = torch.sum(torch.log(prob_target_d + eps) * positive_train_mask) / positive_train_mask.sum() + torch.sum(torch.max((torch.log(torch.ones_like(positive_train_mask) - prob_target_d + eps) -torch.log(1 - u_mean))*unlabeled_train_mask, torch.tensor(0).cuda()))/unlabeled_train_mask.sum()

    loss_2 = torch.sum(
        (torch.log(torch.ones_like(positive_train_mask) - prob_target_c + eps) - torch.log(prob_target_c + eps)) * (
                2 * prob_target_c - 1) * unlabeled_train_mask) / unlabeled_train_mask.sum()

    if not adversarial:
        return -(loss_1 + weight * loss_2)
    else:
        return weight * loss_2
