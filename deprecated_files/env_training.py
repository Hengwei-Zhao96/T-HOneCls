import torch
from HOC.loss_functions.pf_bce_loss import BCELoss


def env_warm_up_training(optimizer, target, action, positive_train_mask, unlabeled_train_mask):
    positive_train_mask = torch.unsqueeze(positive_train_mask, dim=0)
    unlabeled_train_mask = torch.unsqueeze(unlabeled_train_mask, dim=0)

    positive_mask = positive_train_mask
    negative_mask = torch.clamp(unlabeled_train_mask - action * unlabeled_train_mask, 0, 1)

    positive_mask = torch.squeeze(positive_mask, dim=0)
    negative_mask = torch.squeeze(negative_mask, dim=0)
    loss, estimated_p_loss, estimated_u_loss = BCELoss(equal_weight=True)(target, positive_mask,negative_mask)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, estimated_p_loss, estimated_u_loss


def env_training(optimizer, target, action, positive_train_mask, unlabeled_train_mask, model='Separator'):
    positive_train_mask = torch.unsqueeze(positive_train_mask, dim=0)
    unlabeled_train_mask = torch.unsqueeze(unlabeled_train_mask, dim=0)
    if model == 'Separator':
        if action.ndim == 5:
            action = torch.where(action[:, :, :, :, 1] > 0.5, 1, 0)
        # positive_mask = torch.clamp(positive_train_mask + action * unlabeled_train_mask, 0, 1)
        positive_mask = positive_train_mask
        negative_mask = torch.clamp(unlabeled_train_mask - action * unlabeled_train_mask, 0, 1)

        positive_mask = torch.squeeze(positive_mask, dim=0)
        negative_mask = torch.squeeze(negative_mask, dim=0)
        loss, estimated_p_loss, estimated_u_loss = BCELoss(equal_weight=True)(target, positive_mask,
                                                                                    negative_mask)
        # loss, estimated_p_loss, estimated_u_loss = AsyVarPULoss(asy=1)(target, positive_mask,
        #                                                                       negative_mask)
        # loss, estimated_p_loss, estimated_u_loss, _, _ = AsymmetricPULoss()(target, positive_mask,
        #                                                                     negative_mask)
        # if first_stage:
        #     loss, estimated_p_loss, estimated_u_loss, _, _ = AsymmetricPULoss()(target, positive_mask,
        #                                                                         negative_mask)
        # else:
        #     loss, estimated_p_loss, estimated_u_loss, _, _ = BCELoss(equal_weight=True)(target, positive_mask,
        #                                                                                 negative_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, estimated_p_loss, estimated_u_loss
    else:
        pass
