import torch
from HOC.loss_functions.pf_bce_loss import BCELoss


def warming_up_training(optimizer, loss_fun, target, pse_label, positive_train_mask, unlabeled_train_mask):
    positive_train_mask = torch.unsqueeze(positive_train_mask, dim=0)
    unlabeled_train_mask = torch.unsqueeze(unlabeled_train_mask, dim=0)

    positive_mask = positive_train_mask
    if pse_label == None:
        negative_mask = unlabeled_train_mask
    else:
        negative_mask = torch.clamp(unlabeled_train_mask - pse_label * unlabeled_train_mask, 0, 1)

    positive_mask = torch.squeeze(positive_mask, dim=0)
    negative_mask = torch.squeeze(negative_mask, dim=0)

    loss, estimated_p_loss, estimated_n_loss = loss_fun(target, positive_mask, negative_mask)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, estimated_p_loss, estimated_n_loss


def agent_training(agent_optimizer, agent, env_logits, positive_train_mask, unlabeled_train_mask):
    env_prob = torch.sigmoid(env_logits)
    positive_train_mask = torch.unsqueeze(positive_train_mask, dim=0)
    unlabeled_train_mask = torch.unsqueeze(unlabeled_train_mask, dim=0)
    pse_label = torch.where(env_prob > 0.5, 1, 0)

    unlabeled_positive_mask = pse_label * unlabeled_train_mask
    positive_mask = torch.clamp(positive_train_mask + unlabeled_positive_mask, 0, 1)
    negative_mask = unlabeled_train_mask - unlabeled_positive_mask

    reward = env_prob * positive_mask + (torch.ones_like(env_prob) - env_prob) * negative_mask
    agent.rewards = reward

    # estimated_p_loss = -1 * (torch.log(agent.probability[:, :, :, :, 1] + torch.tensor(
    #     1e-8)) * agent.rewards * positive_mask).sum() / positive_mask.sum()
    # estimated_n_loss = -1 * (torch.log(agent.probability[:, :, :, :, 0] + torch.tensor(
    #     1e-8)) * agent.rewards * negative_mask).sum() / negative_mask.sum()

    estimated_p_loss = -1 * (agent.log_probs * agent.rewards * positive_mask).sum() / positive_mask.sum()
    # estimated_n_loss = -1 * (torch.log(
    #     1 - torch.exp(agent.log_probs) + 1e-8) * agent.rewards * negative_mask).sum() / negative_mask.sum()
    estimated_n_loss = torch.tensor(0)

    loss = estimated_p_loss + estimated_n_loss
    # print(loss.item())

    agent_optimizer.zero_grad()
    loss.backward()
    agent_optimizer.step()

    return loss, estimated_p_loss, estimated_n_loss


def env_training(env_optimizer, target, pse_label, positive_train_mask, unlabeled_train_mask, model='Separator'):
    positive_train_mask = torch.unsqueeze(positive_train_mask, dim=0)
    unlabeled_train_mask = torch.unsqueeze(unlabeled_train_mask, dim=0)
    if model == 'Separator':
        if pse_label.ndim == 5:
            pse_label = torch.where(pse_label[:, :, :, :, 1] > 0.5, 1, 0)
        positive_mask = positive_train_mask
        negative_mask = torch.clamp(unlabeled_train_mask - pse_label * unlabeled_train_mask, 0, 1)

        positive_mask = torch.squeeze(positive_mask, dim=0)
        negative_mask = torch.squeeze(negative_mask, dim=0)
        loss, estimated_p_loss, estimated_u_loss = BCELoss(equal_weight=True)(target, positive_mask, negative_mask)
        # loss, estimated_p_loss, estimated_u_loss = AsyVarPULoss(asy=1)(target, positive_mask, negative_mask)

        env_optimizer.zero_grad()
        loss.backward()
        env_optimizer.step()

        return loss, estimated_p_loss, estimated_u_loss

    else:
        pass
