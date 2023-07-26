import torch
from HOC.loss_functions.pf_bce_loss import BCELoss


def agent_warm_up_training(optimizer, target, action, positive_train_mask, unlabeled_train_mask):
    positive_train_mask = torch.unsqueeze(positive_train_mask, dim=0)
    unlabeled_train_mask = torch.unsqueeze(unlabeled_train_mask, dim=0)

    positive_mask = positive_train_mask
    negative_mask = torch.clamp(unlabeled_train_mask - action * unlabeled_train_mask, 0, 1)

    positive_mask = torch.squeeze(positive_mask, dim=0)
    negative_mask = torch.squeeze(negative_mask, dim=0)
    loss, estimated_p_loss, estimated_u_loss = BCELoss(equal_weight=True)(target, positive_mask, negative_mask)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, estimated_p_loss, estimated_u_loss


def agent_training_previous(agent, optimizer, logits, positive_train_mask, unlabeled_train_mask):
    prob = torch.sigmoid(logits)
    positive_train_mask = torch.unsqueeze(positive_train_mask, dim=0)
    unlabeled_train_mask = torch.unsqueeze(unlabeled_train_mask, dim=0)

    threshold_min = torch.min(
        prob * positive_train_mask + (torch.ones_like(positive_train_mask) - positive_train_mask) * 100)
    threshold_mask = torch.where(prob > threshold_min, 1, 0) * torch.clip(positive_train_mask + unlabeled_train_mask, 0,
                                                                          1)

    # threshold = (prob * threshold_mask).sum() / threshold_mask.sum()
    threshold = 0.5
    predict_mask = torch.where(prob > threshold, 1, 0)

    unlabeled_positive_mask = predict_mask * unlabeled_train_mask

    positive_mask = torch.clamp(positive_train_mask + unlabeled_positive_mask, 0, 1)
    negative_mask = unlabeled_train_mask - unlabeled_positive_mask

    reward = prob * positive_mask + (torch.ones_like(prob) - prob) * negative_mask

    agent.rewards.append(reward)
    # positive_loss = -1 * (
    #         agent.log_probs[-1] * agent.rewards[-1] * unlabeled_positive_mask).sum() / unlabeled_positive_mask.sum()
    # negative_loss = -1 * (agent.log_probs[-1] * agent.rewards[-1] * negative_mask).sum() / negative_mask.sum()
    #
    # loss = positive_loss + negative_loss

    loss = -(torch.log(agent.probability[-1][:, :, :, :, 1]+torch.tensor(0.00001)) * agent.rewards[
        -1] * positive_mask).sum() / positive_mask.sum() - (
                       torch.log(agent.probability[-1][:, :, :, :, 0]+torch.tensor(0.00001)) * agent.rewards[
                   -1] * negative_mask).sum() / negative_mask.sum()

    # loss = -1 * ((agent.log_probs[-1] * agent.rewards[-1] * unlabeled_positive_mask).sum() + (
    #         torch.log(torch.ones_like(agent.log_probs[-1]) - torch.exp(agent.log_probs[-1])) * agent.rewards[
    #     -1] * negative_mask).sum()) / unlabeled_train_mask.sum()

    # loss = -1 * (agent.log_probs[-1] * agent.rewards[-1] * unlabeled_train_mask).sum() / unlabeled_train_mask.sum()
    # print(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


