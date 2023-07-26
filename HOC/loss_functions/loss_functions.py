import torch


def sigmoid_loss(pred):
    return torch.sigmoid(-pred)


def bce_loss(pred, target, positive=True):
    if positive:
        return torch.nn.BCEWithLogitsLoss(reduction='none')(pred, target)
    else:
        return torch.nn.BCEWithLogitsLoss(reduction='none')(pred, (1 - target))
