import torch


def sigmoid_loss(pred):
    return torch.sigmoid(-pred)


def bce_loss(pred, target, positive=True):
    if positive:
        return torch.nn.BCEWithLogitsLoss(reduction='none')(pred, target)
    else:
        return torch.nn.BCEWithLogitsLoss(reduction='none')(pred, (1 - target))


def sigmoid_mse_loss(input, target):
    input_sigmoid = torch.sigmoid(input)
    target_sigmoid = torch.sigmoid(target)
    loss = torch.nn.MSELoss()(input_sigmoid, target_sigmoid)
    return loss
