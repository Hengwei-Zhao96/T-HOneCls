import torch.nn as nn
import torch.nn.functional as F
from HOC.utils import LOSS_FUNCTIONS


@LOSS_FUNCTIONS.register_module()
class DS3L_MSE_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, model, mask):
        y_hat = model(x)
        return (F.mse_loss(y_hat.softmax(1), y.softmax(1).detach(), reduction='none').mean(1) * mask)
