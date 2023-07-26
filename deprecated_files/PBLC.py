import torch
import torch.nn as nn


class PBLC(nn.Module):
    def __init__(self, lam=0, pmax=0.9):
        super(PBLC, self).__init__()
        self.lam = lam
        self.pmax = pmax
        self.__set_random_seed()
        self.c = nn.Parameter(torch.rand(1), requires_grad=True)

    def __set_random_seed(self, seed=2333):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def forward(self, x, y):
        x = torch.sigmoid(x)/(torch.sigmoid(x) + (1 - self.c) / self.c)
        # loss = -1 * torch.sum(y * torch.log(x) + (1 - y) * torch.log(1 - x))
        loss = torch.nn.BCELoss()(x, y) + self.lam * torch.abs(torch.max(x) - self.pmax)
        return loss
