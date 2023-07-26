import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(SEBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.seq = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        v = self.gap(x)
        score = self.seq(v.view(v.size(0), v.size(1)))
        y = x * score.view(score.size(0), score.size(1), 1, 1)
        return y
