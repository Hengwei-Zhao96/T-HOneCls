import torch


def logits_norm(logits):
    normalized_1 = torch.sigmoid(logits)
    normalized_0 = torch.ones_like(normalized_1) - normalized_1
    normalized = torch.hstack([normalized_0, normalized_1])
    return normalized
