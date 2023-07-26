import torch
import numpy as np


def get_extra_mask(pred_pro, image_size_w, image_size_h, ratio):
    positive_mask = torch.zeros_like(pred_pro)
    negative_mask = torch.zeros_like(pred_pro)

    pred_pro = pred_pro[:image_size_w, :image_size_h]

    num_samples = pred_pro.numel()
    pred_class = torch.where(pred_pro > 0.5, 1, 0)
    positive_num = int(pred_class.sum())
    negative_num = int(num_samples - positive_num)
    pred_pro_flatten = pred_pro.flatten()

    positive_value, _ = torch.topk(pred_pro_flatten, max(int(positive_num * ratio), 1))
    positive_index = torch.where(pred_pro >= positive_value[-1:])
    positive_mask[positive_index[0], positive_index[1]] = 1

    negative_value, _ = torch.topk(pred_pro_flatten, max(int(negative_num * ratio), 1), largest=False)
    negative_index = torch.where(pred_pro <= negative_value[-1:])
    negative_mask[negative_index[0], negative_index[1]] = 1

    return positive_mask, negative_mask


if __name__ == "__main__":
    a = torch.randn((10, 10))
    b = a[[1, 2, 3], [1, 2, 3]]
    c, d = get_extra_mask(a, 9, 9, 0.5)
