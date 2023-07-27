import os

import numpy as np
import torch

from HOC.utils import all_metric
from HOC.utils import classmap2rgbmap


def fcn_evaluate_fn(model, test_dataloader, meta, cls, path, device, epoch):
    save_rgb_path = os.path.join(path, 'classmap')
    if not os.path.exists(save_rgb_path):
        os.makedirs(save_rgb_path)
    model.eval()
    with torch.no_grad():
        for (im, positive_test_mask, negative_test_mask) in test_dataloader:
            im = im.to(device)
            positive_test_mask = positive_test_mask.squeeze()
            negative_test_mask = negative_test_mask.squeeze()
            pred_pro = torch.sigmoid(model(im)).squeeze().cpu()

            pred_class = torch.where(pred_pro > 0.5, 1, 0)

            cls_fig = classmap2rgbmap(
                pred_class[:meta['image_size'][0], :meta['image_size'][1]].numpy(),
                palette=meta['palette'], cls=cls)
            cls_fig.save(os.path.join(save_rgb_path, str(epoch + 1) + '.png'))

            np.save(os.path.join(path, 'probaility.npy'),
                    pred_pro[:meta['image_size'][0], :meta['image_size'][1]].numpy())

            mask = (positive_test_mask + negative_test_mask).bool()

            label = positive_test_mask
            target = torch.masked_select(label.view(-1), mask.view(-1)).numpy()
            pred_class = torch.masked_select(pred_class.view(-1).cpu(), mask.view(-1)).numpy()
            pred_pro = torch.masked_select(pred_pro.view(-1).cpu(), mask.view(-1)).numpy()

            auc, fpr, tpr, threshold, pre, rec, f1 = all_metric(pred_pro, pred_class, target)

    return auc, fpr, tpr, threshold, pre, rec, f1
