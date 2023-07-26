import copy
import os

import numpy as np
import torch
from HOC.utils import confusion_matrix, mcc_evaluate_metric, pre_rec_f1, roc_auc, classmap2rgbmap, palette_class_mapping


def cal_metric_save_map_closeset(test_gt, pred, novel_class_index, pal_map, palette, save_path, epoch):
    rgb_pred = palette_class_mapping(pred, pal_map)
    cls_fig = classmap2rgbmap(rgb_pred, palette, cls='mcc')
    cls_fig.save(os.path.join(save_path, 'closeset_' + str(epoch) + '.png'))

    test_gt = torch.from_numpy(test_gt)
    pred = torch.from_numpy(pred)

    mask = copy.deepcopy(test_gt)
    mask[test_gt == novel_class_index] = 0
    mask = test_gt.bool()

    close_gt = torch.masked_select(test_gt.view(-1), mask.view(-1)).numpy() - 1
    close_pred = torch.masked_select(pred.view(-1), mask.view(-1)).numpy() - 1
    close_c_matrix = confusion_matrix(close_pred, close_gt, novel_class_index - 1)
    close_acc, close_acc_per_class, close_acc_cls, close_IoU, close_mean_IoU, close_kappa = mcc_evaluate_metric(
        close_c_matrix)

    return close_acc, close_acc_cls, close_kappa


def cal_metric_save_map_openset(test_gt, pred_pro, pred, novel_class_index, pal_map, palette, save_path, epoch):
    rgb_pred = palette_class_mapping(pred, pal_map)
    cls_fig = classmap2rgbmap(rgb_pred, palette, cls='mcc')
    cls_fig.save(os.path.join(save_path, 'openset_' + str(epoch) + '.png'))

    test_gt = torch.from_numpy(test_gt)
    pred_pro = torch.from_numpy(pred_pro)
    pred = torch.from_numpy(pred)
    mask = test_gt.bool()

    open_gt = torch.masked_select(test_gt.contiguous().view(-1), mask.contiguous().view(-1)).numpy() - 1
    open_pred = torch.masked_select(pred.contiguous().view(-1), mask.contiguous().view(-1)).numpy() - 1
    open_c_matrix = confusion_matrix(open_pred, open_gt, novel_class_index)
    open_acc, open_acc_per_class, open_acc_cls, open_IoU, open_mean_IoU, open_kappa = mcc_evaluate_metric(
        open_c_matrix)

    os_gt = copy.deepcopy(test_gt)
    os_gt[test_gt != novel_class_index] = 0
    os_gt[test_gt == novel_class_index] = 1
    os_gt = torch.masked_select(os_gt.view(-1), mask.view(-1)).numpy()

    os_pred_pro = copy.deepcopy(pred_pro)
    os_pred_pro = torch.masked_select(os_pred_pro.contiguous().view(-1), mask.contiguous().view(-1)).numpy()
    auc, fpr, tpr, threshold = roc_auc(np.nan_to_num(os_pred_pro), os_gt)

    os_pred = copy.deepcopy(pred)
    os_pred[pred == novel_class_index] = 1
    os_pred[pred != novel_class_index] = 0
    os_pred = torch.masked_select(os_pred.contiguous().view(-1), mask.contiguous().view(-1)).numpy()
    os_pre, os_rec, os_f1 = pre_rec_f1(os_pred, os_gt)

    return open_acc, open_acc_cls, open_kappa, os_pre, os_rec, os_f1, auc, fpr, tpr, threshold
