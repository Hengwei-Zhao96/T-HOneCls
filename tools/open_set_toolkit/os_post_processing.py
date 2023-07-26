import os
import copy
import numpy as np
import torch

from HOC.utils import confusion_matrix, mcc_evaluate_metric, pre_rec_f1, classmap2rgbmap, palette_class_mapping


def cal_metric_save_map_closeset(test_gt, pred, novel_class_index, pal_map, palette, save_path):
    rgb_pred = palette_class_mapping(pred, pal_map)
    cls_fig = classmap2rgbmap(rgb_pred, palette, cls='mcc')
    cls_fig.save(os.path.join(save_path, 'closeset.png'))

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


def cal_metric_save_map_openset(test_gt, pred, novel_class_index, pal_map, palette, save_path):
    rgb_pred = palette_class_mapping(pred, pal_map)
    cls_fig = classmap2rgbmap(rgb_pred, palette, cls='mcc')
    cls_fig.save(os.path.join(save_path, 'openset.png'))

    test_gt = torch.from_numpy(test_gt)
    pred = torch.from_numpy(pred)
    mask = test_gt.bool()

    open_gt = torch.masked_select(test_gt.view(-1), mask.view(-1)).numpy() - 1
    open_pred = torch.masked_select(pred.view(-1), mask.view(-1)).numpy() - 1
    open_c_matrix = confusion_matrix(open_pred, open_gt, novel_class_index)
    open_acc, open_acc_per_class, open_acc_cls, open_IoU, open_mean_IoU, open_kappa = mcc_evaluate_metric(
        open_c_matrix)

    os_gt = copy.deepcopy(test_gt)
    os_gt[test_gt != novel_class_index] = 0
    os_gt[test_gt == novel_class_index] = 1
    os_gt = torch.masked_select(os_gt.view(-1), mask.view(-1)).numpy()
    os_pred = copy.deepcopy(pred)
    os_pred[pred == novel_class_index] = 1
    os_pred[pred != novel_class_index] = 0
    os_pred = torch.masked_select(os_pred.view(-1), mask.view(-1)).numpy()
    os_pre, os_rec, os_f1 = pre_rec_f1(os_pred, os_gt)

    return open_acc, open_acc_cls, open_kappa, os_pre, os_rec, os_f1


if __name__ == "__main__":
    pro_pred_path = "/home/zhw2021/code/HOneCls/tools/Logtest/FPGA/MccLongKouDataset/MccSingleModelTrainer_PB/CELossPb/FreeNetEncoder/2023-06-08 21-33-49/probaility.npy"
    gt_path = "/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-LongKou/Test100_new_open_label.npy"
    save_path = "/home/zhw2021/code/HOneCls/tools/Logtest/FPGA/MccLongKouDataset/MccSingleModelTrainer_PB/CELossPb/FreeNetEncoder/2023-06-08 21-33-49"
    threshold = 0.5

    pal_mapping = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 0}
    palette = [
        [0, 0, 0],
        [255, 0, 0],
        [238, 154, 0],
        [255, 255, 0],
        [0, 255, 0],
        [0, 255, 255],
        [0, 139, 139],
        [0, 0, 255],
        [255, 255, 255],
        [160, 32, 240]
    ]

    pred_pro = np.load(pro_pred_path)
    gt = np.load(gt_path)
    novel_class_index = gt.max()

    pred_close_class = np.argmax(pred_pro, axis=2) + 1

    close_acc, close_acc_cls, close_kappa = cal_metric_save_map_closeset(test_gt=gt,
                                                                         pred=pred_close_class,
                                                                         novel_class_index=novel_class_index,
                                                                         pal_map=pal_mapping,
                                                                         palette=palette,
                                                                         save_path=save_path)

    mask = pred_pro.max(axis=2)
    pred_open_class = copy.deepcopy(pred_close_class)
    pred_open_class[mask < threshold] = novel_class_index

    open_acc, open_acc_cls, open_kappa, os_pre, os_rec, os_f1 = cal_metric_save_map_openset(test_gt=gt,
                                                                                            pred=pred_open_class,
                                                                                            novel_class_index=novel_class_index,
                                                                                            pal_map=pal_mapping,
                                                                                            palette=palette,
                                                                                            save_path=save_path)

    logging_str = "Close_oa:%f, Close_aa:%f,Cloase_kappa:%f, Open_oa:%f, Open_aa:%f,Open_kappa:%f, Open_pre:%f, Open_rec:%f,Open_f1:%f" % (
        close_acc, close_acc_cls, close_kappa, open_acc, open_acc_cls, open_kappa, os_pre, os_rec, os_f1)
    print(logging_str)
