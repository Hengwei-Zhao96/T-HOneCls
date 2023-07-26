import copy
import os

import cv2
import numpy as np
import torch
from pytorch_ood.loss import CACLoss

from HOC.utils import all_metric, confusion_matrix, mcc_evaluate_metric, pre_rec_f1, roc_auc
from HOC.utils import classmap2rgbmap, palette_class_mapping
from HOC.apis.toolkit_openmax.validation_toolkits import cal_metric_save_map_closeset, cal_metric_save_map_openset


def sklearn_evaluate_fn(image, positive_test_indicator, negative_test_indicator, cls, model, meta, path, epoch):
    save_rgb_path = os.path.join(path, 'classmap')
    if not os.path.exists(save_rgb_path):
        os.makedirs(save_rgb_path)

    c, h, w = image.shape[0], positive_test_indicator.shape[0], positive_test_indicator.shape[1]
    image = image.transpose((1, 2, 0))
    classmap = np.zeros((h, w))
    promap = np.zeros((h, w))
    for i in range(h):
        data = image[i, :, :]
        classmap[i, :] = model.predict(data)
        promap[i] = model.predict_pro(data)

    classmap = classmap[:meta['image_size'][0], :meta['image_size'][1]]
    promap = promap[:meta['image_size'][0], :meta['image_size'][1]]

    cls_fig = classmap2rgbmap(classmap.astype(np.int64), meta['palette'], cls)
    cls_fig.save(os.path.join(save_rgb_path, str(epoch + 1) + '.png'))
    np.save(os.path.join(path, 'probaility.npy'), promap)

    positive_test_indicator = positive_test_indicator[:meta['image_size'][0], :meta['image_size'][1]]
    negative_test_indicator = negative_test_indicator[:meta['image_size'][0], :meta['image_size'][1]]

    mask = (torch.from_numpy(positive_test_indicator) + torch.from_numpy(negative_test_indicator)).bool()
    target = torch.masked_select(torch.from_numpy(positive_test_indicator).reshape(-1), mask.reshape(-1)).numpy()
    pred_class = torch.masked_select(torch.from_numpy(classmap).reshape(-1).cpu(), mask.reshape(-1)).numpy()
    pred_pro = torch.masked_select(torch.from_numpy(promap).reshape(-1).cpu(), mask.reshape(-1)).numpy()

    auc, fpr, tpr, threshold, pre, rec, f1 = all_metric(pred_pro, pred_class, target)

    return auc, fpr, tpr, threshold, pre, rec, f1


def evaluate_fn(image, positive_test_indicator, negative_test_indicator, cls, model, patch_size, meta, device, path,
                epoch):
    save_rgb_path = os.path.join(path, 'classmap')
    if not os.path.exists(save_rgb_path):
        os.makedirs(save_rgb_path)

    c, h, w = image.shape[0], positive_test_indicator.shape[0], positive_test_indicator.shape[1]
    classmap = np.zeros((h, w))
    promap = np.zeros((h, w))

    model.eval()
    with torch.no_grad():
        for i in range(h):
            data = np.zeros((w, c, patch_size, patch_size))
            for j in range(w):
                data[j, :, :, :] = image[:, i:i + patch_size, j:j + patch_size]
            data = torch.from_numpy(data).float().to(device)

            result = torch.sigmoid(model(data)).squeeze().cpu()

            promap[i] = result.numpy()
            classmap[i] = torch.where(result > 0.5, 1, 0).numpy()

    classmap = classmap[:meta['image_size'][0], :meta['image_size'][1]]
    promap = promap[:meta['image_size'][0], :meta['image_size'][1]]

    cls_fig = classmap2rgbmap(classmap.astype(np.int64), meta['palette'], cls)
    cls_fig.save(os.path.join(save_rgb_path, str(epoch + 1) + '.png'))
    np.save(os.path.join(path, 'probaility.npy'), promap)

    positive_test_indicator = positive_test_indicator[:meta['image_size'][0], :meta['image_size'][1]]
    negative_test_indicator = negative_test_indicator[:meta['image_size'][0], :meta['image_size'][1]]

    mask = (torch.from_numpy(positive_test_indicator) + torch.from_numpy(negative_test_indicator)).bool()
    target = torch.masked_select(torch.from_numpy(positive_test_indicator).reshape(-1), mask.reshape(-1)).numpy()
    pred_class = torch.masked_select(torch.from_numpy(classmap).reshape(-1).cpu(), mask.reshape(-1)).numpy()
    pred_pro = torch.masked_select(torch.from_numpy(promap).reshape(-1).cpu(), mask.reshape(-1)).numpy()

    auc, fpr, tpr, threshold, pre, rec, f1 = all_metric(pred_pro, pred_class, target)

    return auc, fpr, tpr, threshold, pre, rec, f1


def mcc_evaluate_fn(image, gt, mask, model, patch_size, meta, device, path, epoch):
    save_rgb_path = os.path.join(path, 'classmap')
    if not os.path.exists(save_rgb_path):
        os.makedirs(save_rgb_path)

    gt = torch.from_numpy(gt)
    mask = torch.from_numpy(mask).bool()
    max_cls = int(gt.max())

    c, h, w = image.shape[0], gt.shape[0], mask.shape[1]
    classmap = torch.zeros((h, w))
    promap = torch.zeros((h, w, max_cls))
    avmap = torch.zeros((h, w, max_cls))

    model.eval()
    with torch.no_grad():
        for i in range(h):
            data = np.zeros((w, c, patch_size, patch_size))
            for j in range(w):
                data[j, :, :, :] = image[:, i:i + patch_size, j:j + patch_size]
            data = torch.from_numpy(data).float().to(device)

            av_results = model(data)
            pro_results = torch.softmax(av_results, dim=1)
            class_results = torch.argmax(av_results, dim=1) + torch.tensor(1, dtype=torch.int)

            avmap[i] = av_results.cpu()
            promap[i] = pro_results.cpu()
            classmap[i] = class_results.cpu()

    avmap = avmap[:meta['image_size'][0], :meta['image_size'][1], :]
    classmap = classmap[:meta['image_size'][0], :meta['image_size'][1]]
    promap = promap[:meta['image_size'][0], :meta['image_size'][1], :]

    gt = gt[:meta['image_size'][0], :meta['image_size'][1]]
    mask = mask[:meta['image_size'][0], :meta['image_size'][1]]

    cls_fig = classmap2rgbmap(classmap.numpy().astype(np.int64), meta['palette'], cls='mcc')
    cls_fig.save(os.path.join(save_rgb_path, str(epoch + 1) + '.png'))
    np.save(os.path.join(path, 'av.npy'), avmap.numpy())
    np.save(os.path.join(path, 'probaility.npy'), promap.numpy())

    masked_gt = torch.masked_select(gt.contiguous().view(-1), mask.contiguous().view(-1)).cpu().numpy() - 1
    pred_class = torch.masked_select(classmap.contiguous().view(-1), mask.contiguous().view(-1)).cpu().numpy() - 1

    c_matrix = confusion_matrix(pred_class.astype(np.int64), masked_gt.astype(np.int64), max_cls)
    acc, acc_per_class, acc_cls, IoU, mean_IoU, kappa = mcc_evaluate_metric(c_matrix)

    return acc, acc_per_class, acc_cls, IoU, mean_IoU, kappa


def os_openmax_evaluate_fn(image, gt, mask, model, detector, patch_size, meta, device, path, epoch):
    save_rgb_path = os.path.join(path, 'classmap')
    if not os.path.exists(save_rgb_path):
        os.makedirs(save_rgb_path)

    gt = torch.from_numpy(gt)
    mask = torch.from_numpy(mask).bool()
    max_cls = int(gt.max())

    c, h, w = image.shape[0], gt.shape[0], mask.shape[1]
    mcc_classmap = torch.zeros((h, w))
    mcc_promap = torch.zeros((h, w, max_cls - 1))
    os_mcc_classmap = torch.zeros((h, w))
    os_mcc_promap = torch.zeros((h, w, max_cls))

    for i in range(h):
        data = np.zeros((w, c, patch_size, patch_size))
        _os_mcc_promap = torch.zeros((w, max_cls))
        for j in range(w):
            data[j, :, :, :] = image[:, i:i + patch_size, j:j + patch_size]
        data = torch.from_numpy(data).float().to(device)

        model.eval()
        with torch.no_grad():
            mcc_pro_results = torch.softmax(model(data), dim=1)
        os_pro_results = detector.predict(data)

        _os_mcc_promap[:, :-1] = os_pro_results[:, 1:]
        _os_mcc_promap[:, -1] = os_pro_results[:, 0]

        class_results = torch.argmax(mcc_pro_results, dim=1) + torch.tensor(1, dtype=torch.int)
        os_class_results = torch.argmax(_os_mcc_promap, dim=1) + torch.tensor(1, dtype=torch.int)

        mcc_promap[i] = mcc_pro_results.cpu()
        mcc_classmap[i] = class_results.cpu()
        os_mcc_promap[i] = _os_mcc_promap.cpu()
        os_mcc_classmap[i] = os_class_results.cpu()

    mcc_classmap = mcc_classmap[:meta['image_size'][0], :meta['image_size'][1]]
    mcc_promap = mcc_promap[:meta['image_size'][0], :meta['image_size'][1], :]
    os_mcc_classmap = os_mcc_classmap[:meta['image_size'][0], :meta['image_size'][1]]
    os_mcc_promap = os_mcc_promap[:meta['image_size'][0], :meta['image_size'][1], :]

    gt = gt[:meta['image_size'][0], :meta['image_size'][1]]

    np.save(os.path.join(path, 'mcc_probaility.npy'), mcc_promap.numpy())
    np.save(os.path.join(path, 'os_mcc_probaility.npy'), os_mcc_promap.numpy())

    close_acc, close_acc_cls, close_kappa = cal_metric_save_map_closeset(test_gt=gt.numpy().astype(np.int64),
                                                                         pred=mcc_classmap.numpy().astype(np.int64),
                                                                         novel_class_index=max_cls,
                                                                         pal_map=meta['palette_class_mapping'],
                                                                         palette=meta['palette'],
                                                                         save_path=save_rgb_path,
                                                                         epoch=epoch)

    open_acc, open_acc_cls, open_kappa, os_pre, os_rec, os_f1, auc, fpr, tpr, threshold = cal_metric_save_map_openset(
        test_gt=gt.numpy().astype(np.int64),
        pred=os_mcc_classmap.numpy().astype(np.int64),
        pred_pro=os_mcc_promap[:, :, -1].numpy(),
        novel_class_index=max_cls,
        pal_map=meta['palette_class_mapping'],
        palette=meta['palette'],
        save_path=save_rgb_path,
        epoch=epoch)

    return close_acc, close_acc_cls, close_kappa, open_acc, open_acc_cls, open_kappa, os_pre, os_rec, os_f1, auc, fpr, tpr, threshold


def os_cac_evaluate_fn(image, gt, mask, model, patch_size, meta, device, path, loss_function, epoch):
    save_rgb_path = os.path.join(path, 'classmap')
    if not os.path.exists(save_rgb_path):
        os.makedirs(save_rgb_path)

    gt = torch.from_numpy(gt)
    mask = torch.from_numpy(mask).bool()
    max_cls = int(gt.max())

    c, h, w = image.shape[0], gt.shape[0], mask.shape[1]
    mcc_classmap = torch.zeros((h, w))
    os_score = torch.zeros((h, w))

    for i in range(h):
        data = np.zeros((w, c, patch_size, patch_size))
        for j in range(w):
            data[j, :, :, :] = image[:, i:i + patch_size, j:j + patch_size]
        data = torch.from_numpy(data).float().to(device)

        model.eval()
        with torch.no_grad():
            mcc_logit_results = model(data)
        distances = loss_function.distance(mcc_logit_results)
        class_results = distances.min(dim=1).indices + torch.tensor(1, dtype=torch.int)

        mcc_classmap[i] = class_results.cpu()
        os_score[i] = CACLoss.score(distances).cpu()

    mcc_classmap = mcc_classmap[:meta['image_size'][0], :meta['image_size'][1]]
    os_score = os_score[:meta['image_size'][0], :meta['image_size'][1]]

    gt = gt[:meta['image_size'][0], :meta['image_size'][1]]

    cv2_os_score = cv2.normalize(os_score.numpy(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    value, _ = cv2.threshold(cv2_os_score, 0, 1, cv2.THRESH_OTSU)
    os_mask = np.where(cv2_os_score > value, 1, 0)

    os_mcc_classmap = copy.deepcopy(mcc_classmap)
    os_mcc_classmap[os_mask == 1] = max_cls

    close_acc, close_acc_cls, close_kappa = cal_metric_save_map_closeset(test_gt=gt.numpy().astype(np.int64),
                                                                         pred=mcc_classmap.numpy().astype(np.int64),
                                                                         novel_class_index=max_cls,
                                                                         pal_map=meta['palette_class_mapping'],
                                                                         palette=meta['palette'],
                                                                         save_path=save_rgb_path,
                                                                         epoch=epoch)

    open_acc, open_acc_cls, open_kappa, os_pre, os_rec, os_f1, auc, fpr, tpr, threshold = cal_metric_save_map_openset(
        test_gt=gt.numpy().astype(np.int64),
        pred_pro=os_score.numpy(),
        pred=os_mcc_classmap.numpy().astype(np.int64),
        novel_class_index=max_cls,
        pal_map=meta['palette_class_mapping'],
        palette=meta['palette'],
        save_path=save_rgb_path,
        epoch=epoch)

    return close_acc, close_acc_cls, close_kappa, open_acc, open_acc_cls, open_kappa, os_pre, os_rec, os_f1, auc, fpr, tpr, threshold


def os_ii_evaluate_fn(image, gt, mask, model, patch_size, meta, device, path, loss_function, epoch):
    save_rgb_path = os.path.join(path, 'classmap')
    if not os.path.exists(save_rgb_path):
        os.makedirs(save_rgb_path)

    gt = torch.from_numpy(gt)
    mask = torch.from_numpy(mask).bool()
    max_cls = int(gt.max())

    c, h, w = image.shape[0], gt.shape[0], mask.shape[1]
    mcc_classmap = torch.zeros((h, w))
    os_score = torch.zeros((h, w))

    for i in range(h):
        data = np.zeros((w, c, patch_size, patch_size))
        for j in range(w):
            data[j, :, :, :] = image[:, i:i + patch_size, j:j + patch_size]
        data = torch.from_numpy(data).float().to(device)

        model.eval()
        with torch.no_grad():
            mcc_logit_results = model(data)
        distances = loss_function.distance(mcc_logit_results).min(dim=1).values
        class_results = loss_function.predict(mcc_logit_results).max(dim=1).indices + torch.tensor(1, dtype=torch.int)

        mcc_classmap[i] = class_results.cpu()
        os_score[i] = distances.cpu()

    mcc_classmap = mcc_classmap[:meta['image_size'][0], :meta['image_size'][1]]
    os_score = os_score[:meta['image_size'][0], :meta['image_size'][1]]

    gt = gt[:meta['image_size'][0], :meta['image_size'][1]]

    cv2_os_score = cv2.normalize(os_score.detach().numpy(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    value, _ = cv2.threshold(cv2_os_score, 0, 1, cv2.THRESH_OTSU)
    os_mask = np.where(cv2_os_score > value, 1, 0)

    os_mcc_classmap = copy.deepcopy(mcc_classmap)
    os_mcc_classmap[os_mask == 1] = max_cls

    close_acc, close_acc_cls, close_kappa = cal_metric_save_map_closeset(test_gt=gt.numpy().astype(np.int64),
                                                                         pred=mcc_classmap.numpy().astype(np.int64),
                                                                         novel_class_index=max_cls,
                                                                         pal_map=meta['palette_class_mapping'],
                                                                         palette=meta['palette'],
                                                                         save_path=save_rgb_path,
                                                                         epoch=epoch)

    open_acc, open_acc_cls, open_kappa, os_pre, os_rec, os_f1, auc, fpr, tpr, threshold = cal_metric_save_map_openset(
        test_gt=gt.numpy().astype(np.int64),
        pred=os_mcc_classmap.numpy().astype(np.int64),
        pred_pro=os_score.detach().numpy(),
        novel_class_index=max_cls,
        pal_map=meta['palette_class_mapping'],
        palette=meta['palette'],
        save_path=save_rgb_path,
        epoch=epoch)

    return close_acc, close_acc_cls, close_kappa, open_acc, open_acc_cls, open_kappa, os_pre, os_rec, os_f1, auc, fpr, tpr, threshold


def os_pytorchood_detector_evaluate_fn(image, gt, mask, model, detector, patch_size, meta, device, path, epoch):
    save_rgb_path = os.path.join(path, 'classmap')
    if not os.path.exists(save_rgb_path):
        os.makedirs(save_rgb_path)

    gt = torch.from_numpy(gt)
    mask = torch.from_numpy(mask).bool()
    max_cls = int(gt.max())

    c, h, w = image.shape[0], gt.shape[0], mask.shape[1]
    mcc_classmap = torch.zeros((h, w))
    mcc_promap = torch.zeros((h, w, max_cls - 1))
    os_score = torch.zeros((h, w))

    for i in range(h):
        data = np.zeros((w, c, patch_size, patch_size))
        for j in range(w):
            data[j, :, :, :] = image[:, i:i + patch_size, j:j + patch_size]
        data = torch.from_numpy(data).float().to(device)

        model.eval()
        with torch.no_grad():
            mcc_pro_results = torch.softmax(model(data), dim=1)
            os_pro_results = detector(data)

        class_results = torch.argmax(mcc_pro_results, dim=1) + torch.tensor(1, dtype=torch.int)

        mcc_promap[i] = mcc_pro_results.cpu()
        mcc_classmap[i] = class_results.cpu()
        os_score[i] = os_pro_results.cpu()

    mcc_classmap = mcc_classmap[:meta['image_size'][0], :meta['image_size'][1]]
    mcc_promap = mcc_promap[:meta['image_size'][0], :meta['image_size'][1], :]
    os_score = os_score[:meta['image_size'][0], :meta['image_size'][1]]

    gt = gt[:meta['image_size'][0], :meta['image_size'][1]]

    np.save(os.path.join(path, 'mcc_probaility.npy'), mcc_promap.numpy())

    cv2_os_score = cv2.normalize(os_score.detach().numpy(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    value, _ = cv2.threshold(cv2_os_score, 0, 1, cv2.THRESH_OTSU)
    os_mask = np.where(cv2_os_score > value, 1, 0)

    os_mcc_classmap = copy.deepcopy(mcc_classmap)
    os_mcc_classmap[os_mask == 1] = max_cls

    close_acc, close_acc_cls, close_kappa = cal_metric_save_map_closeset(test_gt=gt.numpy().astype(np.int64),
                                                                         pred=mcc_classmap.numpy().astype(np.int64),
                                                                         novel_class_index=max_cls,
                                                                         pal_map=meta['palette_class_mapping'],
                                                                         palette=meta['palette'],
                                                                         save_path=save_rgb_path,
                                                                         epoch=epoch)

    open_acc, open_acc_cls, open_kappa, os_pre, os_rec, os_f1, auc, fpr, tpr, threshold = cal_metric_save_map_openset(
        test_gt=gt.numpy().astype(np.int64),
        pred_pro=os_score.numpy(),
        pred=os_mcc_classmap.numpy().astype(np.int64),
        novel_class_index=max_cls,
        pal_map=meta['palette_class_mapping'],
        palette=meta['palette'],
        save_path=save_rgb_path,
        epoch=epoch)

    return close_acc, close_acc_cls, close_kappa, open_acc, open_acc_cls, open_kappa, os_pre, os_rec, os_f1, auc, fpr, tpr, threshold


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


def mcc_fcn_evaluate_fn(model, test_dataloader, meta, path, device, epoch):
    save_rgb_path = os.path.join(path, 'classmap')
    if not os.path.exists(save_rgb_path):
        os.makedirs(save_rgb_path)
    model.eval()
    with torch.no_grad():
        for (im, mask, weight) in test_dataloader:
            im = im.to(device)

            pred_pro = torch.softmax(model(im), dim=1).squeeze().cpu()
            pred_class = pred_pro.argmax(dim=0) + torch.tensor(1, dtype=torch.int)

            cls_fig = classmap2rgbmap(
                pred_class[:meta['image_size'][0], :meta['image_size'][1]].numpy(),
                palette=meta['palette'], cls='mcc')
            cls_fig.save(os.path.join(save_rgb_path, str(epoch + 1) + '.png'))

            np.save(os.path.join(path, 'probaility.npy'),
                    pred_pro[:, :meta['image_size'][0], :meta['image_size'][1]].numpy())

            weight.unsqueeze_(dim=0)
            weight = weight.bool()

            mask = torch.masked_select(mask.view(-1), weight.view(-1)).cpu().numpy() - 1
            pred_class = torch.masked_select(pred_class.view(-1), weight.view(-1)).cpu().numpy() - 1

            c_matrix = confusion_matrix(pred_class, mask, test_dataloader.dataset.num_classes)
            acc, acc_per_class, acc_cls, IoU, mean_IoU, kappa = mcc_evaluate_metric(c_matrix)

    return acc, acc_per_class, acc_cls, IoU, mean_IoU, kappa


def os_mcc_fcn_evaluate_fn(model, test_dataloader, meta, path, device, epoch):
    rgb_save_root_path = os.path.join(path, 'classmap')
    pre_class_rgb_save_path_root = []
    for k, v in sorted(meta['palette_class_mapping'].items()):
        if v == 0:
            pre_class_rgb_save_path_root.append(os.path.join(rgb_save_root_path, 'merge'))
        else:
            pre_class_rgb_save_path_root.append(os.path.join(rgb_save_root_path, str(v)))
    for path_ in pre_class_rgb_save_path_root:
        if not os.path.exists(path_):
            os.makedirs(path_)
    close_rgb_save_root_path = os.path.join(rgb_save_root_path, 'close')
    if not os.path.exists(close_rgb_save_root_path):
        os.makedirs(close_rgb_save_root_path)
    open_rgb_save_root_path = os.path.join(rgb_save_root_path, 'open')
    if not os.path.exists(open_rgb_save_root_path):
        os.makedirs(open_rgb_save_root_path)

    model.eval()
    with torch.no_grad():
        for (im, mask, weight) in test_dataloader:
            im = im.to(device)
            mcc_logit, os_logit = model(im)

            mcc_pred_pro = torch.softmax(mcc_logit, dim=1).squeeze().cpu()
            np.save(os.path.join(path, 'close_probaility.npy'),
                    mcc_pred_pro[:, :meta['image_size'][0], :meta['image_size'][1]].numpy())

            mcc_pred_class = mcc_pred_pro.argmax(dim=0) + torch.tensor(1, dtype=torch.int)
            mcc_pred_class_palette = palette_class_mapping(mcc_pred_class, meta['palette_class_mapping'])
            close_cls_fig = classmap2rgbmap(
                mcc_pred_class_palette[:meta['image_size'][0], :meta['image_size'][1]].numpy(),
                palette=meta['palette'], cls='mcc')
            close_cls_fig.save(os.path.join(close_rgb_save_root_path, str(epoch + 1) + '.png'))

            oc_pred_pro = torch.sigmoid(os_logit).squeeze().cpu()
            np.save(os.path.join(path, 'oc_probaility.npy'),
                    oc_pred_pro[:, :meta['image_size'][0], :meta['image_size'][1]].numpy())

            oc_pred_class = torch.where(oc_pred_pro > 0.5, 1, 0).squeeze().cpu()
            for i in range(oc_pred_class.shape[0]):
                pre_class_map = copy.deepcopy(oc_pred_class[i])
                pre_class_map[oc_pred_class[i] == 1] = meta['palette_class_mapping'][i + 1]
                pre_class_cls_fig = classmap2rgbmap(
                    pre_class_map[:meta['image_size'][0], :meta['image_size'][1]].numpy(),
                    palette=meta['palette'], cls='mcc')
                pre_class_cls_fig.save(os.path.join(pre_class_rgb_save_path_root[i], str(epoch + 1) + '.png'))

            merge_oc_pre_class = torch.where(torch.max(oc_pred_pro, dim=0)[0] > 0.5,
                                             oc_pred_pro.argmax(dim=0) + torch.tensor(1, dtype=torch.int), 0)
            merge_oc_pre_class_palette = palette_class_mapping(merge_oc_pre_class, meta['palette_class_mapping'])
            merge_oc_class_cls_fig = classmap2rgbmap(
                merge_oc_pre_class_palette[:meta['image_size'][0], :meta['image_size'][1]].numpy(),
                palette=meta['palette'], cls='mcc')
            merge_oc_class_cls_fig.save(os.path.join(pre_class_rgb_save_path_root[-1], str(epoch + 1) + '.png'))

            open_pred_class = torch.where(merge_oc_pre_class > 0, mcc_pred_class, test_dataloader.dataset.num_classes)
            open_pred_class_palette = palette_class_mapping(open_pred_class, meta['palette_class_mapping'])
            cls_fig = classmap2rgbmap(
                open_pred_class_palette[:meta['image_size'][0], :meta['image_size'][1]].numpy(),
                palette=meta['palette'], cls='mcc')
            cls_fig.save(os.path.join(open_rgb_save_root_path, str(epoch + 1) + '.png'))

            weight.unsqueeze_(dim=0)

            open_weight = weight.bool()
            open_mask = torch.masked_select(mask.view(-1), open_weight.view(-1)).cpu().numpy() - 1
            open_pred_class = torch.masked_select(open_pred_class.view(-1), open_weight.view(-1)).cpu().numpy() - 1
            open_c_matrix = confusion_matrix(open_pred_class, open_mask, test_dataloader.dataset.num_classes)
            open_acc, open_acc_per_class, open_acc_cls, open_IoU, open_mean_IoU, open_kappa = mcc_evaluate_metric(
                open_c_matrix)

            close_weight = copy.deepcopy(weight)
            close_weight[mask.unsqueeze_(dim=0) == test_dataloader.dataset.num_classes] = 0
            close_weight = close_weight.bool()
            close_mask = torch.masked_select(mask.view(-1), close_weight.view(-1)).cpu().numpy() - 1
            close_pred_class = torch.masked_select(mcc_pred_class.view(-1), close_weight.view(-1)).cpu().numpy() - 1
            close_c_matrix = confusion_matrix(close_pred_class, close_mask, test_dataloader.dataset.num_classes - 1)
            close_acc, close_acc_per_class, close_acc_cls, close_IoU, close_mean_IoU, close_kappa = mcc_evaluate_metric(
                close_c_matrix)

            os_mask = copy.deepcopy(mask)
            os_mask[mask != test_dataloader.dataset.num_classes] = 0
            os_mask[mask == test_dataloader.dataset.num_classes] = 1
            os_label = torch.masked_select(os_mask.view(-1), open_weight.view(-1)).cpu().numpy()
            os_pre_class = copy.deepcopy(merge_oc_pre_class)
            os_pre_class[merge_oc_pre_class == 0] = 1
            os_pre_class[merge_oc_pre_class != 0] = 0
            os_pre_class = torch.masked_select(os_pre_class.view(-1), open_weight.view(-1)).cpu().numpy()
            os_pre, os_rec, os_f1 = pre_rec_f1(os_pre_class, os_label)

            os_pred_pro = copy.deepcopy(torch.max(oc_pred_pro, dim=0)[0])
            os_pred_pro = torch.masked_select(os_pred_pro.view(-1), open_weight.view(-1)).numpy()
            auc, fpr, tpr, threshold = roc_auc(-1 * os_pred_pro, os_label)

    return close_acc, close_acc_cls, close_kappa, open_acc, open_acc_cls, open_kappa, os_pre, os_rec, os_f1, auc, fpr, tpr, threshold


def fusion_fcn_evaluate_fn(model, test_dataloader, meta, cls, path, device, epoch):
    save_rgb_path = os.path.join(path, 'classmap')
    if not os.path.exists(save_rgb_path):
        os.makedirs(save_rgb_path)
    for m in model:
        m.eval()
    with torch.no_grad():
        for (im, positive_test_mask, negative_test_mask) in test_dataloader:
            im = im.to(device)
            positive_test_mask = positive_test_mask.squeeze()
            pred_pro = torch.unsqueeze(torch.zeros_like(negative_test_mask), dim=0).to(device)
            negative_test_mask = negative_test_mask.squeeze()
            for m in model:
                pred_pro += torch.sigmoid(m(im))
            pred_pro = (pred_pro / len(model)).squeeze().cpu()
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


def fcn_inference_fn(model, test_dataloader, meta, cls, path, device, epoch):
    save_rgb_path = os.path.join(path, 'classmap')
    if not os.path.exists(save_rgb_path):
        os.makedirs(save_rgb_path)
    model.eval()
    with torch.no_grad():
        for (im, positive_test_mask, negative_test_mask) in test_dataloader:
            im = im.to(device)

            pred_pro = torch.sigmoid(model(im)).squeeze().cpu()

            pred_class = torch.where(pred_pro > 0.5, 1, 0)

            cls_fig = classmap2rgbmap(
                pred_class[:meta['image_size'][0], :meta['image_size'][1]].numpy(),
                palette=meta['palette'], cls=cls)
            cls_fig.save(os.path.join(save_rgb_path, str(epoch + 1) + '.png'))

            np.save(os.path.join(path, 'probaility.npy'),
                    pred_pro[:meta['image_size'][0], :meta['image_size'][1]].numpy())
