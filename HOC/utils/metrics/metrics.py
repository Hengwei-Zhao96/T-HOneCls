import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score


def roc_auc(pred: np.asarray, target: np.asarray):
    auc = roc_auc_score(y_true=target, y_score=pred)
    fpr, tpr, threshold = roc_curve(y_true=target, y_score=pred)
    return auc, fpr, tpr, threshold


def pre_rec_f1(pred: np.asarray, target: np.asarray):
    pre = precision_score(target, pred, zero_division=0)
    rec = recall_score(target, pred, zero_division=0)
    f1 = f1_score(target, pred, zero_division=0)
    return pre, rec, f1


def all_metric(pred_pro: np.asarray, pred_class: np.asarray, target: np.asarray):
    auc, fpr, tpr, threshold = roc_auc(pred_pro, target)
    pre, rec, f1 = pre_rec_f1(pred_class, target)
    return auc, fpr, tpr, threshold, pre, rec, f1


def confusion_matrix(pred, label, num_classes):
    mask = (label >= 0) & (label < num_classes)
    conf_mat = np.bincount(num_classes * label[mask].astype(int) + pred[mask], minlength=num_classes ** 2).reshape(
        num_classes, num_classes)
    return conf_mat


def mcc_evaluate_metric(conf_mat):
    acc = np.diag(conf_mat).sum() / conf_mat.sum()
    acc_per_class = np.diag(conf_mat) / conf_mat.sum(axis=1)
    acc_cls = np.nanmean(acc_per_class)

    IoU = np.diag(conf_mat) / (conf_mat.sum(axis=1) + conf_mat.sum(axis=0) - np.diag(conf_mat))
    mean_IoU = np.nanmean(IoU)

    # kappa
    pe = np.dot(np.sum(conf_mat, axis=0), np.sum(conf_mat, axis=1)) / (conf_mat.sum() ** 2)
    kappa = (acc - pe) / (1 - pe)
    return acc, acc_per_class, acc_cls, IoU, mean_IoU, kappa
