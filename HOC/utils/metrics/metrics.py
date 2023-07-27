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
