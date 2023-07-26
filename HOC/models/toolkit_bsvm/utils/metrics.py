import numpy as np
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score


def min_dist_calculate(pred, real, threshold_range):
    TP = []
    FU = []
    P_sum = real.sum()
    P_U_sum = real.shape[0]
    for i in threshold_range:
        _pred = np.where(pred > i, 1, 0)
        tp = (_pred * real).sum()
        TP.append(tp)
        fu = np.where(_pred - real == 1, 1, 0).sum()
        FU.append(fu)
    TPR = np.asarray(TP) / P_sum
    PPP = np.asarray(FU) / P_U_sum
    dist = np.sqrt((1 - TPR) ** 2 + PPP ** 2)
    min_dist = np.min(dist)
    id_ = np.argmin(dist)
    threshold = threshold_range[id_]
    return min_dist, threshold


def acc_pre_rec_f1(pred, real, crown_test=None):
    if crown_test is None:
        pred = pred.reshape(real.shape[0])
        pred = np.where(pred >= 0.5, 1, 0)
    acc = accuracy_score(real, pred)
    pre = precision_score(real, pred)
    recall = recall_score(real, pred)
    f1 = f1_score(real, pred)
    return acc, pre, recall, f1