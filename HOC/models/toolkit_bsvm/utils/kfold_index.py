from sklearn.model_selection import KFold
import numpy as np

def train_validation_index(data, n_splits):
    train_indexes = []
    val_indexes = []
    kf = KFold(n_splits=n_splits)
    for train, val in kf.split(data):
        train_indexes.append(train)
        val_indexes.append(val)
    return np.asarray(train_indexes), np.asarray(val_indexes)