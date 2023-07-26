import numpy as np
from sklearn.svm import SVC
import torch.nn as nn


class PUCalibration:
    def __init__(self, estimator, hold_out_ratio=0.2, mode='pul', random_seed=2333):
        self.estimator = estimator
        self.hold_out_ratio = hold_out_ratio
        self.mode = mode
        self.random_seed = random_seed

        self.c = None
        self.result = None
        self.result_p = None

    def fit(self, positive_data, unlabeled_data):
        np.random.seed(self.random_seed)
        positive_size = positive_data.shape[0]
        positive_index = np.random.permutation(positive_size)
        hold_out_size = int(np.ceil(positive_size * self.hold_out_ratio))

        hold_out_index = positive_index[:hold_out_size]
        training_index = positive_index[hold_out_size:]

        hold_out_positive_data = positive_data[hold_out_index]
        training_positive_data = positive_data[training_index]

        train_data = np.concatenate((training_positive_data, unlabeled_data), axis=0)
        train_label = np.concatenate(
            (np.ones(training_positive_data.shape[0]), np.zeros(unlabeled_data.shape[0])), axis=0)
        id_x = np.random.permutation(train_data.shape[0])
        train_data = train_data[id_x]
        train_label = train_label[id_x]

        if isinstance(self.estimator, SVC):
            self.estimator.fit(train_data, train_label)
            hold_out_pro = self.estimator.predict_proba(hold_out_positive_data)[:, 1]
        elif isinstance(self.estimator, nn):
            NotImplemented

        self.c = np.mean(hold_out_pro)

    def predict(self, data):

        if isinstance(self.estimator, SVC):
            pro = self.estimator.predict_proba(data)[:, 1]
        elif isinstance(self.estimator, nn):
            NotImplemented

        if self.mode == 'pul':
            self.result_p = pro / self.c
        elif self.mode == 'pbl':
            self.result_p = (1 - self.c) / self.c * pro / (1 - pro)

        self.result = np.where(self.result_p >= 0.5, 1, 0)
        return self.result
