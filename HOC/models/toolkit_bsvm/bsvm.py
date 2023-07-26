import numpy as np
from sklearn import svm
import logging
from .utils.kfold_index import train_validation_index
from .utils.metrics import min_dist_calculate


class BSVM:

    def __init__(self,
                 # save_path,
                 sigma=(0.05, 0.15, 0.25, 0.35, 0.45, 0.55),
                 cp=(1, 4, 7, 10, 13, 16, 19, 22, 25),
                 cu=(0.1, 0.4, 0.7, 1, 1.3, 1.6, 1.9),
                 threshold=list(np.linspace(0, 1, 51)),
                 n_splits=10,
                 random_seed=2333):
        self.sigma = sigma
        self.cp = cp
        self.cu = cu
        self.threshold = threshold
        self.n_splits = n_splits
        self.random_seed = random_seed

        self.cp_result = 0
        self.cu_result = 0
        self.sigma_result = 0
        self.threshold_result = 0
        self.min_dist = 999999999
        # self.save_path = save_path

        self.result = None
        self.result_p = None

    def fit(self, positive_data, unlabeled_data):
        train_data, train_label, train_indexes, val_indexes = self.__preset_data(positive_data, unlabeled_data)
        for cp in self.cp:
            for cu in self.cu:
                for m in self.sigma:
                    threshold = []
                    min_dist = []
                    print("*")

                    for i in list(range(self.n_splits)):
                        train_data_s = train_data[train_indexes[i], :]
                        train_labels_s = train_label[train_indexes[i]]

                        val_data_s = train_data[val_indexes[i], :]
                        val_label_s = train_label[val_indexes[i]]

                        clf = svm.SVC(C=1.0, kernel='rbf', gamma=m, probability=True, class_weight={1: cp, 0: cu})
                        clf.fit(train_data_s, train_labels_s)
                        val_result = clf.predict_proba(val_data_s)[:, 1]
                        md, t = min_dist_calculate(val_result, val_label_s, threshold_range=self.threshold)
                        min_dist.append(md)
                        threshold.append(t)
                    min_dist_mean = np.asarray(min_dist).mean()
                    threshold_mean = np.asarray(threshold).mean()
                    if min_dist_mean < self.min_dist:
                        self.cp_result = cp
                        self.cu_result = cu
                        self.sigma_result = m
                        self.threshold_result = threshold_mean  # threshold_mean
                        self.min_dist = min_dist_mean
                    logging.info(
                        "Current==>Cp:{},Cu:{},Sigma:{},Thrshold:{},Dist:{}".format(cp,
                                                                                    cu,
                                                                                    m,
                                                                                    threshold_mean,
                                                                                    min_dist_mean))

                    print(
                        "Current==>Cp:{},Cu:{},Sigma:{},Thrshold:{},Dist:{}".format(cp,
                                                                                    cu,
                                                                                    m,
                                                                                    threshold_mean,
                                                                                    min_dist_mean))
                    logging.info(
                        "Result==>Cp:{},Cu:{},Sigma:{},Thrshold:{},Dist:{}".format(self.cp_result,
                                                                                   self.cu_result,
                                                                                   self.sigma_result,
                                                                                   self.threshold_result,
                                                                                   self.min_dist))
                    logging.info("**************************************************************")

        self.clf = svm.SVC(C=1.0, kernel='rbf', gamma=self.sigma_result, probability=True,
                           class_weight={1: self.cp_result, 0: self.cu_result})
        self.clf.fit(train_data, train_label)

    def __preset_data(self, positive_train_data_x, unlabeled_train_data_x):
        train_data = np.concatenate((positive_train_data_x, unlabeled_train_data_x), axis=0)
        train_label = np.concatenate(
            (np.ones(positive_train_data_x.shape[0]), np.zeros(unlabeled_train_data_x.shape[0])), axis=0)
        np.random.seed(self.random_seed)
        id_x = np.random.permutation(train_data.shape[0])
        train_data = train_data[id_x, :]
        train_label = train_label[id_x]

        train_indexes, val_indexes = train_validation_index(train_data, self.n_splits)

        return train_data, train_label, train_indexes, val_indexes

    def predict(self, data):
        self.result_p = self.clf.predict_proba(data)[:, 1]
        self.result = np.where(self.result_p >= self.threshold_result, 1, 0)
        return self.result
