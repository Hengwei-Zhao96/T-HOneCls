import numpy as np
from sklearn.svm import OneClassSVM

from HOC.utils import MODELS


@MODELS.register_module()
class OCSVM:
    def __init__(self, kernel):
        self.classifier = OneClassSVM(kernel=kernel)

    def fit(self, positive_data, unlabeled_data):
        self.classifier.fit(positive_data)

    def predict(self, data):
        return np.where(self.classifier.predict(data) > 0, 1, 0)

    def predict_pro(self, data):
        return self.classifier.decision_function(data)
