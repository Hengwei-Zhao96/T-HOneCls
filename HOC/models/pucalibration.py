from sklearn import svm

from HOC.models.toolkit_pucalibration import PUCalibration as pulearner
from HOC.utils import MODELS


@MODELS.register_module()
class PUL:
    def __init__(self):
        self.estimator = svm.SVC(kernel='rbf', probability=True)
        self.classifier = pulearner(estimator=self.estimator, mode='pul')

    def fit(self, positive_data, unlabeled_data):
        self.classifier.fit(positive_data, unlabeled_data)

    def predict(self, data):
        return self.classifier.predict(data)

    def predict_pro(self, data):
        self.classifier.predict(data)
        return self.classifier.result_p


@MODELS.register_module()
class PBL:
    def __init__(self):
        self.estimator = svm.SVC(kernel='rbf', probability=True)
        self.classifier = pulearner(estimator=self.estimator, mode='pbl')

    def fit(self, positive_data, unlabeled_data):
        self.classifier.fit(positive_data, unlabeled_data)

    def predict(self, data):
        return self.classifier.predict(data)

    def predict_pro(self, data):
        self.classifier.predict(data)
        return self.classifier.result_p
