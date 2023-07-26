import numpy as np

from HOC.models.toolkit_bsvm import BSVM as bsvm
from HOC.utils import MODELS


@MODELS.register_module()
class BSVM:
    def __init__(self,
                 sigma=(0.05, 0.15, 0.25, 0.35, 0.45, 0.55),
                 cp=(1, 4, 7, 10, 13, 16, 19, 22, 25),
                 cu=(0.1, 0.4, 0.7, 1, 1.3, 1.6, 1.9),
                 threshold=list(np.linspace(0, 1, 51)),
                 n_splits=10,
                 random_seed=2333):
        self.classifier = bsvm(sigma=sigma, cp=cp, cu=cu, threshold=threshold, n_splits=n_splits,
                               random_seed=random_seed)

    def fit(self, positive_data, unlabeled_data):
        self.classifier.fit(positive_data, unlabeled_data)

    def predict(self, data):
        return self.classifier.predict(data)

    def predict_pro(self, data):
        self.classifier.predict(data)
        return self.classifier.result_p
