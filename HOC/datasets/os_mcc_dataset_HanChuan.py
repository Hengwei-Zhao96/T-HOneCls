import numpy as np

from HOC.datasets.data_base import OsMccFullImageDataset
from HOC.datasets.data_utils import mean_std_normalize
from HOC.datasets.data_utils.image_io import read_ENVI
from HOC.utils.registry import DATASETS

SEED = 2333


@DATASETS.register_module()
class OsMccHanChuanDataset(OsMccFullImageDataset):
    def __init__(self, image_path, gt_path, train_flage, num_classes, sub_minibatch,
                 num_unlabeled_samples, num_train_samples_per_class=100):
        self.im_path = image_path
        self.gt_path = gt_path

        image = read_ENVI(self.im_path)
        gt = np.load(self.gt_path)

        im_cmean = image.reshape((-1, image.shape[-1])).mean(axis=0)
        im_cstd = image.reshape((-1, image.shape[-1])).std(axis=0)
        image = mean_std_normalize(image, im_cmean, im_cstd)

        mask = np.where(gt != 0, 1, 0)
        # The class index of training gt must start from 1, and unknown class is annotated as 0.
        # The class index of testing gt must start from 1, and unknown class is annotated as the last index.
        # The class index 0 is not be trained and tested.

        super(OsMccHanChuanDataset, self).__init__(image=image,
                                                   gt=gt,
                                                   mask=mask,
                                                   np_seed=SEED,
                                                   num_classes=num_classes,
                                                   train_flage=train_flage,
                                                   num_train_samples_per_class=num_train_samples_per_class,
                                                   sub_minibatch=sub_minibatch,
                                                   num_unlabeled_samples=num_unlabeled_samples)
