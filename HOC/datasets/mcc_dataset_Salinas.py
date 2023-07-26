import numpy as np
from scipy.io import loadmat
from HOC.datasets.data_base import MccFullImageDataset
from HOC.datasets.data_utils import mean_std_normalize, mcc_get_train_test_label
from HOC.utils.registry import DATASETS

SEED = 2333


@DATASETS.register_module()
class MccSalinasDataset(MccFullImageDataset):
    def __init__(self, image_path, gt_path, train_flage, num_classes, num_train_samples_per_class=200,
                 sub_minibatch=10):
        self.im_mat_path = image_path
        self.gt_mat_path = gt_path

        im_mat = loadmat(self.im_mat_path)
        image = im_mat['salinas_corrected']
        # gt_mat = loadmat(self.gt_mat_path)
        # mask = gt_mat['salinas_gt']
        mask = np.load(self.gt_mat_path)

        self.train_flag = train_flage
        self.num_train_samples_per_class = num_train_samples_per_class
        self.sub_minibatch = sub_minibatch
        self.num_classes = num_classes

        im_cmean = image.reshape((-1, image.shape[-1])).mean(axis=0)
        im_cstd = image.reshape((-1, image.shape[-1])).std(axis=0)
        image = mean_std_normalize(image, im_cmean, im_cstd)

        train_label, test_label = mcc_get_train_test_label(gt_mask=mask,
                                                           num_train_samples=self.num_train_samples_per_class,
                                                           num_classes=self.num_classes,
                                                           seed=SEED)

        super(MccSalinasDataset, self).__init__(image=image, mask=mask, train_label=train_label,
                                                test_label=test_label, np_seed=SEED, num_classes=self.num_classes,
                                                train_flage=self.train_flag,
                                                num_train_samples_per_class=self.num_train_samples_per_class,
                                                sub_minibatch=self.sub_minibatch)