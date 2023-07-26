import numpy as np
from scipy.io import loadmat
from HOC.datasets.data_base import OsMccFullImageDataset
from HOC.datasets.data_utils import mean_std_normalize, os_mcc_get_train_test_label
from HOC.utils.registry import DATASETS

SEED = 2333


@DATASETS.register_module()
class OsMccPaviaUDataset(OsMccFullImageDataset):
    def __init__(self, image_path, gt_path, train_flage, num_classes, num_train_samples_per_class=200,
                 sub_minibatch=10, num_unlabeled_samples=4000):
        self.im_mat_path = image_path
        self.gt_mat_path = gt_path

        im_mat = loadmat(self.im_mat_path)
        image = im_mat['paviaU']
        # gt_mat = loadmat(self.gt_mat_path)
        # gt = gt_mat['paviaU_gt']
        gt = np.load(self.gt_mat_path)
        if train_flage:
            gt = np.where(gt > num_classes, 0, gt)

        im_cmean = image.reshape((-1, image.shape[-1])).mean(axis=0)
        im_cstd = image.reshape((-1, image.shape[-1])).std(axis=0)
        image = mean_std_normalize(image, im_cmean, im_cstd)

        mask = os_mcc_get_train_test_label(gt_mask=gt,
                                           num_train_samples=num_train_samples_per_class,
                                           num_classes=num_classes,
                                           train_flage=train_flage,
                                           seed=SEED)

        super(OsMccPaviaUDataset, self).__init__(image=image,
                                                 gt=gt,
                                                 mask=mask,
                                                 np_seed=SEED,
                                                 num_classes=num_classes,
                                                 train_flage=train_flage,
                                                 num_train_samples_per_class=num_train_samples_per_class,
                                                 sub_minibatch=sub_minibatch,
                                                 num_unlabeled_samples=num_unlabeled_samples)
