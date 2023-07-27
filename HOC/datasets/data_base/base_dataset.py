import numpy as np
import torch
from torch.utils import data

from HOC.datasets.data_utils.data_preprocess import divisible_pad
from HOC.datasets.data_utils.minibatch_sample import minibatch_sample

SEED = 2333


class OccFullImageDataset(data.dataset.Dataset):
    def __init__(self, image: np.ndarray, mask: np.ndarray, cls, ratio, train_flag, np_seed,
                 num_train_samples_per_class, sub_num_iter):
        """

        :param image: the shape of image is HxWxC
        :param mask:
        :param cls:
        :param ratio:
        :param train_flag:
        :param np_seed:
        :param num_train_samples_per_class:
        :param sub_num_iter:
        """
        self.image = image
        self.mask = mask
        self.cls = cls
        self.ratio = ratio
        self.train_flag = train_flag
        self.num_train_samples_per_class = num_train_samples_per_class
        self.sub_num_iter = sub_num_iter
        self._seed = np_seed
        self._rs = np.random.RandomState(np_seed)
        # set list length = 9999 to make sure seeds enough
        self.seeds_for_minibatchsample = [e for e in self._rs.randint(low=2147483648, size=9999)]
        self.preset()

    def preset(self):

        if self.train_flag:
            positive_train_indicator = self.get_positive_train_indicator()
            unlabeled_train_indicator = self.get_unlabeled_train_indicator()
            blob = divisible_pad([np.concatenate([self.image.transpose(2, 0, 1),
                                                  self.mask[None, :, :],
                                                  positive_train_indicator[None, :, :],
                                                  unlabeled_train_indicator[None, :, :]], axis=0)], 16)

            self.im = blob[0, :self.image.shape[-1], :, :]
            self.positive_train_indicator = blob[0, -2, :, :]
            self.unlabeled_train_indicator = blob[0, -1, :, :]

            self.positive_inds_list, self.unlabeled_inds_list = minibatch_sample(self.positive_train_indicator,
                                                                                 self.unlabeled_train_indicator,
                                                                                 sub_num_iter=self.sub_num_iter,
                                                                                 seed=self.seeds_for_minibatchsample.pop())
        else:
            positive_test_indicator, negative_test_indicator = self.get_test_indicator()
            blob = divisible_pad([np.concatenate([self.image.transpose(2, 0, 1),
                                                  self.mask[None, :, :],
                                                  positive_test_indicator[None, :, :],
                                                  negative_test_indicator[None, :, :]], axis=0)], 16)

            self.im = blob[0, :self.image.shape[-1], :, :]
            self.positive_test_indicator = blob[0, -2, :, :]
            self.negative_test_indicator = blob[0, -1, :, :]

    def resample_minibatch(self):
        if self.train_flag:
            self.positive_inds_list, self.unlabeled_inds_list = minibatch_sample(self.positive_train_indicator,
                                                                                 self.unlabeled_train_indicator,
                                                                                 sub_num_iter=self.sub_num_iter,
                                                                                 seed=self.seeds_for_minibatchsample.pop())

    def get_positive_train_indicator(self):
        gt_mask_flatten = self.mask.ravel()
        positive_train_indicator = np.zeros_like(gt_mask_flatten)
        positive_train_indicator[np.where(gt_mask_flatten == self.cls)[0]] = 1
        positive_train_indicator = positive_train_indicator.reshape(self.mask.shape)
        return positive_train_indicator

    def get_unlabeled_train_indicator(self):
        rs = np.random.RandomState(self._seed)
        gt_mask_flatten = self.mask.ravel()
        unlabeled_train_indicator = np.zeros_like(gt_mask_flatten)
        # unlabeled_train_indicator[:4000] = 1
        unlabeled_train_indicator[
        :min(int(self.num_train_samples_per_class * self.ratio), gt_mask_flatten.shape[0])] = 1
        rs.shuffle(unlabeled_train_indicator)
        unlabeled_train_indicator = unlabeled_train_indicator.reshape(self.mask.shape)
        return unlabeled_train_indicator

    def get_test_indicator(self):
        gt_mask_flatten = self.mask.ravel()

        positive_test_indicator = np.zeros_like(gt_mask_flatten)
        negative_test_indicator = np.zeros_like(gt_mask_flatten)

        positive_test_indicator[np.where(gt_mask_flatten == self.cls)[0]] = 1

        negative_inds_negative = np.where(gt_mask_flatten != self.cls)[0]
        negative_inds_background = np.where(gt_mask_flatten == 0)[0]

        negative_test_indicator[negative_inds_negative] = 1
        negative_test_indicator[negative_inds_background] = 0

        positive_test_indicator = positive_test_indicator.reshape(self.mask.shape)
        negative_test_indicator = negative_test_indicator.reshape(self.mask.shape)

        return positive_test_indicator, negative_test_indicator

    def __getitem__(self, idx):
        if self.train_flag:
            return self.im, self.positive_inds_list[idx], self.unlabeled_inds_list[idx]
        else:
            return self.im, self.positive_test_indicator, self.negative_test_indicator

    def __len__(self):
        if self.train_flag:
            return len(self.positive_inds_list)
        else:
            return 1
