import numpy as np
import torch
from torch.utils import data

from HOC.datasets.data_utils.data_preprocess import divisible_pad
from HOC.datasets.data_utils.minibatch_sample import minibatch_sample, mcc_minibatch_sample, os_mcc_minibatch_sample

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


class MccFullImageDataset(data.dataset.Dataset):
    def __init__(self, image, mask, train_label, test_label, num_classes, train_flage, np_seed,
                 num_train_samples_per_class, sub_minibatch):
        self.image = image
        self.mask = mask
        self.num_classes = num_classes
        self.training = train_flage
        self.num_train_samples_per_class = num_train_samples_per_class
        self.sub_minibatch = sub_minibatch
        self.train_label = train_label
        self.test_label = test_label
        self._seed = np_seed
        self._rs = np.random.RandomState(np_seed)
        # set list lenght = 9999 to make sure seeds enough
        self.seeds_for_minibatchsample = [e for e in self._rs.randint(low=2 << 31 - 1, size=9999)]
        self.preset()

    def preset(self):
        train_indicator, test_indicator = self.train_label, self.test_label

        blob = divisible_pad([np.concatenate([self.image.transpose(2, 0, 1),
                                              self.mask[None, :, :],
                                              train_indicator[None, :, :],
                                              test_indicator[None, :, :]], axis=0)], 16)

        im = blob[0, :self.image.shape[-1], :, :]
        mask = blob[0, -3, :, :]
        self.train_indicator = blob[0, -2, :, :]
        self.test_indicator = blob[0, -1, :, :]

        if self.training:
            self.train_inds_list = mcc_minibatch_sample(mask, self.train_indicator, self.sub_minibatch,
                                                        seed=self.seeds_for_minibatchsample.pop())

        self.pad_im = im
        self.pad_mask = mask

    def resample_minibatch(self):
        self.train_inds_list = mcc_minibatch_sample(self.pad_mask, self.train_indicator, self.sub_minibatch,
                                                    seed=self.seeds_for_minibatchsample.pop())

    def __getitem__(self, idx):
        if self.training:
            return self.pad_im, self.pad_mask, self.train_inds_list[idx]
        else:
            return self.pad_im, self.pad_mask, self.test_indicator

    def __len__(self):
        if self.training:
            return len(self.train_inds_list)
        else:
            return 1


class OsMccFullImageDataset(data.dataset.Dataset):
    def __init__(self, image, gt, mask, num_classes, train_flage, np_seed,
                 num_train_samples_per_class, sub_minibatch, num_unlabeled_samples):
        self.image = image
        self.gt = gt
        self.mask = mask
        self.num_classes = num_classes
        self.training = train_flage
        self.num_train_samples_per_class = num_train_samples_per_class
        self.sub_minibatch = sub_minibatch
        self.num_unlabeled_samples = num_unlabeled_samples
        self._seed = np_seed
        self._rs = np.random.RandomState(np_seed)
        # set list lenght = 9999 to make sure seeds enough
        self.seeds_for_minibatchsample = [e for e in self._rs.randint(low=2 << 31 - 1, size=9999)]
        self.preset()

    def get_positive_train_indicator(self, cls):
        gt_flatten = self.gt.ravel()
        mask_flatten = self.mask.ravel()
        positive_train_indicator = np.zeros_like(gt_flatten)
        positive_train_indicator[np.where((gt_flatten == cls) & (mask_flatten == 1))[0]] = 1
        positive_train_indicator = positive_train_indicator.reshape(self.gt.shape)
        return positive_train_indicator

    def get_unlabeled_train_indicator(self, seed):
        rs = np.random.RandomState(seed)
        gt_mask_flatten = self.mask.ravel()
        unlabeled_train_indicator = np.zeros_like(gt_mask_flatten)
        unlabeled_train_indicator[:min(self.num_unlabeled_samples, gt_mask_flatten.shape[0])] = 1
        rs.shuffle(unlabeled_train_indicator)
        unlabeled_train_indicator = unlabeled_train_indicator.reshape(self.mask.shape)
        return unlabeled_train_indicator

    # def get_unlabeled_train_indicator(self, seed):
    #     unlabeled_train_indicator = np.load('/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-HongHu/disjoint_unlabeled.npy')
    #     return unlabeled_train_indicator

    def preset(self):
        positive_indicator = []
        unlabeled_indicator = []
        se = self.seeds_for_minibatchsample.pop()
        for i in range(1, self.num_classes + 1):
            positive_indicator.append(self.get_positive_train_indicator(i))
            unlabeled_indicator.append(self.get_unlabeled_train_indicator(se))
            # unlabeled_indicator.append(self.get_unlabeled_train_indicator(self.seeds_for_minibatchsample.pop()))
        positive_indicator = np.asarray(positive_indicator)
        unlabeled_indicator = np.asarray(unlabeled_indicator)

        blob = divisible_pad([np.concatenate([self.image.transpose(2, 0, 1),
                                              self.gt[None, :, :],
                                              self.mask[None, :, :],
                                              positive_indicator,
                                              unlabeled_indicator], axis=0)], 16)

        channels = self.image.shape[-1]

        self.pad_im = blob[0, :channels, :, :]
        self.pad_gt = blob[0, channels:(channels + 1), :, :].squeeze()
        self.pad_mask = blob[0, (channels + 1):(channels + 2), :, :].squeeze()
        self.pad_positive_indicator = blob[0, (channels + 2):(channels + self.num_classes + 2), :, :]
        self.pad_unlabeled_indicator = blob[0, (channels + self.num_classes + 2):, :, :]

        if self.training:
            self.train_inds_list, self.positive_inds_list, self.unlabeled_inds_list = os_mcc_minibatch_sample(
                self.pad_gt, self.pad_mask, self.sub_minibatch, self.pad_positive_indicator,
                self.pad_unlabeled_indicator, num_classes=self.num_classes, seed=self.seeds_for_minibatchsample.pop())

    def resample_minibatch(self):
        if self.training:
            self.train_inds_list, self.positive_inds_list, self.unlabeled_inds_list = os_mcc_minibatch_sample(
                self.pad_gt, self.pad_mask, self.sub_minibatch, self.pad_positive_indicator,
                self.pad_unlabeled_indicator, num_classes=self.num_classes, seed=self.seeds_for_minibatchsample.pop())

    def __getitem__(self, idx):
        if self.training:
            return self.pad_im, self.pad_gt, self.train_inds_list[idx], self.positive_inds_list[idx], \
                   self.unlabeled_inds_list[idx]
        else:
            return self.pad_im, self.pad_gt, self.pad_mask

    def __len__(self):
        if self.training:
            return len(self.train_inds_list)
        else:
            return 1
