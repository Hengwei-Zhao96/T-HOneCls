#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/29 18:03
# @Author : Hw-Zhao
# @Site : 
# @File : dataset_Pavia.py
# @Software: PyCharm

from scipy.io import loadmat
from HOC.datasets.data_base import OccFullImageDataset
import HOC.datasets.data_utils.data_preprocess as preprocess
from HOC.datasets.data_utils import get_train_test_label

SEED = 2333


class NewOccPaviaDataset(OccFullImageDataset):
    def __init__(self, config):
        self.im_mat_path = config['image_mat_path']
        self.gt_mat_path = config['gt_mat_path']

        im_mat = loadmat(self.im_mat_path)
        image = im_mat['paviaU']
        gt_mat = loadmat(self.gt_mat_path)
        mask = gt_mat['paviaU_gt']

        self.train_flag = config['train_flage']
        self.cls = config['cls']
        self.num_train_samples_per_class = config['num_positive_train_samples']

        im_cmean = image.reshape((-1, image.shape[-1])).mean(axis=0)
        im_cstd = image.reshape((-1, image.shape[-1])).std(axis=0)
        image = preprocess.mean_std_normalize(image, im_cmean, im_cstd)

        train_label, test_label = get_train_test_label(cls=self.cls, raw_label=mask,
                                                       num_positive_train_samples=self.num_train_samples_per_class,
                                                       seed=SEED)

        if self.train_flag:
            mask = train_label
        else:
            mask = test_label

        super(NewOccPaviaDataset, self).__init__(image=image,
                                                 mask=mask,
                                                 cls=self.cls,
                                                 ratio=config['ratio'],
                                                 train_flag=self.train_flag,
                                                 np_seed=SEED,
                                                 num_train_samples_per_class=self.num_train_samples_per_class,
                                                 sub_num_iter=config['sub_minibatch'])
