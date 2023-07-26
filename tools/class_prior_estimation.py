import argparse
import logging
import os
import sys

sys.path.append(os.path.abspath('.'))

import numpy as np
from HOC.class_prior_estimation import KMPE
from HOC.apis.build_trainer import build_from_cfg
from HOC.utils import read_config, basic_logging, DATASETS


def _get_patch_data(img_data, x_indicator: np.ndarray, y_indicator: np.ndarray, patch_size: int):
    length = x_indicator.shape[0]
    r = int(patch_size - 1) / 2
    data = np.zeros((length, img_data.shape[0], patch_size, patch_size))
    for i in range(length):
        data[i, :, :, :] = img_data[:, int(x_indicator[i] - r):int(x_indicator[i] + r + 1),
                           int(y_indicator[i] - r):int(y_indicator[i] + r + 1)]
    return data.squeeze()


def Argparse():
    parser = argparse.ArgumentParser(
        description='Class Prior Estimation')
    parser.add_argument('-c', '--cfg', type=str,
                        default='/home/zhw2021/code/HOneCls/configs/KMPE/IndianPines/11.py',
                        help='File path of config')

    return parser.parse_args()


if __name__ == "__main__":
    args = Argparse()
    config = read_config(args.cfg)

    folder_name = os.path.join(os.path.abspath('.'),
                               'Log',
                               'class_prior_estimation',
                               config['dataset']['train']['type'],
                               str(config['dataset']['train']['params']['ccls']))

    save_path = basic_logging(folder_name)
    print("The save path is:", save_path)

    train_pf_dataset = build_from_cfg(config['dataset']['train'], DATASETS)

    image = train_pf_dataset.im
    positive_train_indicator = train_pf_dataset.positive_train_indicator
    unlabeled_train_indicator = train_pf_dataset.unlabeled_train_indicator

    train_positive_id_x, train_positive_id_y = np.where(positive_train_indicator == 1)
    train_unlabeled_id_x, train_unlabeled_id_y = np.where(unlabeled_train_indicator == 1)

    positive_train_data = _get_patch_data(image, train_positive_id_x, train_positive_id_y, patch_size=1)
    unlabeled_train_data = _get_patch_data(image, train_unlabeled_id_x, train_unlabeled_id_y, patch_size=1)

    X_component = positive_train_data.squeeze()
    X_mixture = unlabeled_train_data.squeeze()

    logging.info("The shape of Positive data is:{}".format(X_component.shape))
    logging.info("The shape of Unlabeled data is:{}".format(X_mixture.shape))

    (KM1, KM2) = KMPE(X_mixture, X_component)

    logging.info("KM1_estimate={}".format(KM1))
    logging.info("KM2_estimate={}".format(KM2))
