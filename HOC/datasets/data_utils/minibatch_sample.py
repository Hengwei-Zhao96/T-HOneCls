import numpy as np


def minibatch_sample(positive_train_indicator: np.ndarray, unlabeled_train_indicator: np.ndarray, sub_num_iter,
                     seed):
    rs = np.random.RandomState(seed)

    positive_train_indicator_inds = np.where(positive_train_indicator.ravel() == 1)[0]
    positive_minibatch_size = int(len(positive_train_indicator_inds) / sub_num_iter)
    unlabeled_train_indicator_inds = np.where(unlabeled_train_indicator.ravel() == 1)[0]
    unlabeled_minibatch_size = int(len(unlabeled_train_indicator_inds) / sub_num_iter)

    rs.shuffle(positive_train_indicator_inds)
    rs.shuffle(unlabeled_train_indicator_inds)

    positive_train_inds_list = []
    unlabeled_train_inds_list = []
    cnt = 0
    while True:
        positive_train_inds = np.zeros_like(positive_train_indicator).ravel()
        unlabeled_train_inds = np.zeros_like(unlabeled_train_indicator).ravel()

        positive_left = cnt * positive_minibatch_size
        positive_right = min((cnt + 1) * positive_minibatch_size, len(positive_train_indicator_inds))
        if positive_left < positive_right:
            positive_fetch_inds = positive_train_indicator_inds[positive_left:positive_right]
            positive_train_inds[positive_fetch_inds] = 1
            positive_train_inds_list.append(positive_train_inds.reshape(positive_train_indicator.shape))

        unlabeled_left = cnt * unlabeled_minibatch_size
        unlabeled_right = min((cnt + 1) * unlabeled_minibatch_size, len(unlabeled_train_indicator_inds))
        if unlabeled_left < unlabeled_right:
            unlabeled_fetch_inds = unlabeled_train_indicator_inds[unlabeled_left:unlabeled_right]
            unlabeled_train_inds[unlabeled_fetch_inds] = 1
            unlabeled_train_inds_list.append(unlabeled_train_inds.reshape(unlabeled_train_indicator.shape))

        cnt += 1
        if positive_train_inds.sum() == 0 or unlabeled_train_inds.sum() == 0:
            dataset_length = min(len(positive_train_inds_list), len(unlabeled_train_inds_list))
            return positive_train_inds_list[:dataset_length], unlabeled_train_inds_list[:dataset_length]
