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


def mcc_minibatch_sample(gt_mask, train_indicator, minibatch_size, seed):
    """

        Args:
            gt_mask: 2-D array of shape [height, width]
            train_indicator: 2-D array of shape [height, width]
            minibatch_size:

        Returns:

    """
    rs = np.random.RandomState(seed)
    # split into N classes
    cls_list = np.unique(gt_mask)
    inds_dict_per_class = dict()
    for cls in cls_list:
        train_inds_per_class = np.where(gt_mask == cls, train_indicator, np.zeros_like(train_indicator))
        inds = np.where(train_inds_per_class.ravel() == 1)[0]
        rs.shuffle(inds)

        inds_dict_per_class[cls] = inds

    train_inds_list = []
    cnt = 0
    while True:
        train_inds = np.zeros_like(train_indicator).ravel()
        for cls, inds in inds_dict_per_class.items():
            left = cnt * minibatch_size
            if left >= len(inds):
                continue
            # remain last batch though the real size is smaller than minibatch_size
            right = min((cnt + 1) * minibatch_size, len(inds))
            fetch_inds = inds[left:right]
            train_inds[fetch_inds] = 1

        cnt += 1
        if train_inds.sum() == 0:
            return train_inds_list
        train_inds_list.append(train_inds.reshape(train_indicator.shape))


def os_mcc_minibatch_sample(gt, mask, sub_minibatch_size, positive_train_indicator, unlabeled_train_indicator,
                            num_classes, seed):
    train_inds_list = mcc_minibatch_sample(gt, mask, sub_minibatch_size, seed=seed)

    num_iter = len(train_inds_list)

    c_p_inds_list = []
    c_u_inds_list = []
    for i in range(num_classes):
        p_list, u_list = minibatch_sample(
            positive_train_indicator=positive_train_indicator[i],
            unlabeled_train_indicator=unlabeled_train_indicator[i],
            sub_num_iter=num_iter,
            seed=seed)
        c_p_inds_list.append(p_list)
        c_u_inds_list.append(u_list)

    positive_inds_list = []
    unlabeled_inds_list = []
    for i in range(num_iter):
        positive_inds_list.append([])
        unlabeled_inds_list.append([])
        for j in range(num_classes):
            positive_inds_list[i].append(c_p_inds_list[j][i])
            unlabeled_inds_list[i].append(c_u_inds_list[j][i])

    for i in range(num_iter):
        positive_inds_list[i] = np.asarray(positive_inds_list[i])
        unlabeled_inds_list[i] = np.asarray(unlabeled_inds_list[i])

    return train_inds_list, positive_inds_list, unlabeled_inds_list
