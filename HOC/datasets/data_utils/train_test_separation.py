import numpy as np


def get_train_test_label(cls, raw_label: np.ndarray, num_positive_train_samples, seed):
    rs = np.random.RandomState(seed)

    gt_mask_flatten = raw_label.ravel()

    train_label = raw_label.copy()
    test_label = raw_label.copy()

    positive_inds = np.where(gt_mask_flatten == cls)[0]
    rs.shuffle(positive_inds)

    positive_train_mask = np.zeros_like(gt_mask_flatten)
    positive_test_mask = np.ones_like(gt_mask_flatten)

    positive_train_mask[positive_inds[:num_positive_train_samples]] = 1
    positive_test_mask[positive_inds[:num_positive_train_samples]] = 0

    positive_train_mask = positive_train_mask.reshape(raw_label.shape)
    positive_test_mask = positive_test_mask.reshape(raw_label.shape)

    train_label = train_label * positive_train_mask
    test_label = test_label * positive_test_mask

    return train_label, test_label


def mcc_get_train_test_label(gt_mask, num_train_samples, num_classes, seed=2333, train_test_disjoint=False):
    """

        Args:
            gt_mask: 2-D array of shape [height, width]
            num_train_samples: int
            num_classes: scalar
            seed: int

        Returns:
            train_indicator, test_indicator
    """
    rs = np.random.RandomState(seed)

    gt_mask_flatten = gt_mask.ravel()
    train_indicator = np.zeros_like(gt_mask_flatten)
    test_indicator = np.zeros_like(gt_mask_flatten)
    if not train_test_disjoint:
        for i in range(1, num_classes + 1):
            inds = np.where(gt_mask_flatten == i)[0]
            rs.shuffle(inds)

            train_inds = inds[:num_train_samples]
            test_inds = inds[num_train_samples:]

            train_indicator[train_inds] = 1
            test_indicator[test_inds] = 1
    else:
        for i in range(1, num_classes + 1):
            inds = np.where(gt_mask_flatten == i)[0]
            rs.shuffle(inds)

            train_indicator[inds] = 1
            test_indicator[inds] = 1

    train_indicator = train_indicator.reshape(gt_mask.shape)
    test_indicator = test_indicator.reshape(gt_mask.shape)

    return train_indicator, test_indicator


def os_mcc_get_train_test_label(gt_mask, num_train_samples, num_classes, train_flage, seed=2333):
    rs = np.random.RandomState(seed)

    gt_mask_flatten = gt_mask.ravel()
    indicator = np.zeros_like(gt_mask_flatten)
    if train_flage:
        for i in range(1, num_classes + 1):
            c_inds = np.where(gt_mask_flatten == i)[0]
            rs.shuffle(c_inds)

            inds = c_inds[:num_train_samples]
            indicator[inds] = 1
    else:
        for i in range(1, num_classes + 1):
            c_inds = np.where(gt_mask_flatten == i)[0]
            rs.shuffle(c_inds)

            if i == num_classes:
                inds = c_inds
            else:
                inds = c_inds[num_train_samples:]
            indicator[inds] = 1

    indicator = indicator.reshape(gt_mask.shape)

    return indicator
