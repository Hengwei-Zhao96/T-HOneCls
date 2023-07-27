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
