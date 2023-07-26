import importlib
import numpy as np


def get_cfg_dataloader(dataset):
    if dataset == 'LongKou':
        config = importlib.import_module('.config_LongKou', package='configs').config
        DataLoader = importlib.import_module('.dataloader', package='data').NewOccLongKouLoader
    elif dataset == 'HongHu':
        config = importlib.import_module('.config_HongHu', package='configs').config
        DataLoader = importlib.import_module('.dataloader', package='data').NewOccHongHuLoader
    elif dataset == 'HanChuan':
        config = importlib.import_module('.config_HanChuan', package='configs').config
        DataLoader = importlib.import_module('.dataloader', package='data').NewOccHanChuanLoader
    elif dataset == 'PineNematodes':
        config = importlib.import_module('.config_PineNematodes', package='configs').config
        DataLoader = importlib.import_module('.dataloader', package='data').NewOccPineLoader
    elif dataset == 'Salinas':
        config = importlib.import_module('.config_Salinas', package='configs').config
        DataLoader = importlib.import_module('.dataloader', package='data').NewOccSalinasLoader
    elif dataset == 'Pavia':
        config = importlib.import_module('.config_Pavia', package='configs').config
        DataLoader = importlib.import_module('.dataloader', package='data').NewOccPaviaLoader
    else:
        raise NotImplemented

    return config, DataLoader


def get_patch_data(img_data, x_indicator: np.ndarray, y_indicator: np.ndarray, patch_size: int):
    length = x_indicator.shape[0]
    r = int(patch_size - 1) / 2
    data = np.zeros((length, img_data.shape[0], patch_size, patch_size))
    for i in range(length):
        data[i, :, :, :] = img_data[:, int(x_indicator[i] - r):int(x_indicator[i] + r + 1),
                           int(y_indicator[i] - r):int(y_indicator[i] + r + 1)]
    return data


def get_patch_dataset(patch_size, dataset, cls_index):
    config, DataLoader = get_cfg_dataloader(dataset=dataset)
    R = int((patch_size - 1) / 2)
    config['data']['train']['params']['cls'] = cls_index
    config['data']['test']['params']['cls'] = cls_index

    train_dataloader = DataLoader(config=config['data']['train']['params'])
    test_dataloader = DataLoader(config=config['data']['test']['params'])

    image_data = train_dataloader.dataset.im
    image_data = np.pad(image_data, ((0, 0), (R, R), (R, R)), mode='constant')
    positive_train_indicator = train_dataloader.dataset.positive_train_indicator
    positive_train_indicator = np.pad(positive_train_indicator, (R, R), mode='constant')
    unlabeled_train_indicator = train_dataloader.dataset.unlabeled_train_indicator
    unlabeled_train_indicator = np.pad(unlabeled_train_indicator, (R, R), mode='constant')
    positive_test_indicator = test_dataloader.dataset.positive_test_indicator
    negative_test_indicator = test_dataloader.dataset.negative_test_indicator

    train_positive_id_x, train_positive_id_y = np.where(positive_train_indicator == 1)
    train_unlabeled_id_x, train_unlabeled_id_y = np.where(unlabeled_train_indicator == 1)

    positive_train_data = get_patch_data(image_data, train_positive_id_x, train_positive_id_y, patch_size)
    unlabeled_train_data = get_patch_data(image_data, train_unlabeled_id_x, train_unlabeled_id_y, patch_size)

    return config, image_data, positive_train_data.squeeze(), unlabeled_train_data.squeeze(), positive_test_indicator, negative_test_indicator
