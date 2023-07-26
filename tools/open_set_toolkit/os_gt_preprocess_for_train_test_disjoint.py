import copy
from PIL import Image
import numpy as np
from osgeo import gdal


def read_ENVI(filepath):
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, cols, rows)
    if len(data.shape) == 3:
        data = data.transpose((1, 2, 0))
    return data


if __name__ == "__main__":
    train_label = read_ENVI("/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-HongHu/Train100")
    test_label = read_ENVI("/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-HongHu/Test100")
    new_train_label_save_path = "/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-HongHu/Train100_new_label"
    new_test_label_save_path = "/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-HongHu/Test100_new_open_label"
    new_train_img_save_path = "/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-HongHu/new_train100_label.png"
    new_test_img_save_path = "/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-HongHu/new_test100_label.png"
    unlabeled_class = [1, 2, 3, 5, 22]

    unlabeled_class.sort()  # unlabeled data must sort from little to big.
    print("Raw train label")
    print(np.unique(train_label))
    print("Raw test label")
    print(np.unique(test_label))
    print("**************************************************")
    print("The number of background class of training label:%f" % len(np.where(train_label == 0)[0]))
    print("The number of background class of testing label:%f" % len(np.where(test_label == 0)[0]))
    print("**************************************************")

    total_class = len(np.unique(train_label)) - 1
    new_total_class = total_class - len(unlabeled_class) + 1

    new_train_label = copy.deepcopy(train_label)
    new_test_label = copy.deepcopy(test_label).astype(np.float64)
    for c in unlabeled_class:
        new_train_label[train_label == c] = 0
        print("The label of new train label-mask unknown class")
        print(np.unique(new_train_label))
        new_test_label[new_test_label == c] = -1
        print("The label of new test label-mask unknown class")
        print(np.unique(new_test_label))
    print("**************************************************")
    sum = 1
    for c in unlabeled_class:
        new_train_label[(train_label > c) & (new_train_label != 0)] = train_label[(train_label > c) & (
                new_train_label != 0)] - sum
        print("The label of train label-change class index")
        print(np.unique(new_train_label))
        new_test_label[(test_label > c) & (new_test_label != -1)] = test_label[
                                                                        (test_label > c) & (new_test_label != -1)] - sum
        print("The label of new test label-change class index")
        print(np.unique(new_test_label))
        sum += 1
    print("**************************************************")
    new_test_label[new_test_label == -1] = new_total_class
    new_test_label = new_test_label.astype(np.int8)
    print("The label of new train label")
    print(np.unique(new_train_label))
    print("The label of new test label")
    print(np.unique(new_test_label))
    print("**************************************************")
    print("The number of background class of new train label:%f" % len(np.where(new_train_label == 0)[0]))
    print("The number of background class of new test label:%f" % len(np.where(new_test_label == 0)[0]))
    print("**************************************************")
    print(len(np.where(new_train_label != 0)[0]))
    np.save(new_train_label_save_path, new_train_label)
    train_im = Image.fromarray(new_train_label)
    train_im.save(new_train_img_save_path)
    np.save(new_test_label_save_path, new_test_label)
    test_im = Image.fromarray(new_test_label)
    test_im.save(new_test_img_save_path)
