import copy
from osgeo import gdal
import numpy as np


def read_ENVI(filepath):
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, cols, rows)
    if len(data.shape) == 3:
        data = data.transpose((1, 2, 0))
    return data


in_distribution_classes = [4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
total_classes = 22
gt_path = '/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-HongHu/WHU-Hi-HongHu_gt'
save_path = '/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-HongHu/disjoint_unlabeled'

out_of_distribution_classes = []
for i in range(1, total_classes + 1):
    if i in in_distribution_classes:
        continue
    else:
        out_of_distribution_classes.append(i)

gt = read_ENVI(gt_path)

new_gt = copy.deepcopy(gt)

for i in in_distribution_classes:
    new_gt[np.where(gt == i)] = 0

mask = new_gt.ravel()
indicater = np.zeros_like(mask)
ids = np.where(mask != 0)[0]
rs = np.random.RandomState(2333)
rs.shuffle(ids)
indicater[ids[:4000]] = 1
indicater = indicater.reshape(gt.shape)

np.save(save_path, indicater)
