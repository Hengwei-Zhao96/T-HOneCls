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


if __name__ == "__main__":
    data = read_ENVI("/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-HongHu/WHU-Hi-HongHu")
    np.save("/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-HongHu/hh.npy", data)
