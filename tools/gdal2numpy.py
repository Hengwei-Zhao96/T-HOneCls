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
    path = r"K:\项目材料\173-赵\数据库\昼间数据\昼间城市模拟数据\标签\gt"
    data = read_ENVI(path)
    np.save(r"K:\项目材料\173-赵\数据库\昼间数据\最终版\公园\gt_npy.npy", data)
    print('Done!')
