from osgeo import gdal
import numpy as np
from matplotlib import pyplot


def read_ENVI(filepath):
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, cols, rows)
    if len(data.shape) == 3:
        data = data.transpose((1, 2, 0))
    return data


def save_mat(path, data):
    import scipy.io as scio
    scio.savemat(path, {'spectral': data})


def read_mat():
    import scipy.io as scio
    data = scio.loadmat(r'E:\FGOCC\UAVdata\data_all.mat')
    s1 = scio.loadmat(r'E:\FGOCC\UAVdata\data_end1.mat')
    s2 = scio.loadmat(r'E:\FGOCC\UAVdata\data_end2.mat')
    print()


if __name__ == "__main__":
    cid1 = 11
    cid2 = 18
    mean = False

    image = read_ENVI(r'E:\FGOCC\UAVdata\data')
    mask = read_ENVI(r'E:\FGOCC\UAVdata\gt')

    id1_x, id1_y = np.where((mask == cid1))
    id2_x, id2_y = np.where((mask == cid2))

    spectral_1 = image[id1_x, id1_y]
    spectral_2 = image[id2_x, id2_y]

    band_id = np.asarray(range(1, image.shape[2] + 1))

    spectral_1_mean = np.mean(spectral_1, axis=0)
    spectral_2_mean = np.mean(spectral_2, axis=0)

    if mean:
        pyplot.plot(band_id, spectral_1_mean)
        pyplot.plot(band_id, spectral_2_mean)


    else:
        std_1 = np.std(spectral_1, axis=0)
        std_2 = np.std(spectral_2, axis=0)

        # pyplot.plot(band_id, spectral_1_mean+std_1)
        # pyplot.plot(band_id, spectral_1_mean-std_1)

        pyplot.plot(band_id, spectral_2_mean + std_2)
        pyplot.plot(band_id, spectral_2_mean - std_2)

    pyplot.show()
