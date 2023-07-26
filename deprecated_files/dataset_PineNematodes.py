from osgeo import gdal
import HOC.datasets.data_utils.data_preprocess as preprocess
from HOC.datasets.data_base import OccFullImageDataset
from HOC.datasets.data_utils import get_train_test_label

SEED = 2333


def read_ENVI(filepath):
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, cols, rows)
    if len(data.shape) == 3:
        data = data.transpose((1, 2, 0))
    return data


class NewOccPineDataset(OccFullImageDataset):
    def __init__(self, config):
        self.im_mat_path = config['image_mat_path']
        self.gt_mat_path = config['gt_mat_path']

        image = read_ENVI(self.im_mat_path)
        mask = read_ENVI(self.gt_mat_path)

        self.train_flag = config['train_flage']
        self.cls = config['cls']
        self.num_train_samples_per_class = config['num_positive_train_samples']

        im_cmean = image.reshape((-1, image.shape[-1])).mean(axis=0)
        im_cstd = image.reshape((-1, image.shape[-1])).std(axis=0)
        image = preprocess.mean_std_normalize(image, im_cmean, im_cstd)

        train_label, test_label = get_train_test_label(cls=self.cls, raw_label=mask,
                                                       num_positive_train_samples=self.num_train_samples_per_class,
                                                       seed=SEED)

        if self.train_flag:
            mask = train_label
        else:
            mask = test_label

        super(NewOccPineDataset, self).__init__(image=image,
                                                mask=mask,
                                                cls=self.cls,
                                                ratio=config['ratio'],
                                                train_flag=self.train_flag,
                                                np_seed=SEED,
                                                num_train_samples_per_class=self.num_train_samples_per_class,
                                                sub_num_iter=config['sub_minibatch'])
