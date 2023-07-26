from HOC.datasets.data_base import OccFullImageDataset
from HOC.datasets.data_utils import mean_std_normalize
from HOC.datasets.data_utils.image_io import read_ENVI
from HOC.utils.registry import DATASETS

SEED = 2333


@DATASETS.register_module()
class HanChuanDataset(OccFullImageDataset):
    def __init__(self, image_path, gt_path, train_flage, num_positive_train_samples, sub_minibatch, ccls, ratio):
        self.im_path = image_path
        self.gt_path = gt_path

        image = read_ENVI(self.im_path)
        mask = read_ENVI(self.gt_path)

        self.train_flag = train_flage
        self.cls = ccls
        self.num_train_samples_per_class = num_positive_train_samples

        im_cmean = image.reshape((-1, image.shape[-1])).mean(axis=0)
        im_cstd = image.reshape((-1, image.shape[-1])).std(axis=0)
        image = mean_std_normalize(image, im_cmean, im_cstd)

        super(HanChuanDataset, self).__init__(image=image,
                                              mask=mask,
                                              cls=self.cls,
                                              ratio=ratio,
                                              train_flag=self.train_flag,
                                              np_seed=SEED,
                                              num_train_samples_per_class=self.num_train_samples_per_class,
                                              sub_num_iter=sub_minibatch)
