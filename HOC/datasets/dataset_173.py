from HOC.datasets.data_base import OccFullImageDataset
from HOC.datasets.data_utils import mean_std_normalize, get_train_test_label
from HOC.datasets.data_utils.image_io import read_ENVI
from HOC.utils.registry import DATASETS

SEED = 2333


@DATASETS.register_module()
class OneSevenThreeDataset(OccFullImageDataset):
    def __init__(self, image_path, gt_path, train_flage, num_positive_train_samples, sub_minibatch, ccls, ratio):
        self.im_path = image_path
        self.gt_path = gt_path

        image = read_ENVI(self.im_path)
        mask = read_ENVI(self.gt_path)

        self.train_flag = train_flage
        self.num_train_samples_per_class = num_positive_train_samples
        self.sub_minibatch = sub_minibatch
        self.cls = ccls
        self.ratio = ratio

        im_cmean = image.reshape((-1, image.shape[-1])).mean(axis=0)
        im_cstd = image.reshape((-1, image.shape[-1])).std(axis=0)
        image = mean_std_normalize(image, im_cmean, im_cstd)

        train_label, test_label = get_train_test_label(cls=self.cls, raw_label=mask,
                                                       num_positive_train_samples=self.num_train_samples_per_class,
                                                       seed=SEED)

        if self.train_flag:
            mask = train_label
        else:
            mask = test_label

        super(OneSevenThreeDataset, self).__init__(image=image,
                                                   mask=mask,
                                                   cls=self.cls,
                                                   ratio=self.ratio,
                                                   train_flag=self.train_flag,
                                                   np_seed=SEED,
                                                   num_train_samples_per_class=self.num_train_samples_per_class,
                                                   sub_num_iter=self.sub_minibatch)
