from torch.utils.data.dataloader import DataLoader

from HOC.datasets.data_base import MinibatchSampler
from HOC.utils import DATASETS
from HOC.utils.build import build_from_cfg


class HOCDataLoader(DataLoader):
    def __init__(self, dataset):
        self.dataset = dataset
        sampler = MinibatchSampler(self.dataset)
        super(HOCDataLoader, self).__init__(dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            sampler=sampler,
                                            batch_sampler=None,
                                            pin_memory=True,
                                            drop_last=False,
                                            timeout=0,
                                            worker_init_fn=None)


def build_dataloader(cfg):
    dataset = build_from_cfg(cfg, DATASETS)
    return HOCDataLoader(dataset)
