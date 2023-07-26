import torch
from torch.utils.data import Dataset, Sampler


class DS3LHyperData(Dataset):
    def __init__(self, dataset, label, transformation=None):
        self.dataset = dataset
        self.labels = label
        self.transformation = transformation

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        if self.transformation is None:
            return self.dataset[item, :, :, :], self.labels[item], item
        else:
            return self.transformation(self.dataset[item, :, :, :]), self.labels[item], item


class DS3LRandomSampler(Sampler):
    """ sampling without replacement """

    def __init__(self, num_data, num_sample):
        iterations = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
