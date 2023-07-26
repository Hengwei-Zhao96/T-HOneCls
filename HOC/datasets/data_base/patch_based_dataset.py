from torch.utils.data import Dataset


class HyperData(Dataset):
    def __init__(self, dataset, label, transformation=None):
        self.dataset = dataset
        self.labels = label
        self.transformation = transformation

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        if self.transformation is None:
            return self.dataset[item, :, :, :], self.labels[item]
        else:
            return self.transformation(self.dataset[item, :, :, :]), self.labels[item]
