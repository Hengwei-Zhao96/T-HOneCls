import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class TorchDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return self.data.shape[0]


class PUCalibration:
    def __init__(self, estimator: nn.Module, epoch: int, optimizer, batch_size: int = 256, hold_out_ratio=0.2,
                 mode='pul', random_seed=2333):
        self.estimator = estimator.cuda()
        self.epoch = epoch
        self.optimizer = optimizer
        self.batch_size = batch_size

        self.hold_out_ratio = hold_out_ratio
        self.mode = mode
        self.random_seed = random_seed

        self.c = None
        self.result = None
        self.result_p = None

    def fit(self, positive_data: np.ndarray, unlabeled_data: np.ndarray):
        np.random.seed(self.random_seed)
        positive_size = positive_data.shape[0]
        positive_index = np.random.permutation(positive_size)
        hold_out_size = int(np.ceil(positive_size * self.hold_out_ratio))

        hold_out_index = positive_index[:hold_out_size]
        training_index = positive_index[hold_out_size:]

        hold_out_positive_data = positive_data[hold_out_index]
        training_positive_data = positive_data[training_index]

        train_data = np.concatenate((training_positive_data, unlabeled_data), axis=0)
        train_label = np.concatenate(
            (np.ones(training_positive_data.shape[0]), np.zeros(unlabeled_data.shape[0])), axis=0)
        id_x = np.random.permutation(train_data.shape[0])
        train_data = train_data[id_x]
        train_label = train_label[id_x]

        dataset = TorchDataset(data=torch.from_numpy(train_data).float(), label=torch.from_numpy(train_label).float())
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        self.estimator.train()
        bar = tqdm(range(self.epoch))
        for _ in bar:
            training_loss = 0.0
            num = 0
            for x, y in dataloader:
                x, y = x.cuda(), y.cuda()
                targets = self.estimator(x).squeeze()
                loss = nn.BCEWithLogitsLoss()(targets, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                training_loss += loss.item()
                num += 1
            bar.set_description('loss: %.4f' % (training_loss / num))

        hold_out_pro = torch.sigmoid(
            self.estimator(torch.from_numpy(hold_out_positive_data).float().cuda())).squeeze().cpu().detach().numpy()

        self.c = np.mean(hold_out_pro)

    def predict(self, data: np.ndarray):
        self.estimator.eval()
        with torch.no_grad():
            pro = torch.sigmoid(self.estimator(torch.from_numpy(data).float().cuda())).squeeze().cpu().detach().numpy()

        if self.mode == 'pul':
            self.result_p = pro / self.c
        elif self.mode == 'pbl':
            self.result_p = (1 - self.c) / self.c * pro / (1 - pro)

        self.result = np.where(self.result_p >= 0.5, 1, 0)
        return self.result
