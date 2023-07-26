import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm


class TorchDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return self.data.shape[0]


class PR(BaseEstimator):
    def __init__(self, estimator: nn.Module, epoch: int, optimizer, batch_size: int = 256, random_seed=2333):
        self.estimator = estimator.cuda()
        self.epoch = epoch
        self.optimizer = optimizer
        self.batch_size = batch_size

        self.random_seed = random_seed

    def define_train_data(self, X, y):
        np.random.seed(self.random_seed)

        id_x = np.random.permutation(X.shape[0])
        X = X[id_x]
        y = y[id_x]

        self.dataset = TorchDataset(data=torch.from_numpy(X).float(), label=torch.from_numpy(y).long())

    def define_test_data(self, X):
        self.test_data = X

    def fit(self, X, y):
        dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size,
                                sampler=SubsetRandomSampler(X))
        self.estimator.train()
        bar = tqdm(range(self.epoch))
        for _ in bar:
            training_loss = 0.0
            num = 0
            for x, y in dataloader:
                x, y = x.cuda(), y.cuda()
                targets = self.estimator(x).squeeze()
                loss = nn.CrossEntropyLoss()(targets, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                training_loss += loss.item()
                num += 1
            bar.set_description('loss: %.4f' % (training_loss / num))

    def predict_proba(self, data: np.ndarray):
        data = self.test_data[data]
        self.estimator.eval()
        with torch.no_grad():
            probs = torch.softmax(
                self.estimator(torch.from_numpy(data).float().cuda()), dim=1).squeeze().cpu().detach().numpy()

        return probs

    def predict(self, data: np.ndarray):
        probs = self.predict_proba(data)
        return probs.argmax(axis=1)
