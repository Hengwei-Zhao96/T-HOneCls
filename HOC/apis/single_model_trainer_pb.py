import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader as torchDataLoader
from tqdm import tqdm

from HOC.apis import BaseTrainer
from HOC.apis.validation import evaluate_fn
from HOC.datasets import build_dataloader
from HOC.datasets.data_base import HyperData
from HOC.loss_functions import build_loss_function
from HOC.models import build_model
from HOC.optimization import build_optimizer
from HOC.optimization import build_lr_scheduler
from HOC.utils import TRAINER


def _get_patch_data(img_data, x_indicator: np.ndarray, y_indicator: np.ndarray, patch_size: int):
    length = x_indicator.shape[0]
    r = int(patch_size - 1) / 2
    data = np.zeros((length, img_data.shape[0], patch_size, patch_size))
    for i in range(length):
        data[i, :, :, :] = img_data[:, int(x_indicator[i] - r):int(x_indicator[i] + r + 1),
                           int(y_indicator[i] - r):int(y_indicator[i] + r + 1)]
    return data.squeeze()


@TRAINER.register_module()
class SingleModelTrainer_PB(BaseTrainer):
    def __init__(self, trainer, dataset, model, loss_function, optimizer, lr_scheduler, meta):
        super(SingleModelTrainer_PB, self).__init__(trainer=trainer, dataset=dataset, model=model,
                                                    loss_function=loss_function,
                                                    optimizer=optimizer, lr_scheduler=lr_scheduler, meta=meta)
        self.build_scalar_recoder()

    def _build_train_dataloader(self):
        train_pf_dataloader = build_dataloader(self.cfg_dataset['train'])
        r = int((self.cfg_trainer['params']['patch_size'] - 1) / 2)

        image = train_pf_dataloader.dataset.im
        image = np.pad(image, ((0, 0), (r, r), (r, r)), mode='constant')
        positive_train_indicator = train_pf_dataloader.dataset.positive_train_indicator
        positive_train_indicator = np.pad(positive_train_indicator, (r, r), mode='constant')
        unlabeled_train_indicator = train_pf_dataloader.dataset.unlabeled_train_indicator
        unlabeled_train_indicator = np.pad(unlabeled_train_indicator, (r, r), mode='constant')

        train_positive_id_x, train_positive_id_y = np.where(positive_train_indicator == 1)
        train_unlabeled_id_x, train_unlabeled_id_y = np.where(unlabeled_train_indicator == 1)

        positive_train_data = _get_patch_data(image, train_positive_id_x, train_positive_id_y,
                                              self.cfg_trainer['params']['patch_size'])
        unlabeled_train_data = _get_patch_data(image, train_unlabeled_id_x, train_unlabeled_id_y,
                                               self.cfg_trainer['params']['patch_size'])

        train_data = np.concatenate((positive_train_data, unlabeled_train_data))
        label = np.concatenate((np.ones(positive_train_data.shape[0]), np.zeros(unlabeled_train_data.shape[0])))
        inds = np.random.permutation(train_data.shape[0])
        train_data = train_data[inds]
        label = label[inds]

        dataset = HyperData(torch.from_numpy(train_data).float(), torch.from_numpy(label))
        return image, torchDataLoader(dataset=dataset, batch_size=self.cfg_trainer['params']['batch_size_pb'],
                                      shuffle=True)

    def _build_validation_dataloader(self):
        validation_pf_dataloader = build_dataloader(self.cfg_dataset['test'])
        positive_validation_indicator = validation_pf_dataloader.dataset.positive_test_indicator
        negative_validation_indicator = validation_pf_dataloader.dataset.negative_test_indicator
        return positive_validation_indicator, negative_validation_indicator

    def _build_model(self):
        return build_model(self.cfg_model)

    def _build_loss_function(self):
        return build_loss_function(self.cfg_loss_function)

    def _build_optimizer(self):
        self.cfg_optimizer['params'].update(params=self.model.parameters())
        return build_optimizer(self.cfg_optimizer)

    def _build_lr_scheduler(self):
        self.cfg_lr_scheduler['params'].update(optimizer=self.optimizer)
        return build_lr_scheduler(self.cfg_lr_scheduler)

    def _build_components(self):
        self.image, self.train_dataloader = self._build_train_dataloader()
        self.positive_validation_indicator, self.negative_validation_indicator = self._build_validation_dataloader()
        self.model = self._build_model().to(self.device)
        self.loss_function = self._build_loss_function()
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()

    def build_scalar_recoder(self):
        self.f1_recorder = self._build_scalar_recoder()
        self.pre_recorder = self._build_scalar_recoder()
        self.rec_recorder = self._build_scalar_recoder()
        self.loss_recorder = self._build_scalar_recoder()
        self.p_loss_recorder = self._build_scalar_recoder()
        self.u_loss_recorder = self._build_scalar_recoder()
        self.grad_l2_norm = self._build_scalar_recoder()

    def _train(self, epoch):
        epoch_loss = 0.0
        epoch_p_loss = 0.0
        epoch_u_loss = 0.0
        num_iter = 0
        self.model.train()
        for (data, y) in self.train_dataloader:
            data = data.to(self.device)
            y = y.to(self.device)

            target = self.model(data)

            loss, p_loss, u_loss = self.loss_function(target, y, epoch, self.device)

            self.optimizer.zero_grad()
            loss.backward()

            self.clip_recoder_grad_norm(model=self.model, grad_l2_norm=self.grad_l2_norm)

            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_p_loss += p_loss.item()
            epoch_u_loss += u_loss.item()
            num_iter += 1

        self.lr_scheduler.step()
        self.loss_recorder.update_scalar(epoch_loss / num_iter)
        self.p_loss_recorder.update_scalar(epoch_p_loss / num_iter)
        self.u_loss_recorder.update_scalar(epoch_u_loss / num_iter)

    def _validation(self, epoch):
        auc, fpr, tpr, threshold, pre, rec, f1 = evaluate_fn(image=self.image,
                                                             positive_test_indicator=self.positive_validation_indicator,
                                                             negative_test_indicator=self.negative_validation_indicator,
                                                             cls=self.cfg_dataset['train']['params']['ccls'],
                                                             model=self.model,
                                                             patch_size=self.cfg_trainer['params']['patch_size'],
                                                             meta=self.meta,
                                                             device=self.device,
                                                             path=self.save_path,
                                                             epoch=epoch)

        self.pre_recorder.update_scalar(pre)
        self.rec_recorder.update_scalar(rec)
        self.f1_recorder.update_scalar(f1)

        if epoch == (self.cfg_trainer['params']['max_iters'] - 1):
            auc_roc = dict()
            auc_roc['fpr'] = fpr
            auc_roc['tpr'] = tpr
            auc_roc['threshold'] = threshold
            auc_roc['auc'] = auc
            np.save(os.path.join(self.save_path, 'auc_roc.npy'), auc_roc)

    def _save_checkpoint(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, 'checkpoint.pth'))

    def run(self, validation=True):
        bar = tqdm(list(range(self.cfg_trainer['params']['max_iters'])))
        for i in bar:
            self._train(i)

            if i == self.cfg_trainer['params']['max_iters'] - 1:
                validation = True
            else:
                validation = False

            if validation:
                self._validation(i)
            else:
                self.pre_recorder.update_scalar(0)
                self.rec_recorder.update_scalar(0)
                self.f1_recorder.update_scalar(0)

            logging_string = "{} epoch, loss {:.4f}, p_loss {:.4f}, u_loss {:.4f}, P {:.6f}, R {:.6f}, F1 {:.6f}".format(
                i + 1,
                self.loss_recorder.scalar[-1],
                self.p_loss_recorder.scalar[-1],
                self.u_loss_recorder.scalar[-1],
                self.pre_recorder.scalar[-1],
                self.rec_recorder.scalar[-1],
                self.f1_recorder.scalar[-1])

            bar.set_description(logging_string)
            logging.info(logging_string)

        self._save_checkpoint()

        self.loss_recorder.save_scalar_npy('loss_npy', self.save_path)
        self.loss_recorder.save_lineplot_fig('Loss', 'loss', self.save_path)
        self.p_loss_recorder.save_scalar_npy('p_loss_npy', self.save_path)
        self.p_loss_recorder.save_lineplot_fig('Estimated Positive Loss', 'p_loss', self.save_path)
        self.u_loss_recorder.save_scalar_npy('u_loss_npy', self.save_path)
        self.u_loss_recorder.save_lineplot_fig('Estimated Unlabeled Loss', 'u_loss', self.save_path)
        self.grad_l2_norm.save_scalar_npy('grad_l2_norm_npy', self.save_path)
        self.grad_l2_norm.save_lineplot_fig('Grad L2 Norm', 'grad_l2_norm', self.save_path)
        if validation:
            self.f1_recorder.save_scalar_npy('f1_npy', self.save_path)
            self.f1_recorder.save_lineplot_fig('F1-score', 'f1-score', self.save_path)
            self.rec_recorder.save_scalar_npy('recall_npy', self.save_path)
            self.rec_recorder.save_lineplot_fig('Recall', 'recall', self.save_path)
            self.pre_recorder.save_scalar_npy('precision_npy', self.save_path)
            self.pre_recorder.save_lineplot_fig('Precision', 'precision', self.save_path)
