import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from HOC.apis import BaseTrainer
from HOC.apis.validation import fcn_evaluate_fn
from HOC.apis.toolkit_gan import pan_training
from HOC.datasets import build_dataloader
from HOC.models import build_model
from HOC.optimization import build_optimizer
from HOC.optimization import build_lr_scheduler
from HOC.utils import TRAINER


@TRAINER.register_module()
class PANTrainer(BaseTrainer):
    def __init__(self, trainer, dataset, model, loss_function, optimizer, lr_scheduler, meta):
        super(PANTrainer, self).__init__(trainer=trainer, dataset=dataset, model=model,
                                         loss_function=loss_function,
                                         optimizer=optimizer, lr_scheduler=lr_scheduler, meta=meta)
        self.build_scalar_recoder()

    def _build_train_dataloader(self):
        return build_dataloader(self.cfg_dataset['train'])

    def _build_validation_dataloader(self):
        return build_dataloader(self.cfg_dataset['test'])

    def _build_model(self):
        return build_model(self.cfg_model)

    def _build_loss_function(self):
        pass

    def _build_optimizer(self):
        self.cfg_optimizer['params'].update(params=self.model_d.parameters())
        optimizer_d = build_optimizer(self.cfg_optimizer)

        self.cfg_optimizer['params'].update(params=self.model_c.parameters())
        optimizer_c = build_optimizer(self.cfg_optimizer)

        return optimizer_d, optimizer_c

    def _build_lr_scheduler(self):
        self.cfg_lr_scheduler['params'].update(optimizer=self.optimizer_d)
        lr_scheduler_d = build_lr_scheduler(self.cfg_lr_scheduler)

        self.cfg_lr_scheduler['params'].update(optimizer=self.optimizer_c)
        lr_scheduler_c = build_lr_scheduler(self.cfg_lr_scheduler)

        return lr_scheduler_d, lr_scheduler_c

    def _build_components(self):
        self.train_dataloader = self._build_train_dataloader()
        self.validation_dataloader = self._build_validation_dataloader()
        self.model_d = self._build_model().to(self.device)
        self.model_c = self._build_model().to(self.device)
        self.optimizer_d, self.optimizer_c = self._build_optimizer()
        self.lr_scheduler_d, self.lr_scheduler_c = self._build_lr_scheduler()

    def build_scalar_recoder(self):
        self.f1_recorder_d = self._build_scalar_recoder()
        self.pre_recorder_d = self._build_scalar_recoder()
        self.rec_recorder_d = self._build_scalar_recoder()
        self.grad_l2_norm_d = self._build_scalar_recoder()

        self.f1_recorder_c = self._build_scalar_recoder()
        self.pre_recorder_c = self._build_scalar_recoder()
        self.rec_recorder_c = self._build_scalar_recoder()
        self.grad_l2_norm_c = self._build_scalar_recoder()

        self.loss_recorder_d = self._build_scalar_recoder()
        self.loss_recorder_c = self._build_scalar_recoder()

    def _train(self, epoch):
        epoch_loss_d = 0.0
        epoch_loss_c = 0.0

        num_iter_d = 0
        for (data, positive_train_mask, unlabeled_train_mask) in self.train_dataloader:
            data = data.to(self.device)
            positive_train_mask = positive_train_mask.to(self.device)
            unlabeled_train_mask = unlabeled_train_mask.to(self.device)

            self.model_d.train()
            self.model_c.eval()

            d_target_d = self.model_d(data)
            d_target_c = self.model_c(data)

            loss_d = pan_training(target_d=d_target_d, target_c=d_target_c, positive_train_mask=positive_train_mask,
                                  unlabeled_train_mask=unlabeled_train_mask,
                                  weight=self.cfg_trainer['params']['weight'], adversarial=False)

            self.optimizer_d.zero_grad()
            loss_d.backward()
            self.clip_recoder_grad_norm(model=self.model_d, grad_l2_norm=self.grad_l2_norm_d)
            self.optimizer_d.step()

            epoch_loss_d += loss_d.item()

            num_iter_d += 1
        self.lr_scheduler_d.step()
        self.loss_recorder_d.update_scalar(epoch_loss_d / num_iter_d)

        num_iter_c = 0
        for (data, positive_train_mask, unlabeled_train_mask) in self.train_dataloader:
            data = data.to(self.device)
            positive_train_mask = positive_train_mask.to(self.device)
            unlabeled_train_mask = unlabeled_train_mask.to(self.device)

            self.model_d.eval()
            self.model_c.train()

            c_target_d = self.model_d(data)
            c_target_c = self.model_c(data)

            loss_c = pan_training(target_d=c_target_d, target_c=c_target_c, positive_train_mask=positive_train_mask,
                                  unlabeled_train_mask=unlabeled_train_mask,
                                  weight=self.cfg_trainer['params']['weight'], adversarial=True)

            self.optimizer_c.zero_grad()
            loss_c.backward()
            self.clip_recoder_grad_norm(model=self.model_c, grad_l2_norm=self.grad_l2_norm_c)
            self.optimizer_c.step()

            epoch_loss_c += loss_c.item()

            num_iter_c += 1
        self.lr_scheduler_c.step()
        self.loss_recorder_c.update_scalar(epoch_loss_c / num_iter_c)

    def _validation(self, epoch):
        auc_d, fpr_d, tpr_d, threshold_d, pre_d, rec_d, f1_d = fcn_evaluate_fn(
            model=self.model_d,
            test_dataloader=self.validation_dataloader,
            meta=self.meta,
            cls=self.cfg_dataset['train']['params']['ccls'],
            device=self.device,
            path=self.save_path_d,
            epoch=epoch)

        auc_c, fpr_c, tpr_c, threshold_c, pre_c, rec_c, f1_c = fcn_evaluate_fn(
            model=self.model_c,
            test_dataloader=self.validation_dataloader,
            meta=self.meta,
            cls=self.cfg_dataset['train']['params']['ccls'],
            device=self.device,
            path=self.save_path_c,
            epoch=epoch)

        self.pre_recorder_d.update_scalar(pre_d)
        self.rec_recorder_d.update_scalar(rec_d)
        self.f1_recorder_d.update_scalar(f1_d)

        self.pre_recorder_c.update_scalar(pre_c)
        self.rec_recorder_c.update_scalar(rec_c)
        self.f1_recorder_c.update_scalar(f1_c)

        if epoch == (self.cfg_trainer['params']['max_iters'] - 1):
            auc_roc_d = dict()
            auc_roc_d['fpr'] = fpr_d
            auc_roc_d['tpr'] = tpr_d
            auc_roc_d['threshold'] = threshold_d
            auc_roc_d['auc'] = auc_d
            np.save(os.path.join(self.save_path_d, 'auc_roc_d.npy'), auc_roc_d)

            auc_roc_c = dict()
            auc_roc_c['fpr'] = fpr_c
            auc_roc_c['tpr'] = tpr_c
            auc_roc_c['threshold'] = threshold_c
            auc_roc_c['auc'] = auc_c
            np.save(os.path.join(self.save_path_c, 'auc_roc_c.npy'), auc_roc_c)

    def _save_checkpoint(self):
        torch.save(self.model_d.state_dict(), os.path.join(self.save_path_d, 'checkpoint_d.pth'))
        torch.save(self.model_c.state_dict(), os.path.join(self.save_path_c, 'checkpoint_c.pth'))

    def run(self, validation=True):
        self.save_path_d = os.path.join(self.save_path, 'model_d')
        self.save_path_c = os.path.join(self.save_path, 'model_c')
        bar = tqdm(list(range(self.cfg_trainer['params']['max_iters'])))
        for i in bar:
            self._train(i)
            if validation:
                self._validation(i)
            else:
                self.pre_recorder_d.update_scalar(0)
                self.rec_recorder_d.update_scalar(0)
                self.f1_recorder_d.update_scalar(0)
                self.pre_recorder_c.update_scalar(0)
                self.rec_recorder_c.update_scalar(0)
                self.f1_recorder_c.update_scalar(0)

            logging_string = "{} epoch, loss_d {:.4f}, loss_c {:.4f}, P_d {:.6f}, R_d {:.6f}, F1_d {:.6f}, P_c {:.6f}," \
                             " R_c {:.6f}, F1_c {:.6f}".format(
                i + 1,
                self.loss_recorder_d.scalar[-1],
                self.loss_recorder_c.scalar[-1],
                self.pre_recorder_d.scalar[-1],
                self.rec_recorder_d.scalar[-1],
                self.f1_recorder_d.scalar[-1],
                self.pre_recorder_c.scalar[-1],
                self.rec_recorder_c.scalar[-1],
                self.f1_recorder_c.scalar[-1]
            )

            bar.set_description(logging_string)
            logging.info(logging_string)

        self._save_checkpoint()

        self.loss_recorder_d.save_scalar_npy('loss_npy_d', self.save_path_d)
        self.loss_recorder_d.save_lineplot_fig('Loss', 'loss_d', self.save_path_d)
        self.grad_l2_norm_d.save_scalar_npy('grad_l2_norm_npy_d', self.save_path_d)
        self.grad_l2_norm_d.save_lineplot_fig('Grad L2 Norm', 'grad_l2_norm_d', self.save_path_d)

        self.loss_recorder_c.save_scalar_npy('loss_npy_c', self.save_path_c)
        self.loss_recorder_c.save_lineplot_fig('Loss', 'loss_c', self.save_path_c)
        self.grad_l2_norm_c.save_scalar_npy('grad_l2_norm_npy_c', self.save_path_c)
        self.grad_l2_norm_c.save_lineplot_fig('Grad L2 Norm', 'grad_l2_norm_c', self.save_path_c)

        if validation:
            self.f1_recorder_d.save_scalar_npy('f1_npy_1', self.save_path_d)
            self.f1_recorder_d.save_lineplot_fig('F1-score', 'f1-score_d', self.save_path_d)
            self.rec_recorder_d.save_scalar_npy('recall_npy_d', self.save_path_d)
            self.rec_recorder_d.save_lineplot_fig('Recall', 'recall_d', self.save_path_d)
            self.pre_recorder_d.save_scalar_npy('precision_npy_d', self.save_path_d)
            self.pre_recorder_d.save_lineplot_fig('Precision', 'precision_d', self.save_path_d)

            self.f1_recorder_c.save_scalar_npy('f1_npy_c', self.save_path_c)
            self.f1_recorder_c.save_lineplot_fig('F1-score', 'f1-score_c', self.save_path_c)
            self.rec_recorder_c.save_scalar_npy('recall_npy_c', self.save_path_c)
            self.rec_recorder_c.save_lineplot_fig('Recall', 'recall_c', self.save_path_c)
            self.pre_recorder_c.save_scalar_npy('precision_npy_c', self.save_path_c)
            self.pre_recorder_c.save_lineplot_fig('Precision', 'precision_c', self.save_path_c)
