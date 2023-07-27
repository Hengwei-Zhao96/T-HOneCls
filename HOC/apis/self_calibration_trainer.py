import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from HOC.apis import BaseTrainer
from HOC.apis.validation import fcn_evaluate_fn
from HOC.datasets import build_dataloader
from HOC.loss_functions import build_loss_function
from HOC.models import build_model
from HOC.optimization import build_optimizer
from HOC.optimization import build_lr_scheduler
from HOC.apis.toolkit_self_calibration.kl_loss import KLLoss
from HOC.apis.toolkit_self_calibration.update_ema_model import update_ema_variables
from HOC.utils import TRAINER


@TRAINER.register_module()
class SelfCalibrationTrainer(BaseTrainer):
    def __init__(self, trainer, dataset, model, loss_function, optimizer, lr_scheduler, meta):
        super(SelfCalibrationTrainer, self).__init__(trainer=trainer, dataset=dataset, model=model,
                                                     loss_function=loss_function,
                                                     optimizer=optimizer, lr_scheduler=lr_scheduler, meta=meta)
        self.build_scalar_recoder()
        self.ema_label = None

    def _build_train_dataloader(self):
        return build_dataloader(self.cfg_dataset['train'])

    def _build_validation_dataloader(self):
        return build_dataloader(self.cfg_dataset['test'])

    def _build_model(self):
        return build_model(self.cfg_model)

    def _build_loss_function(self):
        return build_loss_function(self.cfg_loss_function)

    def _build_optimizer(self):
        self.cfg_optimizer['params'].update(params=self.model_s.parameters())
        return build_optimizer(self.cfg_optimizer)

    def _build_lr_scheduler(self):
        self.cfg_lr_scheduler['params'].update(optimizer=self.optimizer_s)
        return build_lr_scheduler(self.cfg_lr_scheduler)

    def _build_components(self):
        self.train_dataloader = self._build_train_dataloader()
        self.validation_dataloader = self._build_validation_dataloader()
        self.model_s = self._build_model().to(self.device)
        self.model_t = self._build_model().to(self.device)
        self.loss_function_s = self._build_loss_function()
        self.optimizer_s = self._build_optimizer()
        self.lr_scheduler_s = self._build_lr_scheduler()

    def build_scalar_recoder(self):
        self.f1_recorder_s = self._build_scalar_recoder()
        self.pre_recorder_s = self._build_scalar_recoder()
        self.rec_recorder_s = self._build_scalar_recoder()
        self.loss_recorder_s = self._build_scalar_recoder()
        self.p_loss_recorder_s = self._build_scalar_recoder()
        self.u_loss_recorder_s = self._build_scalar_recoder()
        self.grad_l2_norm_s = self._build_scalar_recoder()

        self.f1_recorder_t = self._build_scalar_recoder()
        self.pre_recorder_t = self._build_scalar_recoder()
        self.rec_recorder_t = self._build_scalar_recoder()
        self.loss_recorder_t = self._build_scalar_recoder()

        self.loss_recorder_s_t = self._build_scalar_recoder()

    def _train(self, epoch):
        epoch_loss_s = 0.0
        epoch_p_loss_s = 0.0
        epoch_u_loss_s = 0.0
        epoch_loss_t = 0.0
        epoch_loss_s_t = 0.0

        num_iter = 0
        self.model_s.train()
        for (data, positive_train_mask, unlabeled_train_mask) in self.train_dataloader:
            data = data.to(self.device)
            positive_train_mask = positive_train_mask.to(self.device)
            unlabeled_train_mask = unlabeled_train_mask.to(self.device)

            target_s = self.model_s(data)
            with torch.no_grad():
                target_t = self.model_t(data)

            loss_s, p_loss_s, u_loss_s = self.loss_function_s(target_s, positive_train_mask, unlabeled_train_mask,
                                                              epoch, self.device)

            loss_t = KLLoss(equal_weight=False)(target_s, target_t, positive_train_mask, unlabeled_train_mask)

            loss_s_t = loss_s + self.cfg_trainer['params']['beta'] * loss_t

            self.optimizer_s.zero_grad()
            loss_s_t.backward()

            self.clip_recoder_grad_norm(model=self.model_s, grad_l2_norm=self.grad_l2_norm_s)

            self.optimizer_s.step()

            update_ema_variables(model=self.model_s, ema_model=self.model_t,
                                 alpha=self.cfg_trainer['params']['ema_model_alpha'],
                                 global_step=epoch)

            epoch_loss_s += loss_s.item()
            epoch_p_loss_s += p_loss_s.item()
            epoch_u_loss_s += u_loss_s.item()
            epoch_loss_t += loss_t.item()
            epoch_loss_s_t += loss_s_t.item()
            num_iter += 1

        self.lr_scheduler_s.step()

        self.loss_recorder_s.update_scalar(epoch_loss_s / num_iter)
        self.p_loss_recorder_s.update_scalar(epoch_p_loss_s / num_iter)
        self.u_loss_recorder_s.update_scalar(epoch_u_loss_s / num_iter)
        self.loss_recorder_t.update_scalar(epoch_loss_t / num_iter)
        self.loss_recorder_s_t.update_scalar(epoch_loss_s_t / num_iter)

    def _validation(self, epoch):
        auc_s, fpr_s, tpr_s, threshold_s, pre_s, rec_s, f1_s = fcn_evaluate_fn(
            model=self.model_s,
            test_dataloader=self.validation_dataloader,
            meta=self.meta,
            cls=self.cfg_dataset['train']['params']['ccls'],
            device=self.device,
            path=self.save_path_s,
            epoch=epoch)

        auc_t, fpr_t, tpr_t, threshold_t, pre_t, rec_t, f1_t = fcn_evaluate_fn(
            model=self.model_t,
            test_dataloader=self.validation_dataloader,
            meta=self.meta,
            cls=self.cfg_dataset['train']['params']['ccls'],
            device=self.device,
            path=self.save_path_t,
            epoch=epoch)

        self.pre_recorder_s.update_scalar(pre_s)
        self.rec_recorder_s.update_scalar(rec_s)
        self.f1_recorder_s.update_scalar(f1_s)

        self.pre_recorder_t.update_scalar(pre_t)
        self.rec_recorder_t.update_scalar(rec_t)
        self.f1_recorder_t.update_scalar(f1_t)

        if epoch == (self.cfg_trainer['params']['max_iters'] - 1):
            auc_roc_s = dict()
            auc_roc_s['fpr'] = fpr_s
            auc_roc_s['tpr'] = tpr_s
            auc_roc_s['threshold'] = threshold_s
            auc_roc_s['auc'] = auc_s
            np.save(os.path.join(self.save_path_s, 'auc_roc_s.npy'), auc_roc_s)

            auc_roc_t = dict()
            auc_roc_t['fpr'] = fpr_t
            auc_roc_t['tpr'] = tpr_t
            auc_roc_t['threshold'] = threshold_t
            auc_roc_t['auc'] = auc_t
            np.save(os.path.join(self.save_path_t, 'auc_roc.npy'), auc_roc_t)

    def _save_checkpoint(self):
        torch.save(self.model_s.state_dict(), os.path.join(self.save_path_s, 'checkpoint_s.pth'))
        torch.save(self.model_t.state_dict(), os.path.join(self.save_path_t, 'checkpoint_t.pth'))

    def run(self, validation=True):
        self.save_path_s = os.path.join(self.save_path, 'model_s')
        self.save_path_t = os.path.join(self.save_path, 'model_t')
        for param in self.model_t.parameters():
            param.detach_()
        bar = tqdm(list(range(self.cfg_trainer['params']['max_iters'])))
        for i in bar:
            self._train(i)
            if validation:
                self._validation(i)
            else:
                self.pre_recorder_s.update_scalar(0)
                self.rec_recorder_s.update_scalar(0)
                self.f1_recorder_s.update_scalar(0)

                self.pre_recorder_t.update_scalar(0)
                self.rec_recorder_t.update_scalar(0)
                self.f1_recorder_t.update_scalar(0)

            logging_string = "{} epoch, loss {:.4f}, loss_s {:.4f}, p_loss {:.4f}, u_loss {:.4f}, loss_t {:.4f}," \
                             " P_s {:.6f}, R_s {:.6f}, F1_s {:.6f}, P_t {:.6f}, R_t {:.6f}, F1_t {:.6f}".format(
                i + 1,
                self.loss_recorder_s_t.scalar[-1],
                self.loss_recorder_s.scalar[-1],
                self.p_loss_recorder_s.scalar[-1],
                self.u_loss_recorder_s.scalar[-1],
                self.loss_recorder_t.scalar[-1],
                self.pre_recorder_s.scalar[-1],
                self.rec_recorder_s.scalar[-1],
                self.f1_recorder_s.scalar[-1],
                self.pre_recorder_t.scalar[-1],
                self.rec_recorder_t.scalar[-1],
                self.f1_recorder_t.scalar[-1])

            bar.set_description(logging_string)
            logging.info(logging_string)

        self._save_checkpoint()

        self.loss_recorder_s.save_scalar_npy('loss_npy_s', self.save_path_s)
        self.loss_recorder_s.save_lineplot_fig('Loss', 'loss_s', self.save_path_s)
        self.p_loss_recorder_s.save_scalar_npy('p_loss_npy_s', self.save_path_s)
        self.p_loss_recorder_s.save_lineplot_fig('Estimated Positive Loss', 'p_loss_s', self.save_path_s)
        self.u_loss_recorder_s.save_scalar_npy('u_loss_npy_s', self.save_path_s)
        self.u_loss_recorder_s.save_lineplot_fig('Estimated Unlabeled Loss', 'u_loss_s', self.save_path_s)
        self.loss_recorder_t.save_scalar_npy('loss_npy_t', self.save_path_t)
        self.loss_recorder_t.save_lineplot_fig('Loss', 'loss_t', self.save_path_t)
        self.loss_recorder_s_t.save_scalar_npy('loss_npy_s_t', self.save_path)
        self.loss_recorder_s_t.save_lineplot_fig('Loss', 'loss_s_t', self.save_path)
        self.grad_l2_norm_s.save_scalar_npy('grad_l2_norm_npy_s', self.save_path_s)
        self.grad_l2_norm_s.save_lineplot_fig('Grad L2 Norm', 'grad_l2_norm_s', self.save_path_s)
        if validation:
            self.f1_recorder_s.save_scalar_npy('f1_npy_s', self.save_path_s)
            self.f1_recorder_s.save_lineplot_fig('F1-score', 'f1-score_s', self.save_path_s)
            self.rec_recorder_s.save_scalar_npy('recall_npy_s', self.save_path_s)
            self.rec_recorder_s.save_lineplot_fig('Recall', 'recall_s', self.save_path_s)
            self.pre_recorder_s.save_scalar_npy('precision_npy_s', self.save_path_s)
            self.pre_recorder_s.save_lineplot_fig('Precision', 'precision_s', self.save_path_s)

            self.f1_recorder_t.save_scalar_npy('f1_npy_t', self.save_path_t)
            self.f1_recorder_t.save_lineplot_fig('F1-score', 'f1-score_t', self.save_path_t)
            self.rec_recorder_t.save_scalar_npy('recall_npy_t', self.save_path_t)
            self.rec_recorder_t.save_lineplot_fig('Recall', 'recall_t', self.save_path_t)
            self.pre_recorder_t.save_scalar_npy('precision_npy_t', self.save_path_t)
            self.pre_recorder_t.save_lineplot_fig('Precision', 'precision_t', self.save_path_t)
