import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from HOC.apis import BaseTrainer
from HOC.apis.validation import os_mcc_fcn_evaluate_fn
from HOC.datasets import build_dataloader
from HOC.loss_functions import build_loss_function
from HOC.models import build_model
from HOC.optimization import build_optimizer
from HOC.optimization import build_lr_scheduler
from HOC.apis.toolkit_self_calibration.kl_loss import OsKLLoss
from HOC.apis.toolkit_self_calibration.update_ema_model import update_ema_variables
from HOC.utils import TRAINER


@TRAINER.register_module()
class OsSelfCalibrationTrainer(BaseTrainer):
    def __init__(self, trainer, dataset, model, loss_function, optimizer, lr_scheduler, meta):
        super(OsSelfCalibrationTrainer, self).__init__(trainer=trainer, dataset=dataset, model=model,
                                                       loss_function=loss_function, optimizer=optimizer,
                                                       lr_scheduler=lr_scheduler, meta=meta)
        self.build_scalar_recoder()

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
        self.cfg_lr_scheduler['params'].update(optimizer=self.optimizer)
        return build_lr_scheduler(self.cfg_lr_scheduler)

    def _build_components(self):
        self.train_dataloader = self._build_train_dataloader()
        self.validation_dataloader = self._build_validation_dataloader()
        self.model_s = self._build_model().to(self.device)
        self.model_t = self._build_model().to(self.device)
        self.loss_function = self._build_loss_function()
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()

    def build_scalar_recoder(self):
        self.close_oa_recorder_s = self._build_scalar_recoder()
        self.close_aa_recorder_s = self._build_scalar_recoder()
        self.close_kappa_recorder_s = self._build_scalar_recoder()
        self.open_oa_recorder_s = self._build_scalar_recoder()
        self.open_aa_recorder_s = self._build_scalar_recoder()
        self.open_kappa_recorder_s = self._build_scalar_recoder()
        self.os_pre_recorder_s = self._build_scalar_recoder()
        self.os_rec_recorder_s = self._build_scalar_recoder()
        self.os_f1_recorder_s = self._build_scalar_recoder()
        self.os_auc_recorder_s = self._build_scalar_recoder()

        self.close_oa_recorder_t = self._build_scalar_recoder()
        self.close_aa_recorder_t = self._build_scalar_recoder()
        self.close_kappa_recorder_t = self._build_scalar_recoder()
        self.open_oa_recorder_t = self._build_scalar_recoder()
        self.open_aa_recorder_t = self._build_scalar_recoder()
        self.open_kappa_recorder_t = self._build_scalar_recoder()
        self.os_pre_recorder_t = self._build_scalar_recoder()
        self.os_rec_recorder_t = self._build_scalar_recoder()
        self.os_f1_recorder_t = self._build_scalar_recoder()
        self.os_auc_recorder_t = self._build_scalar_recoder()

        self.loss_recorder = self._build_scalar_recoder()
        self.grad_l2_norm = self._build_scalar_recoder()

    def _train(self, epoch):
        epoch_loss = 0.0
        num_iter = 0
        self.model_s.train()
        for (data, gt, mcc_mask, positive_mask, unlabeled_mask) in self.train_dataloader:
            data = data.to(self.device)
            gt = gt.to(self.device)
            mcc_mask = mcc_mask.to(self.device)
            positive_mask = positive_mask.to(self.device)
            unlabeled_mask = unlabeled_mask.to(self.device)

            target_s, os_target_s = self.model_s(data)
            with torch.no_grad():
                target_t, os_target_t = self.model_t(data)

            loss_s = self.loss_function(target_s, gt, mcc_mask, os_target_s, positive_mask, unlabeled_mask)
            loss_t = OsKLLoss()(os_target_s, os_target_t)

            loss_s_t = loss_s + self.cfg_trainer['params']['beta'] * loss_t

            self.optimizer.zero_grad()
            loss_s_t.backward()

            self.clip_recoder_grad_norm(model=self.model_s, grad_l2_norm=self.grad_l2_norm)

            self.optimizer.step()

            update_ema_variables(model=self.model_s, ema_model=self.model_t,
                                 alpha=self.cfg_trainer['params']['ema_model_alpha'],
                                 global_step=epoch)

            epoch_loss += loss_s_t.item()
            num_iter += 1

        self.lr_scheduler.step()
        self.loss_recorder.update_scalar(epoch_loss / num_iter)

    def _validation(self, epoch):
        close_acc_s, close_aa_s, close_kappa_s, open_acc_s, open_aa_s, open_kappa_s, os_pre_s, os_rec_s, os_f1_s, auc_s, fpr_s, tpr_s, threshold_s = os_mcc_fcn_evaluate_fn(
            model=self.model_s,
            test_dataloader=self.validation_dataloader,
            meta=self.meta,
            device=self.device,
            path=self.save_path_s,
            epoch=epoch)

        close_acc_t, close_aa_t, close_kappa_t, open_acc_t, open_aa_t, open_kappa_t, os_pre_t, os_rec_t, os_f1_t, auc_t, fpr_t, tpr_t, threshold_t = os_mcc_fcn_evaluate_fn(
            model=self.model_t,
            test_dataloader=self.validation_dataloader,
            meta=self.meta,
            device=self.device,
            path=self.save_path_t,
            epoch=epoch)

        self.close_oa_recorder_s.update_scalar(close_acc_s)
        self.close_aa_recorder_s.update_scalar(close_aa_s)
        self.close_kappa_recorder_s.update_scalar(close_kappa_s)
        self.open_oa_recorder_s.update_scalar(open_acc_s)
        self.open_aa_recorder_s.update_scalar(open_aa_s)
        self.open_kappa_recorder_s.update_scalar(open_kappa_s)
        self.os_pre_recorder_s.update_scalar(os_pre_s)
        self.os_rec_recorder_s.update_scalar(os_rec_s)
        self.os_f1_recorder_s.update_scalar(os_f1_s)
        self.os_auc_recorder_s.update_scalar(auc_s)

        self.close_oa_recorder_t.update_scalar(close_acc_t)
        self.close_aa_recorder_t.update_scalar(close_aa_t)
        self.close_kappa_recorder_t.update_scalar(close_kappa_t)
        self.open_oa_recorder_t.update_scalar(open_acc_t)
        self.open_aa_recorder_t.update_scalar(open_aa_t)
        self.open_kappa_recorder_t.update_scalar(open_kappa_t)
        self.os_pre_recorder_t.update_scalar(os_pre_t)
        self.os_rec_recorder_t.update_scalar(os_rec_t)
        self.os_f1_recorder_t.update_scalar(os_f1_t)
        self.os_auc_recorder_t.update_scalar(auc_t)

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
                self.close_oa_recorder_s.update_scalar(0)
                self.close_aa_recorder_s.update_scalar(0)
                self.close_kappa_recorder_s.update_scalar(0)
                self.open_oa_recorder_s.update_scalar(0)
                self.open_aa_recorder_s.update_scalar(0)
                self.open_kappa_recorder_s.update_scalar(0)
                self.os_pre_recorder_s.update_scalar(0)
                self.os_rec_recorder_s.update_scalar(0)
                self.os_f1_recorder_s.update_scalar(0)
                self.os_auc_recorder_s.update_scalar(0)

                self.close_oa_recorder_t.update_scalar(0)
                self.close_aa_recorder_t.update_scalar(0)
                self.close_kappa_recorder_t.update_scalar(0)
                self.open_oa_recorder_t.update_scalar(0)
                self.open_aa_recorder_t.update_scalar(0)
                self.open_kappa_recorder_t.update_scalar(0)
                self.os_pre_recorder_t.update_scalar(0)
                self.os_rec_recorder_t.update_scalar(0)
                self.os_f1_recorder_t.update_scalar(0)
                self.os_auc_recorder_t.update_scalar(0)

            logging_string = "{} epoch, loss {:.4f}, Close_OA_s {:.6f}, Close_AA_s {:.6f}, Close_Kappa_s {:.6f}," \
                             " Open_OA_s {:.6f}, Open_AA_s {:.6f}, Open_Kappa_s {:.6f}, Pre_s {:.6f}," \
                             " Rec_s {:.6f}, F1_s {:.6f}, AUC_s {:.6f}, Close_OA_t {:.6f}, Close_AA_t {:.6f}, Close_Kappa_t {:.6f}," \
                             " Open_OA_t {:.6f}, Open_AA_t {:.6f}, Open_Kappa_t {:.6f}, Pre_t {:.6f}," \
                             " Rec_t {:.6f}, F1_t {:.6f}, AUC_t {:.6f}".format(
                i + 1,
                self.loss_recorder.scalar[-1],
                self.close_oa_recorder_s.scalar[-1],
                self.close_aa_recorder_s.scalar[-1],
                self.close_kappa_recorder_s.scalar[-1],
                self.open_oa_recorder_s.scalar[-1],
                self.open_aa_recorder_s.scalar[-1],
                self.open_kappa_recorder_s.scalar[-1],
                self.os_pre_recorder_s.scalar[-1],
                self.os_rec_recorder_s.scalar[-1],
                self.os_f1_recorder_s.scalar[-1],
                self.os_auc_recorder_s.scalar[-1],
                self.close_oa_recorder_t.scalar[-1],
                self.close_aa_recorder_t.scalar[-1],
                self.close_kappa_recorder_t.scalar[-1],
                self.open_oa_recorder_t.scalar[-1],
                self.open_aa_recorder_t.scalar[-1],
                self.open_kappa_recorder_t.scalar[-1],
                self.os_pre_recorder_t.scalar[-1],
                self.os_rec_recorder_t.scalar[-1],
                self.os_f1_recorder_t.scalar[-1],
                self.os_auc_recorder_t.scalar[-1])

            bar_logging_string = "{} epoch, loss {:.4f}, Close_OA_s {:.6f},  Open_OA_s {:.6f}, F1_s {:.6f}, AUC_s {:.6f}," \
                                 " Close_OA_t {:.6f}, Open_OA_t {:.6f}, F1_t {:.6f}, AUC_t {:.6f},".format(
                i + 1,
                self.loss_recorder.scalar[-1],
                self.close_oa_recorder_s.scalar[-1],
                self.open_oa_recorder_s.scalar[-1],
                self.os_f1_recorder_s.scalar[-1],
                self.os_auc_recorder_s.scalar[-1],
                self.close_oa_recorder_t.scalar[-1],
                self.open_oa_recorder_t.scalar[-1],
                self.os_f1_recorder_t.scalar[-1],
                self.os_auc_recorder_t.scalar[-1])

            bar.set_description(bar_logging_string)
            logging.info(logging_string)

        self._save_checkpoint()

        self.loss_recorder.save_scalar_npy('loss_npy', self.save_path)
        self.loss_recorder.save_lineplot_fig('Loss', 'loss', self.save_path)
        self.grad_l2_norm.save_scalar_npy('grad_l2_norm_npy', self.save_path)
        self.grad_l2_norm.save_lineplot_fig('Grad L2 Norm', 'grad_l2_norm', self.save_path)
        if validation:
            self.close_oa_recorder_s.save_scalar_npy('close_oa_npy', self.save_path_s)
            self.close_oa_recorder_s.save_lineplot_fig('Close_OA', 'close_oa', self.save_path_s)
            self.close_aa_recorder_s.save_scalar_npy('close_aa_npy', self.save_path_s)
            self.close_aa_recorder_s.save_lineplot_fig('Close_AA', 'close_aa', self.save_path_s)
            self.close_kappa_recorder_s.save_scalar_npy('close_kappa_npy', self.save_path_s)
            self.close_kappa_recorder_s.save_lineplot_fig('Close_Kappa', 'close_kappa', self.save_path_s)
            self.open_oa_recorder_s.save_scalar_npy('open_oa_npy', self.save_path_s)
            self.open_oa_recorder_s.save_lineplot_fig('Open_OA', 'open_oa', self.save_path_s)
            self.open_aa_recorder_s.save_scalar_npy('open_aa_npy', self.save_path_s)
            self.open_aa_recorder_s.save_lineplot_fig('Open_AA', 'open_aa', self.save_path_s)
            self.open_kappa_recorder_s.save_scalar_npy('open_kappa_npy', self.save_path_s)
            self.open_kappa_recorder_s.save_lineplot_fig('Open_Kappa', 'open_kappa', self.save_path_s)
            self.os_pre_recorder_s.save_scalar_npy('os_pre_npy', self.save_path_s)
            self.os_pre_recorder_s.save_lineplot_fig('Os_Pre', 'os_pre', self.save_path_s)
            self.os_rec_recorder_s.save_scalar_npy('os_rec_npy', self.save_path_s)
            self.os_rec_recorder_s.save_lineplot_fig('Os_Rec', 'os_rec', self.save_path_s)
            self.os_f1_recorder_s.save_scalar_npy('os_f1_npy', self.save_path_s)
            self.os_f1_recorder_s.save_lineplot_fig('Os_F1', 'os_f1', self.save_path_s)
            self.os_auc_recorder_s.save_scalar_npy('os_auc_npy', self.save_path_s)
            self.os_auc_recorder_s.save_lineplot_fig('Os_AUC', 'os_auc', self.save_path_s)

            self.close_oa_recorder_t.save_scalar_npy('close_oa_npy', self.save_path_t)
            self.close_oa_recorder_t.save_lineplot_fig('Close_OA', 'close_oa', self.save_path_t)
            self.close_aa_recorder_t.save_scalar_npy('close_aa_npy', self.save_path_t)
            self.close_aa_recorder_t.save_lineplot_fig('Close_AA', 'close_aa', self.save_path_t)
            self.close_kappa_recorder_t.save_scalar_npy('close_kappa_npy', self.save_path_t)
            self.close_kappa_recorder_t.save_lineplot_fig('Close_Kappa', 'close_kappa', self.save_path_t)
            self.open_oa_recorder_t.save_scalar_npy('open_oa_npy', self.save_path_t)
            self.open_oa_recorder_t.save_lineplot_fig('Open_OA', 'open_oa', self.save_path_t)
            self.open_aa_recorder_t.save_scalar_npy('open_aa_npy', self.save_path_t)
            self.open_aa_recorder_t.save_lineplot_fig('Open_AA', 'open_aa', self.save_path_t)
            self.open_kappa_recorder_t.save_scalar_npy('open_kappa_npy', self.save_path_t)
            self.open_kappa_recorder_t.save_lineplot_fig('Open_Kappa', 'open_kappa', self.save_path_t)
            self.os_pre_recorder_t.save_scalar_npy('os_pre_npy', self.save_path_t)
            self.os_pre_recorder_t.save_lineplot_fig('Os_Pre', 'os_pre', self.save_path_t)
            self.os_rec_recorder_t.save_scalar_npy('os_rec_npy', self.save_path_t)
            self.os_rec_recorder_t.save_lineplot_fig('Os_Rec', 'os_rec', self.save_path_t)
            self.os_f1_recorder_t.save_scalar_npy('os_f1_npy', self.save_path_t)
            self.os_f1_recorder_t.save_lineplot_fig('Os_F1', 'os_f1', self.save_path_t)
            self.os_auc_recorder_t.save_scalar_npy('os_auc_npy', self.save_path_t)
            self.os_auc_recorder_t.save_lineplot_fig('Os_AUC', 'os_auc', self.save_path_t)
