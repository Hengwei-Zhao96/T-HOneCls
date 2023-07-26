import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from HOC.apis import BaseTrainer
from HOC.apis.validation import fcn_evaluate_fn, mcc_fcn_evaluate_fn, os_mcc_fcn_evaluate_fn
from HOC.datasets import build_dataloader
from HOC.loss_functions import build_loss_function
from HOC.models import build_model
from HOC.optimization import build_optimizer
from HOC.optimization import build_lr_scheduler
from HOC.utils import TRAINER


@TRAINER.register_module()
class SingleModelTrainer(BaseTrainer):
    def __init__(self, trainer, dataset, model, loss_function, optimizer, lr_scheduler, meta):
        super(SingleModelTrainer, self).__init__(trainer=trainer, dataset=dataset, model=model,
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
        return build_loss_function(self.cfg_loss_function)

    def _build_optimizer(self):
        self.cfg_optimizer['params'].update(params=self.model.parameters())
        return build_optimizer(self.cfg_optimizer)

    def _build_lr_scheduler(self):
        self.cfg_lr_scheduler['params'].update(optimizer=self.optimizer)
        return build_lr_scheduler(self.cfg_lr_scheduler)

    def _build_components(self):
        self.train_dataloader = self._build_train_dataloader()
        self.validation_dataloader = self._build_validation_dataloader()
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
        for (data, positive_train_mask, unlabeled_train_mask) in self.train_dataloader:
            data = data.to(self.device)
            positive_train_mask = positive_train_mask.to(self.device)
            unlabeled_train_mask = unlabeled_train_mask.to(self.device)

            target = self.model(data)

            loss, p_loss, u_loss = self.loss_function(target, positive_train_mask, unlabeled_train_mask, epoch,
                                                      self.device)

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
        auc, fpr, tpr, threshold, pre, rec, f1 = fcn_evaluate_fn(
            model=self.model,
            test_dataloader=self.validation_dataloader,
            meta=self.meta,
            cls=self.cfg_dataset['train']['params']['ccls'],
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


@TRAINER.register_module()
class MccSingleModelTrainer(BaseTrainer):
    def __init__(self, trainer, dataset, model, loss_function, optimizer, lr_scheduler, meta):
        super(MccSingleModelTrainer, self).__init__(trainer=trainer, dataset=dataset, model=model,
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
        return build_loss_function(self.cfg_loss_function)

    def _build_optimizer(self):
        self.cfg_optimizer['params'].update(params=self.model.parameters())
        return build_optimizer(self.cfg_optimizer)

    def _build_lr_scheduler(self):
        self.cfg_lr_scheduler['params'].update(optimizer=self.optimizer)
        return build_lr_scheduler(self.cfg_lr_scheduler)

    def _build_components(self):
        self.train_dataloader = self._build_train_dataloader()
        self.validation_dataloader = self._build_validation_dataloader()
        self.model = self._build_model().to(self.device)
        self.loss_function = self._build_loss_function()
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()

    def build_scalar_recoder(self):
        self.loss_recorder = self._build_scalar_recoder()
        self.oa_recorder = self._build_scalar_recoder()
        self.kappa_recorder = self._build_scalar_recoder()

    def _train(self, epoch):
        epoch_loss = 0.0
        num_iter = 0
        self.model.train()
        for (data, mask, weight) in self.train_dataloader:
            data = data.to(self.device)
            mask = mask.to(self.device)
            weight = weight.to(self.device)

            target = self.model(data)

            loss = self.loss_function(target, mask, weight)

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()
            num_iter += 1

        self.lr_scheduler.step()
        self.loss_recorder.update_scalar(epoch_loss / num_iter)

    def _validation(self, epoch):
        acc, acc_per_class, acc_cls, IoU, mean_IoU, kappa = mcc_fcn_evaluate_fn(
            model=self.model,
            test_dataloader=self.validation_dataloader,
            meta=self.meta,
            device=self.device,
            path=self.save_path,
            epoch=epoch)

        self.oa_recorder.update_scalar(acc)
        self.kappa_recorder.update_scalar(kappa)

    def _save_checkpoint(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, 'checkpoint.pth'))

    def run(self, validation=True):
        bar = tqdm(list(range(self.cfg_trainer['params']['max_iters'])))
        for i in bar:
            self._train(i)
            if validation:
                self._validation(i)
            else:
                self.oa_recorder.update_scalar(0)
                self.kappa_recorder.update_scalar(0)

            logging_string = "{} epoch, loss {:.4f}, OA {:.6f}, Kappa {:.6f}".format(
                i + 1,
                self.loss_recorder.scalar[-1],
                self.oa_recorder.scalar[-1],
                self.kappa_recorder.scalar[-1])

            bar.set_description(logging_string)
            logging.info(logging_string)

        self._save_checkpoint()

        self.loss_recorder.save_scalar_npy('loss_npy', self.save_path)
        self.loss_recorder.save_lineplot_fig('Loss', 'loss', self.save_path)
        if validation:
            self.oa_recorder.save_scalar_npy('oa_npy', self.save_path)
            self.oa_recorder.save_lineplot_fig('OA', 'oa', self.save_path)
            self.kappa_recorder.save_scalar_npy('kappa_npy', self.save_path)
            self.kappa_recorder.save_lineplot_fig('Kappa', 'kappa', self.save_path)


@TRAINER.register_module()
class OsMccSingleModelTrainer(BaseTrainer):  ## This Class need to be updated
    def __init__(self, trainer, dataset, model, loss_function, optimizer, lr_scheduler, meta):
        super(OsMccSingleModelTrainer, self).__init__(trainer=trainer, dataset=dataset, model=model,
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
        return build_loss_function(self.cfg_loss_function)

    def _build_optimizer(self):
        self.cfg_optimizer['params'].update(params=self.model.parameters())
        return build_optimizer(self.cfg_optimizer)

    def _build_lr_scheduler(self):
        self.cfg_lr_scheduler['params'].update(optimizer=self.optimizer)
        return build_lr_scheduler(self.cfg_lr_scheduler)

    def _build_components(self):
        self.train_dataloader = self._build_train_dataloader()
        self.validation_dataloader = self._build_validation_dataloader()
        self.model = self._build_model().to(self.device)
        self.loss_function = self._build_loss_function()
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()

    def build_scalar_recoder(self):
        self.loss_recorder = self._build_scalar_recoder()
        self.close_oa_recorder = self._build_scalar_recoder()
        self.close_aa_recorder = self._build_scalar_recoder()
        self.close_kappa_recorder = self._build_scalar_recoder()
        self.open_oa_recorder = self._build_scalar_recoder()
        self.open_aa_recorder = self._build_scalar_recoder()
        self.open_kappa_recorder = self._build_scalar_recoder()
        self.os_pre_recorder = self._build_scalar_recoder()
        self.os_rec_recorder = self._build_scalar_recoder()
        self.os_f1_recorder = self._build_scalar_recoder()
        self.grad_l2_norm = self._build_scalar_recoder()

    def _train(self, epoch):
        epoch_loss = 0.0
        num_iter = 0
        self.model.train()
        for (data, gt, mcc_mask, positive_mask, unlabeled_mask) in self.train_dataloader:
            data = data.to(self.device)
            gt = gt.to(self.device)
            mcc_mask = mcc_mask.to(self.device)
            positive_mask = positive_mask.to(self.device)
            unlabeled_mask = unlabeled_mask.to(self.device)

            target, os_target = self.model(data)

            loss = self.loss_function(target, gt, mcc_mask, os_target, positive_mask, unlabeled_mask)

            self.optimizer.zero_grad()
            loss.backward()

            self.clip_recoder_grad_norm(model=self.model, grad_l2_norm=self.grad_l2_norm)

            self.optimizer.step()

            epoch_loss += loss.item()
            num_iter += 1

        self.lr_scheduler.step()
        self.loss_recorder.update_scalar(epoch_loss / num_iter)

    def _validation(self, epoch):
        close_acc, close_aa, close_kappa, open_acc, open_aa, open_kappa, os_pre, os_rec, os_f1 = os_mcc_fcn_evaluate_fn(
            model=self.model,
            test_dataloader=self.validation_dataloader,
            meta=self.meta,
            device=self.device,
            path=self.save_path,
            epoch=epoch)

        self.close_oa_recorder.update_scalar(close_acc)
        self.close_aa_recorder.update_scalar(close_aa)
        self.close_kappa_recorder.update_scalar(close_kappa)

        self.open_oa_recorder.update_scalar(open_acc)
        self.open_aa_recorder.update_scalar(open_aa)
        self.open_kappa_recorder.update_scalar(open_kappa)

        self.os_pre_recorder.update_scalar(os_pre)
        self.os_rec_recorder.update_scalar(os_rec)
        self.os_f1_recorder.update_scalar(os_f1)

    def _save_checkpoint(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, 'checkpoint.pth'))

    def run(self, validation=True):
        bar = tqdm(list(range(self.cfg_trainer['params']['max_iters'])))
        for i in bar:
            self._train(i)
            if validation:
                self._validation(i)
            else:
                self.close_oa_recorder.update_scalar(0)
                self.close_aa_recorder.update_scalar(0)
                self.close_kappa_recorder.update_scalar(0)

                self.open_oa_recorder.update_scalar(0)
                self.open_aa_recorder.update_scalar(0)
                self.open_kappa_recorder.update_scalar(0)

                self.os_pre_recorder.update_scalar(0)
                self.os_rec_recorder.update_scalar(0)
                self.os_f1_recorder.update_scalar(0)

            logging_string = "{} epoch, loss {:.4f}, Close_OA {:.6f}, Close_AA {:.6f}, Close_Kappa {:.6f}," \
                             " Open_OA {:.6f}, Open_AA {:.6f}, Open_Kappa {:.6f}, Pre {:.6f}," \
                             " Rec {:.6f}, F1 {:.6f}".format(
                i + 1,
                self.loss_recorder.scalar[-1],
                self.close_oa_recorder.scalar[-1],
                self.close_aa_recorder.scalar[-1],
                self.close_kappa_recorder.scalar[-1],
                self.open_oa_recorder.scalar[-1],
                self.open_aa_recorder.scalar[-1],
                self.open_kappa_recorder.scalar[-1],
                self.os_pre_recorder.scalar[-1],
                self.os_rec_recorder.scalar[-1],
                self.os_f1_recorder.scalar[-1])

            bar.set_description(logging_string)
            logging.info(logging_string)

        self._save_checkpoint()

        self.loss_recorder.save_scalar_npy('loss_npy', self.save_path)
        self.loss_recorder.save_lineplot_fig('Loss', 'loss', self.save_path)
        if validation:
            self.close_oa_recorder.save_scalar_npy('close_oa_npy', self.save_path)
            self.close_oa_recorder.save_lineplot_fig('Close_OA', 'close_oa', self.save_path)
            self.close_aa_recorder.save_scalar_npy('close_aa_npy', self.save_path)
            self.close_aa_recorder.save_lineplot_fig('Close_AA', 'close_aa', self.save_path)
            self.close_kappa_recorder.save_scalar_npy('close_kappa_npy', self.save_path)
            self.close_kappa_recorder.save_lineplot_fig('Close_Kappa', 'close_kappa', self.save_path)

            self.open_oa_recorder.save_scalar_npy('open_oa_npy', self.save_path)
            self.open_oa_recorder.save_lineplot_fig('Open_OA', 'open_oa', self.save_path)
            self.open_aa_recorder.save_scalar_npy('open_aa_npy', self.save_path)
            self.open_aa_recorder.save_lineplot_fig('Open_AA', 'open_aa', self.save_path)
            self.open_kappa_recorder.save_scalar_npy('open_kappa_npy', self.save_path)
            self.open_kappa_recorder.save_lineplot_fig('Open_Kappa', 'open_kappa', self.save_path)

            self.os_pre_recorder.save_scalar_npy('os_pre_npy', self.save_path)
            self.os_pre_recorder.save_lineplot_fig('Os_Pre', 'os_pre', self.save_path)
            self.os_rec_recorder.save_scalar_npy('os_rec_npy', self.save_path)
            self.os_rec_recorder.save_lineplot_fig('Os_Rec', 'os_rec', self.save_path)
            self.os_f1_recorder.save_scalar_npy('os_f1_npy', self.save_path)
            self.os_f1_recorder.save_lineplot_fig('Os_F1', 'os_f1', self.save_path)
