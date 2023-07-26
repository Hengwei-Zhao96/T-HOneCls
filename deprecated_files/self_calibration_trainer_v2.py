import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from HOC.apis import BaseTrainer
from HOC.apis.validation import fcn_evaluate_fn, fusion_fcn_evaluate_fn
from HOC.datasets import build_dataloader
from HOC.loss_functions import build_loss_function
from HOC.models import build_model
from HOC.optimization import build_optimizer
from HOC.optimization import build_lr_scheduler
from HOC.apis.toolkit_self_calibration.l2_loss import L2Loss
from HOC.apis.toolkit_self_calibration.kl_loss import KLLoss
from HOC.apis.toolkit_self_calibration.update_ema_model import update_ema_variables
from HOC.apis.toolkit_self_calibration.update_ema_label import update_ema_labels
from HOC.utils import TRAINER


@TRAINER.register_module()
class SelfCalibrationTrainerV2(BaseTrainer):
    def __init__(self, trainer, dataset, model, loss_function, optimizer, lr_scheduler, meta):
        super(SelfCalibrationTrainerV2, self).__init__(trainer=trainer, dataset=dataset, model=model,
                                                       loss_function=loss_function,
                                                       optimizer=optimizer, lr_scheduler=lr_scheduler, meta=meta)
        self.build_scalar_recoder()
        self.ema_label_s1 = None
        self.ema_label_s2 = None

    def _build_train_dataloader(self):
        return build_dataloader(self.cfg_dataset['train'])

    def _build_validation_dataloader(self):
        return build_dataloader(self.cfg_dataset['test'])

    def _build_model(self):
        return build_model(self.cfg_model)

    def _build_loss_function(self):
        return build_loss_function(self.cfg_loss_function)

    def _build_optimizer(self):
        self.cfg_optimizer['params'].update(
            params=[{'params': self.model_s1.parameters()}, {'params': self.model_s2.parameters()}])
        return build_optimizer(self.cfg_optimizer)

    def _build_lr_scheduler(self):
        self.cfg_lr_scheduler['params'].update(optimizer=self.optimizer)
        return build_lr_scheduler(self.cfg_lr_scheduler)

    def _build_components(self):
        self.train_dataloader = self._build_train_dataloader()
        self.validation_dataloader = self._build_validation_dataloader()
        self.model_s1 = self._build_model().to(self.device)
        self.model_t1 = self._build_model().to(self.device)
        self.model_s2 = self._build_model().to(self.device)
        self.model_t2 = self._build_model().to(self.device)
        self.loss_function = self._build_loss_function()
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()

    def build_scalar_recoder(self):
        self.f1_recorder_s1 = self._build_scalar_recoder()
        self.pre_recorder_s1 = self._build_scalar_recoder()
        self.rec_recorder_s1 = self._build_scalar_recoder()
        self.loss_recorder_s1 = self._build_scalar_recoder()
        self.p_loss_recorder_s1 = self._build_scalar_recoder()
        self.u_loss_recorder_s1 = self._build_scalar_recoder()
        self.grad_l2_norm_s1 = self._build_scalar_recoder()

        self.f1_recorder_s2 = self._build_scalar_recoder()
        self.pre_recorder_s2 = self._build_scalar_recoder()
        self.rec_recorder_s2 = self._build_scalar_recoder()
        self.loss_recorder_s2 = self._build_scalar_recoder()
        self.p_loss_recorder_s2 = self._build_scalar_recoder()
        self.u_loss_recorder_s2 = self._build_scalar_recoder()
        self.grad_l2_norm_s2 = self._build_scalar_recoder()

        self.f1_recorder_t1 = self._build_scalar_recoder()
        self.pre_recorder_t1 = self._build_scalar_recoder()
        self.rec_recorder_t1 = self._build_scalar_recoder()
        self.loss_recorder_s1_t1 = self._build_scalar_recoder()

        self.f1_recorder_t2 = self._build_scalar_recoder()
        self.pre_recorder_t2 = self._build_scalar_recoder()
        self.rec_recorder_t2 = self._build_scalar_recoder()
        self.loss_recorder_s2_t2 = self._build_scalar_recoder()

        self.f1_recorder_t1_t2 = self._build_scalar_recoder()
        self.pre_recorder_t1_t2 = self._build_scalar_recoder()
        self.rec_recorder_t1_t2 = self._build_scalar_recoder()

        self.loss_recorder_s1_s2 = self._build_scalar_recoder()
        self.loss_recorder_all = self._build_scalar_recoder()

    def _train(self, epoch):
        epoch_loss_s1 = 0.0
        epoch_p_loss_s1 = 0.0
        epoch_u_loss_s1 = 0.0
        epoch_loss_s1_t1 = 0.0
        epoch_loss_s2 = 0.0
        epoch_p_loss_s2 = 0.0
        epoch_u_loss_s2 = 0.0
        epoch_loss_s2_t2 = 0.0
        epoch_loss_s1_s2 = 0.0
        epoch_loss_all = 0.0

        num_iter = 0
        self.model_s1.train()
        self.model_s2.train()
        for (data, positive_train_mask, unlabeled_train_mask) in self.train_dataloader:
            data = data.to(self.device)
            positive_train_mask = positive_train_mask.to(self.device)
            unlabeled_train_mask = unlabeled_train_mask.to(self.device)

            target_s1 = self.model_s1(data)
            target_s2 = self.model_s2(data)
            with torch.no_grad():
                target_t1 = self.model_t1(data)
                target_t2 = self.model_t2(data)

            # if self.cfg_trainer['params']['ema_label']:
            #     self.ema_label_s1 = update_ema_labels(unlabeled_mask=unlabeled_train_mask,
            #                                           ema_label=self.ema_label_s1,
            #                                           target_t=target_t1,
            #                                           alpha=self.cfg_trainer['params']['ema_label_alpha'],
            #                                           warm_up_epoch=self.cfg_trainer['params'][
            #                                               'ema_label_warm_up_epoch'],
            #                                           epoch=epoch)
            #     self.ema_label_s2 = update_ema_labels(unlabeled_mask=unlabeled_train_mask,
            #                                           ema_label=self.ema_label_s2,
            #                                           target_t=target_t2,
            #                                           alpha=self.cfg_trainer['params']['ema_label_alpha'],
            #                                           warm_up_epoch=self.cfg_trainer['params'][
            #                                               'ema_label_warm_up_epoch'],
            #                                           epoch=epoch)
            #
            #     loss_s1, p_loss_s1, u_loss_s1 = self.loss_function(target_s1, positive_train_mask, unlabeled_train_mask,
            #                                                        self.ema_label_s1, epoch, self.device)
            #     loss_s2, p_loss_s2, u_loss_s2 = self.loss_function(target_s2, positive_train_mask, unlabeled_train_mask,
            #                                                        self.ema_label_s2, epoch, self.device)
            # else:
            loss_s1, p_loss_s1, u_loss_s1 = self.loss_function(target_s1, positive_train_mask, unlabeled_train_mask,
                                                                   epoch, self.device)
            loss_s2, p_loss_s2, u_loss_s2 = self.loss_function(target_s2, positive_train_mask, unlabeled_train_mask,
                                                                   epoch, self.device)

            loss_s1_t1 = L2Loss(equal_weight=False)(target_s1, target_t2, positive_train_mask, unlabeled_train_mask)
            loss_s2_t2 = L2Loss(equal_weight=False)(target_s2, target_t1, positive_train_mask, unlabeled_train_mask)

            loss_s1_s2 = KLLoss(equal_weight=False)(target_s1, target_s2, positive_train_mask, unlabeled_train_mask)

            loss_all = (loss_s1 + loss_s2) + (loss_s1_t1 + loss_s2_t2) + self.cfg_trainer['params']['beta'] * loss_s1_s2

            self.optimizer.zero_grad()
            loss_all.backward()

            self.clip_recoder_grad_norm(model=self.model_s1, grad_l2_norm=self.grad_l2_norm_s1)
            self.clip_recoder_grad_norm(model=self.model_s2, grad_l2_norm=self.grad_l2_norm_s2)

            self.optimizer.step()

            update_ema_variables(model=self.model_s1, ema_model=self.model_t1,
                                 alpha=self.cfg_trainer['params']['ema_model_alpha'],
                                 global_step=epoch)
            update_ema_variables(model=self.model_s2, ema_model=self.model_t2,
                                 alpha=self.cfg_trainer['params']['ema_model_alpha'],
                                 global_step=epoch)

            epoch_loss_s1 += loss_s1.item()
            epoch_p_loss_s1 += p_loss_s1.item()
            epoch_u_loss_s1 += u_loss_s1.item()
            epoch_loss_s1_t1 += loss_s1_t1.item()
            epoch_loss_s2 += loss_s2.item()
            epoch_p_loss_s2 += p_loss_s2.item()
            epoch_u_loss_s2 += u_loss_s2.item()
            epoch_loss_s2_t2 += loss_s2_t2.item()
            epoch_loss_s1_s2 += loss_s1_s2.item()
            epoch_loss_all += loss_all.item()
            num_iter += 1

        self.lr_scheduler.step()

        self.loss_recorder_s1.update_scalar(epoch_loss_s1 / num_iter)
        self.p_loss_recorder_s1.update_scalar(epoch_p_loss_s1 / num_iter)
        self.u_loss_recorder_s1.update_scalar(epoch_u_loss_s1 / num_iter)
        self.loss_recorder_s1_t1.update_scalar(epoch_loss_s1_t1 / num_iter)
        self.loss_recorder_s2.update_scalar(epoch_loss_s2 / num_iter)
        self.p_loss_recorder_s2.update_scalar(epoch_p_loss_s2 / num_iter)
        self.u_loss_recorder_s2.update_scalar(epoch_u_loss_s2 / num_iter)
        self.loss_recorder_s2_t2.update_scalar(epoch_loss_s2_t2 / num_iter)
        self.loss_recorder_s1_s2.update_scalar(epoch_loss_s1_s2 / num_iter)
        self.loss_recorder_all.update_scalar(epoch_loss_all / num_iter)

    def _validation(self, epoch):
        auc_s1, fpr_s1, tpr_s1, threshold_s1, pre_s1, rec_s1, f1_s1 = fcn_evaluate_fn(
            model=self.model_s1,
            test_dataloader=self.validation_dataloader,
            meta=self.meta,
            cls=self.cfg_dataset['train']['params']['ccls'],
            device=self.device,
            path=self.save_path_s1,
            epoch=epoch)

        # auc_s1, fpr_s1, tpr_s1, threshold_s1, pre_s1, rec_s1, f1_s1 = 0, 0, 0, 0, 0, 0, 0

        auc_s2, fpr_s2, tpr_s2, threshold_s2, pre_s2, rec_s2, f1_s2 = fcn_evaluate_fn(
            model=self.model_s2,
            test_dataloader=self.validation_dataloader,
            meta=self.meta,
            cls=self.cfg_dataset['train']['params']['ccls'],
            device=self.device,
            path=self.save_path_s2,
            epoch=epoch)
        # auc_s2, fpr_s2, tpr_s2, threshold_s2, pre_s2, rec_s2, f1_s2 = 0, 0, 0, 0, 0, 0, 0

        auc_t1, fpr_t1, tpr_t1, threshold_t1, pre_t1, rec_t1, f1_t1 = fcn_evaluate_fn(
            model=self.model_t1,
            test_dataloader=self.validation_dataloader,
            meta=self.meta,
            cls=self.cfg_dataset['train']['params']['ccls'],
            device=self.device,
            path=self.save_path_t1,
            epoch=epoch)
        # auc_t1, fpr_t1, tpr_t1, threshold_t1, pre_t1, rec_t1, f1_t1 = 0, 0, 0, 0, 0, 0, 0

        auc_t2, fpr_t2, tpr_t2, threshold_t2, pre_t2, rec_t2, f1_t2 = fcn_evaluate_fn(
            model=self.model_t2,
            test_dataloader=self.validation_dataloader,
            meta=self.meta,
            cls=self.cfg_dataset['train']['params']['ccls'],
            device=self.device,
            path=self.save_path_t2,
            epoch=epoch)

        # auc_t2, fpr_t2, tpr_t2, threshold_t2, pre_t2, rec_t2, f1_t2 = 0, 0, 0, 0, 0, 0, 0

        auc_t1_t2, fpr_t1_t2, tpr_t1_t2, threshold_t1_t2, pre_t1_t2, rec_t1_t2, f1_t1_t2 = fusion_fcn_evaluate_fn(
            model=[self.model_t1, self.model_t2],
            test_dataloader=self.validation_dataloader,
            meta=self.meta,
            cls=self.cfg_dataset['train']['params']['ccls'],
            device=self.device,
            path=self.save_path_t1_t2,
            epoch=epoch)

        self.pre_recorder_s1.update_scalar(pre_s1)
        self.rec_recorder_s1.update_scalar(rec_s1)
        self.f1_recorder_s1.update_scalar(f1_s1)

        self.pre_recorder_s2.update_scalar(pre_s2)
        self.rec_recorder_s2.update_scalar(rec_s2)
        self.f1_recorder_s2.update_scalar(f1_s2)

        self.pre_recorder_t1.update_scalar(pre_t1)
        self.rec_recorder_t1.update_scalar(rec_t1)
        self.f1_recorder_t1.update_scalar(f1_t1)

        self.pre_recorder_t2.update_scalar(pre_t2)
        self.rec_recorder_t2.update_scalar(rec_t2)
        self.f1_recorder_t2.update_scalar(f1_t2)

        self.pre_recorder_t1_t2.update_scalar(pre_t1_t2)
        self.rec_recorder_t1_t2.update_scalar(rec_t1_t2)
        self.f1_recorder_t1_t2.update_scalar(f1_t1_t2)

        if epoch == (self.cfg_trainer['params']['max_iters'] - 1):
            auc_roc_s1 = dict()
            auc_roc_s1['fpr'] = fpr_s1
            auc_roc_s1['tpr'] = tpr_s1
            auc_roc_s1['threshold'] = threshold_s1
            auc_roc_s1['auc'] = auc_s1
            np.save(os.path.join(self.save_path_s1, 'auc_roc_s1.npy'), auc_roc_s1)

            auc_roc_s2 = dict()
            auc_roc_s2['fpr'] = fpr_s2
            auc_roc_s2['tpr'] = tpr_s2
            auc_roc_s2['threshold'] = threshold_s2
            auc_roc_s2['auc'] = auc_s2
            np.save(os.path.join(self.save_path_s2, 'auc_roc_s2.npy'), auc_roc_s2)

            auc_roc_t1 = dict()
            auc_roc_t1['fpr'] = fpr_t1
            auc_roc_t1['tpr'] = tpr_t1
            auc_roc_t1['threshold'] = threshold_t1
            auc_roc_t1['auc'] = auc_t1
            np.save(os.path.join(self.save_path_t1, 'auc_roc_t1.npy'), auc_roc_t1)

            auc_roc_t2 = dict()
            auc_roc_t2['fpr'] = fpr_t2
            auc_roc_t2['tpr'] = tpr_t2
            auc_roc_t2['threshold'] = threshold_t2
            auc_roc_t2['auc'] = auc_t2
            np.save(os.path.join(self.save_path_t2, 'auc_roc_t2.npy'), auc_roc_t2)

            auc_roc_t1_t2 = dict()
            auc_roc_t1_t2['fpr'] = fpr_t1_t2
            auc_roc_t1_t2['tpr'] = tpr_t1_t2
            auc_roc_t1_t2['threshold'] = threshold_t1_t2
            auc_roc_t1_t2['auc'] = auc_t1_t2
            np.save(os.path.join(self.save_path_t1_t2, 'auc_roc_t1_t2.npy'), auc_roc_t1_t2)

    def _save_checkpoint(self):
        torch.save(self.model_s1.state_dict(), os.path.join(self.save_path_s1, 'checkpoint_s1.pth'))
        torch.save(self.model_t1.state_dict(), os.path.join(self.save_path_t1, 'checkpoint_t1.pth'))
        torch.save(self.model_s2.state_dict(), os.path.join(self.save_path_s2, 'checkpoint_s2.pth'))
        torch.save(self.model_t2.state_dict(), os.path.join(self.save_path_t2, 'checkpoint_t2.pth'))

    def run(self, validation=True):
        self.save_path_s1 = os.path.join(self.save_path, 'model_s1')
        self.save_path_s2 = os.path.join(self.save_path, 'model_s2')
        self.save_path_t1 = os.path.join(self.save_path, 'model_t1')
        self.save_path_t2 = os.path.join(self.save_path, 'model_t2')
        self.save_path_t1_t2 = os.path.join(self.save_path, 'model_t1_t2')

        for param1 in self.model_t1.parameters():
            param1.detach_()
        for param2 in self.model_t2.parameters():
            param2.detach_()

        bar = tqdm(list(range(self.cfg_trainer['params']['max_iters'])))

        for i in bar:
            self._train(i)
            if validation:
                self._validation(i)
            else:
                self.pre_recorder_s1.update_scalar(0)
                self.rec_recorder_s1.update_scalar(0)
                self.f1_recorder_s1.update_scalar(0)

                self.pre_recorder_s2.update_scalar(0)
                self.rec_recorder_s2.update_scalar(0)
                self.f1_recorder_s2.update_scalar(0)

                self.pre_recorder_t1.update_scalar(0)
                self.rec_recorder_t1.update_scalar(0)
                self.f1_recorder_t1.update_scalar(0)

                self.pre_recorder_t2.update_scalar(0)
                self.rec_recorder_t2.update_scalar(0)
                self.f1_recorder_t2.update_scalar(0)

                self.pre_recorder_t1_t2.update_scalar(0)
                self.rec_recorder_t1_t2.update_scalar(0)
                self.f1_recorder_t1_t2.update_scalar(0)

            logging_string = "{} epoch, loss_all {:.4f}," \
                             " loss_s1 {:.4f}, p_loss1 {:.4f}, u_loss1 {:.4f}, loss_s1_t1 {:.4f}," \
                             " loss_s2 {:.4f}, p_loss2 {:.4f}, u_loss2 {:.4f}, loss_s2_t2 {:.4f}," \
                             " loss_s1_s2 {:.4f}," \
                             " P_s1 {:.6f}, R_s1 {:.6f}, F1_s1 {:.6f}, P_t1 {:.6f}, R_t1 {:.6f}, F1_t1 {:.6f}," \
                             " P_s2 {:.6f}, R_s2 {:.6f}, F1_s2 {:.6f}, P_t2 {:.6f}, R_t2 {:.6f}, F1_t2 {:.6f}," \
                             " P_t1_t2 {:.6f}, R_t1_t2 {:.6f}, F1_t1_t2 {:.6f}".format(
                i + 1,
                self.loss_recorder_all.scalar[-1],
                self.loss_recorder_s1.scalar[-1],
                self.p_loss_recorder_s1.scalar[-1],
                self.u_loss_recorder_s1.scalar[-1],
                self.loss_recorder_s1_t1.scalar[-1],
                self.loss_recorder_s2.scalar[-1],
                self.p_loss_recorder_s2.scalar[-1],
                self.u_loss_recorder_s2.scalar[-1],
                self.loss_recorder_s2_t2.scalar[-1],
                self.loss_recorder_s1_s2.scalar[-1],
                self.pre_recorder_s1.scalar[-1],
                self.rec_recorder_s1.scalar[-1],
                self.f1_recorder_s1.scalar[-1],
                self.pre_recorder_t1.scalar[-1],
                self.rec_recorder_t1.scalar[-1],
                self.f1_recorder_t1.scalar[-1],
                self.pre_recorder_s2.scalar[-1],
                self.rec_recorder_s2.scalar[-1],
                self.f1_recorder_s2.scalar[-1],
                self.pre_recorder_t2.scalar[-1],
                self.rec_recorder_t2.scalar[-1],
                self.f1_recorder_t2.scalar[-1],
                self.pre_recorder_t1_t2.scalar[-1],
                self.rec_recorder_t1_t2.scalar[-1],
                self.f1_recorder_t1_t2.scalar[-1]
            )

            bar_string = "{} epoch, " \
                         "P_s1 {:.6f}, R_s1 {:.6f}, F1_s1 {:.6f}, P_t1 {:.6f}, R_t1 {:.6f}, F1_t1 {:.6f}, " \
                         "P_s2 {:.6f}, R_s2 {:.6f}, F1_s2 {:.6f}, P_t2 {:.6f}, R_t2 {:.6f}, F1_t2 {:.6f}, " \
                         "P_t1_t2 {:.6f}, R_t1_t2 {:.6f}, F1_t1_t2 {:.6f}".format(
                i + 1,
                self.pre_recorder_s1.scalar[-1],
                self.rec_recorder_s1.scalar[-1],
                self.f1_recorder_s1.scalar[-1],
                self.pre_recorder_t1.scalar[-1],
                self.rec_recorder_t1.scalar[-1],
                self.f1_recorder_t1.scalar[-1],
                self.pre_recorder_s2.scalar[-1],
                self.rec_recorder_s2.scalar[-1],
                self.f1_recorder_s2.scalar[-1],
                self.pre_recorder_t2.scalar[-1],
                self.rec_recorder_t2.scalar[-1],
                self.f1_recorder_t2.scalar[-1],
                self.pre_recorder_t1_t2.scalar[-1],
                self.rec_recorder_t1_t2.scalar[-1],
                self.f1_recorder_t1_t2.scalar[-1]
            )

            bar.set_description(bar_string)
            logging.info(logging_string)

        self._save_checkpoint()

        self.loss_recorder_s1.save_scalar_npy('loss_npy_s1', self.save_path_s1)
        self.loss_recorder_s1.save_lineplot_fig('Loss', 'loss_s1', self.save_path_s1)
        self.p_loss_recorder_s1.save_scalar_npy('p_loss_npy_s1', self.save_path_s1)
        self.p_loss_recorder_s1.save_lineplot_fig('Estimated Positive Loss', 'p_loss_s1', self.save_path_s1)
        self.u_loss_recorder_s1.save_scalar_npy('u_loss_npy_s1', self.save_path_s1)
        self.u_loss_recorder_s1.save_lineplot_fig('Estimated Unlabeled Loss', 'u_loss_s1', self.save_path_s1)
        self.loss_recorder_s1_t1.save_scalar_npy('loss_npy_s1_t1', self.save_path_t1)
        self.loss_recorder_s1_t1.save_lineplot_fig('Loss', 'loss_s1_t1', self.save_path_t1)
        self.grad_l2_norm_s1.save_scalar_npy('grad_l2_norm_npy_s1', self.save_path_s1)
        self.grad_l2_norm_s1.save_lineplot_fig('Grad L2 Norm', 'grad_l2_norm_s1', self.save_path_s1)

        self.loss_recorder_s2.save_scalar_npy('loss_npy_s2', self.save_path_s2)
        self.loss_recorder_s2.save_lineplot_fig('Loss', 'loss_s2', self.save_path_s2)
        self.p_loss_recorder_s2.save_scalar_npy('p_loss_npy_s2', self.save_path_s2)
        self.p_loss_recorder_s2.save_lineplot_fig('Estimated Positive Loss', 'p_loss_s2', self.save_path_s2)
        self.u_loss_recorder_s2.save_scalar_npy('u_loss_npy_s2', self.save_path_s2)
        self.u_loss_recorder_s2.save_lineplot_fig('Estimated Unlabeled Loss', 'u_loss_s2', self.save_path_s2)
        self.loss_recorder_s2_t2.save_scalar_npy('loss_npy_s2_t2', self.save_path_t2)
        self.loss_recorder_s2_t2.save_lineplot_fig('Loss', 'loss_s2_t2', self.save_path_t2)
        self.grad_l2_norm_s2.save_scalar_npy('grad_l2_norm_npy_s2', self.save_path_s2)
        self.grad_l2_norm_s2.save_lineplot_fig('Grad L2 Norm', 'grad_l2_norm_s2', self.save_path_s2)

        self.loss_recorder_s1_s2.save_scalar_npy('loss_npy_s1_s2', self.save_path)
        self.loss_recorder_s1_s2.save_lineplot_fig('Loss', 'loss_s1_s2', self.save_path)
        self.loss_recorder_all.save_scalar_npy('loss_npy_all', self.save_path)
        self.loss_recorder_all.save_lineplot_fig('Loss', 'loss_all', self.save_path)

        if validation:
            self.f1_recorder_s1.save_scalar_npy('f1_npy_s1', self.save_path_s1)
            self.f1_recorder_s1.save_lineplot_fig('F1-score', 'f1-score_s1', self.save_path_s1)
            self.rec_recorder_s1.save_scalar_npy('recall_npy_s1', self.save_path_s1)
            self.rec_recorder_s1.save_lineplot_fig('Recall', 'recall_s1', self.save_path_s1)
            self.pre_recorder_s1.save_scalar_npy('precision_npy_s1', self.save_path_s1)
            self.pre_recorder_s1.save_lineplot_fig('Precision', 'precision_s1', self.save_path_s1)

            self.f1_recorder_t1.save_scalar_npy('f1_npy_t1', self.save_path_t1)
            self.f1_recorder_t1.save_lineplot_fig('F1-score', 'f1-score_t1', self.save_path_t1)
            self.rec_recorder_t1.save_scalar_npy('recall_npy_t1', self.save_path_t1)
            self.rec_recorder_t1.save_lineplot_fig('Recall', 'recall_t1', self.save_path_t1)
            self.pre_recorder_t1.save_scalar_npy('precision_npy_t1', self.save_path_t1)
            self.pre_recorder_t1.save_lineplot_fig('Precision', 'precision_t1', self.save_path_t1)

            self.f1_recorder_s2.save_scalar_npy('f1_npy_s2', self.save_path_s2)
            self.f1_recorder_s2.save_lineplot_fig('F1-score', 'f1-score_s2', self.save_path_s2)
            self.rec_recorder_s2.save_scalar_npy('recall_npy_s2', self.save_path_s2)
            self.rec_recorder_s2.save_lineplot_fig('Recall', 'recall_s2', self.save_path_s2)
            self.pre_recorder_s2.save_scalar_npy('precision_npy_s2', self.save_path_s2)
            self.pre_recorder_s2.save_lineplot_fig('Precision', 'precision_s2', self.save_path_s2)

            self.f1_recorder_t2.save_scalar_npy('f1_npy_t2', self.save_path_t2)
            self.f1_recorder_t2.save_lineplot_fig('F1-score', 'f1-score_t2', self.save_path_t2)
            self.rec_recorder_t2.save_scalar_npy('recall_npy_t2', self.save_path_t2)
            self.rec_recorder_t2.save_lineplot_fig('Recall', 'recall_t2', self.save_path_t2)
            self.pre_recorder_t2.save_scalar_npy('precision_npy_t2', self.save_path_t2)
            self.pre_recorder_t2.save_lineplot_fig('Precision', 'precision_t2', self.save_path_t2)

            self.f1_recorder_t1_t2.save_scalar_npy('f1_npy_t1_t2', self.save_path_t1_t2)
            self.f1_recorder_t1_t2.save_lineplot_fig('F1-score', 'f1-score_t1_t2', self.save_path_t1_t2)
            self.rec_recorder_t1_t2.save_scalar_npy('recall_npy_t1_t2', self.save_path_t1_t2)
            self.rec_recorder_t1_t2.save_lineplot_fig('Recall', 'recall_t1_t2', self.save_path_t1_t2)
            self.pre_recorder_t1_t2.save_scalar_npy('precision_npy_t1_t2', self.save_path_t1_t2)
            self.pre_recorder_t1_t2.save_lineplot_fig('Precision', 'precision_t1_t2', self.save_path_t1_t2)
