import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from HOC.apis import BaseTrainer
from HOC.apis.validation import fcn_evaluate_fn, fusion_fcn_evaluate_fn
from HOC.apis.toolkit_noisy_label_learning import get_small_loss_unlabeled_samples
from HOC.datasets import build_dataloader
from HOC.loss_functions import BCELossPf, ConKLLossPf
from HOC.models import build_model
from HOC.optimization import build_optimizer
from HOC.optimization import build_lr_scheduler
from HOC.utils import TRAINER


@TRAINER.register_module()
class JoCoRTrainer(BaseTrainer):
    def __init__(self, trainer, dataset, model, loss_function, optimizer, lr_scheduler, meta):
        super(JoCoRTrainer, self).__init__(trainer=trainer, dataset=dataset, model=model,
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
        self.cfg_optimizer['params'].update(
            params=[{'params': self.model1.parameters()}, {'params': self.model2.parameters()}])
        return build_optimizer(self.cfg_optimizer)

    def _build_lr_scheduler(self):
        self.cfg_lr_scheduler['params'].update(optimizer=self.optimizer)
        return build_lr_scheduler(self.cfg_lr_scheduler)

    def _build_components(self):
        self.train_dataloader = self._build_train_dataloader()
        self.validation_dataloader = self._build_validation_dataloader()
        self.model1 = self._build_model().to(self.device)
        self.model2 = self._build_model().to(self.device)
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()

    def build_scalar_recoder(self):
        self.f1_recorder_1 = self._build_scalar_recoder()
        self.pre_recorder_1 = self._build_scalar_recoder()
        self.rec_recorder_1 = self._build_scalar_recoder()
        self.grad_l2_norm_1 = self._build_scalar_recoder()

        self.f1_recorder_2 = self._build_scalar_recoder()
        self.pre_recorder_2 = self._build_scalar_recoder()
        self.rec_recorder_2 = self._build_scalar_recoder()
        self.grad_l2_norm_2 = self._build_scalar_recoder()

        self.f1_recorder_f = self._build_scalar_recoder()
        self.pre_recorder_f = self._build_scalar_recoder()
        self.rec_recorder_f = self._build_scalar_recoder()

        self.loss_recorder = self._build_scalar_recoder()
        self.p_loss_recorder = self._build_scalar_recoder()
        self.u_loss_recorder = self._build_scalar_recoder()

    def _train(self, epoch):
        epoch_loss = 0.0
        epoch_p_loss = 0.0
        epoch_u_loss = 0.0

        num_iter = 0

        for (data, positive_train_mask, unlabeled_train_mask) in self.train_dataloader:
            data = data.to(self.device)
            positive_train_mask = positive_train_mask.to(self.device)
            unlabeled_train_mask = unlabeled_train_mask.to(self.device)

            self.model1.eval()
            self.model2.eval()

            eval_target_1 = self.model1(data)
            eval_target_2 = self.model2(data)

            ratio = min(epoch / self.cfg_trainer['params']['Tk'] * self.cfg_trainer['params']['class_prior'],
                        self.cfg_trainer['params']['class_prior'])

            _, _, _, bce_loss_matrix1 = BCELossPf(equal_weight=True, reduction=False)(eval_target_1,
                                                                                      positive_train_mask,
                                                                                      unlabeled_train_mask, epoch,
                                                                                      self.device)

            _, _, _, bce_loss_matrix2 = BCELossPf(equal_weight=True, reduction=False)(eval_target_2,
                                                                                      positive_train_mask,
                                                                                      unlabeled_train_mask, epoch,
                                                                                      self.device)
            _, _, _, conkl_loss_matrix = ConKLLossPf(equal_weight=True, reduction=False)(eval_target_1, eval_target_2,
                                                                                         positive_train_mask,
                                                                                         unlabeled_train_mask, epoch,
                                                                                         self.device)

            loss_matrix = (1 - self.cfg_trainer['params']['lam']) * (bce_loss_matrix1 + bce_loss_matrix2) + \
                          self.cfg_trainer['params']['lam'] * conkl_loss_matrix

            pse_negative_label = get_small_loss_unlabeled_samples(unlabeled_train_mask=unlabeled_train_mask,
                                                                  loss_mask=loss_matrix, ratio=ratio)

            self.model1.train()
            self.model2.train()

            target_1 = self.model1(data)
            bce_loss_1, bce_p_loss_1, bce_u_loss_1 = BCELossPf(equal_weight=True)(target_1, positive_train_mask,
                                                                                  pse_negative_label, epoch,
                                                                                  self.device)

            target_2 = self.model2(data)
            bce_loss_2, bce_p_loss_2, bce_u_loss_2 = BCELossPf(equal_weight=True)(target_2, positive_train_mask,
                                                                                  pse_negative_label, epoch,
                                                                                  self.device)

            conkl_loss, conkl_p_loss, conkl_u_loss = ConKLLossPf(equal_weight=True)(target_1, target_2,
                                                                                    positive_train_mask,
                                                                                    unlabeled_train_mask, epoch,
                                                                                    self.device)

            loss = (1 - self.cfg_trainer['params']['lam']) * (bce_loss_1 + bce_loss_2) + self.cfg_trainer['params'][
                'lam'] * conkl_loss
            p_loss = (1 - self.cfg_trainer['params']['lam']) * (bce_p_loss_1 + bce_p_loss_2) + \
                     self.cfg_trainer['params']['lam'] * conkl_p_loss
            u_loss = (1 - self.cfg_trainer['params']['lam']) * (bce_u_loss_1 + bce_u_loss_2) + \
                     self.cfg_trainer['params']['lam'] * conkl_u_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.clip_recoder_grad_norm(model=self.model1, grad_l2_norm=self.grad_l2_norm_1)
            self.clip_recoder_grad_norm(model=self.model2, grad_l2_norm=self.grad_l2_norm_2)
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
        auc_1, fpr_1, tpr_1, threshold_1, pre_1, rec_1, f1_1 = fcn_evaluate_fn(
            model=self.model1,
            test_dataloader=self.validation_dataloader,
            meta=self.meta,
            cls=self.cfg_dataset['train']['params']['ccls'],
            device=self.device,
            path=self.save_path_1,
            epoch=epoch)

        auc_2, fpr_2, tpr_2, threshold_2, pre_2, rec_2, f1_2 = fcn_evaluate_fn(
            model=self.model2,
            test_dataloader=self.validation_dataloader,
            meta=self.meta,
            cls=self.cfg_dataset['train']['params']['ccls'],
            device=self.device,
            path=self.save_path_2,
            epoch=epoch)

        auc_f, fpr_f, tpr_f, threshold_f, pre_f, rec_f, f1_f = fusion_fcn_evaluate_fn(
            model=[self.model1, self.model2],
            test_dataloader=self.validation_dataloader,
            meta=self.meta,
            cls=self.cfg_dataset['train']['params']['ccls'],
            device=self.device,
            path=self.save_path_f,
            epoch=epoch)

        self.pre_recorder_1.update_scalar(pre_1)
        self.rec_recorder_1.update_scalar(rec_1)
        self.f1_recorder_1.update_scalar(f1_1)

        self.pre_recorder_2.update_scalar(pre_2)
        self.rec_recorder_2.update_scalar(rec_2)
        self.f1_recorder_2.update_scalar(f1_2)

        self.pre_recorder_f.update_scalar(pre_f)
        self.rec_recorder_f.update_scalar(rec_f)
        self.f1_recorder_f.update_scalar(f1_f)

        if epoch == (self.cfg_trainer['params']['max_iters'] - 1):
            auc_roc_1 = dict()
            auc_roc_1['fpr'] = fpr_1
            auc_roc_1['tpr'] = tpr_1
            auc_roc_1['threshold'] = threshold_1
            auc_roc_1['auc'] = auc_1
            np.save(os.path.join(self.save_path_1, 'auc_roc_1.npy'), auc_roc_1)

            auc_roc_2 = dict()
            auc_roc_2['fpr'] = fpr_2
            auc_roc_2['tpr'] = tpr_2
            auc_roc_2['threshold'] = threshold_2
            auc_roc_2['auc'] = auc_2
            np.save(os.path.join(self.save_path_2, 'auc_roc_2.npy'), auc_roc_2)

            auc_roc_f = dict()
            auc_roc_f['fpr'] = fpr_f
            auc_roc_f['tpr'] = tpr_f
            auc_roc_f['threshold'] = threshold_f
            auc_roc_f['auc'] = auc_f
            np.save(os.path.join(self.save_path_f, 'auc_roc_f.npy'), auc_roc_f)

    def _save_checkpoint(self):
        torch.save(self.model1.state_dict(), os.path.join(self.save_path_1, 'checkpoint_1.pth'))
        torch.save(self.model2.state_dict(), os.path.join(self.save_path_2, 'checkpoint_2.pth'))

    def run(self, validation=True):
        self.save_path_1 = os.path.join(self.save_path, 'model1')
        self.save_path_2 = os.path.join(self.save_path, 'model2')
        self.save_path_f = os.path.join(self.save_path, 'modelf')
        bar = tqdm(list(range(self.cfg_trainer['params']['max_iters'])))
        for i in bar:
            self._train(i)
            if validation:
                self._validation(i)
            else:
                self.pre_recorder_1.update_scalar(0)
                self.rec_recorder_1.update_scalar(0)
                self.f1_recorder_1.update_scalar(0)
                self.pre_recorder_2.update_scalar(0)
                self.rec_recorder_2.update_scalar(0)
                self.f1_recorder_2.update_scalar(0)
                self.pre_recorder_f.update_scalar(0)
                self.rec_recorder_f.update_scalar(0)
                self.f1_recorder_f.update_scalar(0)

            logging_string = "{} epoch, P_1 {:.6f}, R_1 {:.6f}, F1_1 {:.6f}, P_2 {:.6f}, R_2 {:.6f}, F1_2 {:.6f}," \
                             " P {:.6f}, R {:.6f}, F1 {:.6f}".format(
                i + 1,
                self.pre_recorder_1.scalar[-1],
                self.rec_recorder_1.scalar[-1],
                self.f1_recorder_1.scalar[-1],
                self.pre_recorder_2.scalar[-1],
                self.rec_recorder_2.scalar[-1],
                self.f1_recorder_2.scalar[-1],
                self.pre_recorder_f.scalar[-1],
                self.rec_recorder_f.scalar[-1],
                self.f1_recorder_f.scalar[-1]
            )

            bar.set_description(logging_string)
            logging.info(logging_string)

        self._save_checkpoint()

        self.loss_recorder.save_scalar_npy('loss_npy', self.save_path)
        self.loss_recorder.save_lineplot_fig('Loss', 'loss', self.save_path)
        self.p_loss_recorder.save_scalar_npy('p_loss_npy', self.save_path)
        self.p_loss_recorder.save_lineplot_fig('Estimated Positive Loss', 'p_loss', self.save_path)
        self.u_loss_recorder.save_scalar_npy('u_loss_npy', self.save_path)
        self.u_loss_recorder.save_lineplot_fig('Estimated Unlabeled Loss', 'u_loss', self.save_path)
        self.grad_l2_norm_1.save_scalar_npy('grad_l2_norm_npy', self.save_path_1)
        self.grad_l2_norm_1.save_lineplot_fig('Grad L2 Norm', 'grad_l2_norm_1', self.save_path_1)
        self.grad_l2_norm_2.save_scalar_npy('grad_l2_norm_npy', self.save_path_2)
        self.grad_l2_norm_2.save_lineplot_fig('Grad L2 Norm', 'grad_l2_norm_2', self.save_path_2)

        if validation:
            self.f1_recorder_1.save_scalar_npy('f1_npy_1', self.save_path_1)
            self.f1_recorder_1.save_lineplot_fig('F1-score', 'f1-score_1', self.save_path_1)
            self.rec_recorder_1.save_scalar_npy('recall_npy_1', self.save_path_1)
            self.rec_recorder_1.save_lineplot_fig('Recall', 'recall_1', self.save_path_1)
            self.pre_recorder_1.save_scalar_npy('precision_npy_1', self.save_path_1)
            self.pre_recorder_1.save_lineplot_fig('Precision', 'precision_1', self.save_path_1)

            self.f1_recorder_2.save_scalar_npy('f1_npy_2', self.save_path_2)
            self.f1_recorder_2.save_lineplot_fig('F1-score', 'f1-score_2', self.save_path_2)
            self.rec_recorder_2.save_scalar_npy('recall_npy_2', self.save_path_2)
            self.rec_recorder_2.save_lineplot_fig('Recall', 'recall_2', self.save_path_2)
            self.pre_recorder_2.save_scalar_npy('precision_npy_2', self.save_path_2)
            self.pre_recorder_2.save_lineplot_fig('Precision', 'precision_2', self.save_path_2)

            self.f1_recorder_f.save_scalar_npy('f1_npy_f', self.save_path_f)
            self.f1_recorder_f.save_lineplot_fig('F1-score', 'f1-score_f', self.save_path_f)
            self.rec_recorder_f.save_scalar_npy('recall_npy_f', self.save_path_f)
            self.rec_recorder_f.save_lineplot_fig('Recall', 'recall_f', self.save_path_f)
            self.pre_recorder_f.save_scalar_npy('precision_npy_f', self.save_path_f)
            self.pre_recorder_f.save_lineplot_fig('Precision', 'precision_f', self.save_path_f)
