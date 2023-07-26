import copy
import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader as torchDataLoader
from pytorch_ood.loss import CACLoss  # , IILoss, CrossEntropyLoss, OutlierExposureLoss, EntropicOpenSetLoss
# from pytorch_ood.detector import MaxSoftmax
from tqdm import tqdm

from HOC.apis import BaseTrainer
from HOC.apis.toolkit_ds3l.dataset_sampler import DS3LHyperData, DS3LRandomSampler
from HOC.apis.toolkit_openmax.openmax import HOCOpenMax
from HOC.apis.toolkit_pytorchood_detector.pytorchood_detector import detector_init
from HOC.apis.validation import mcc_evaluate_fn, os_openmax_evaluate_fn, os_cac_evaluate_fn, os_ii_evaluate_fn, \
    os_pytorchood_detector_evaluate_fn
from HOC.datasets import build_dataloader
from HOC.datasets.data_base import HyperData
from HOC.loss_functions import build_loss_function
from HOC.models import build_model
from HOC.models.ds3l_model import WNet
from HOC.optimization import build_optimizer
from HOC.optimization import build_lr_scheduler
from HOC.utils import TRAINER


class HsiGetPatch:
    def __init__(self, im, gt, mask, patch_size, max_cls):
        self.im = copy.deepcopy(im)
        self.gt = copy.deepcopy(gt)
        self.mask = copy.deepcopy(mask)
        self.max_cls = max_cls
        self.patch_size = patch_size
        self.im_x, self.im_y, self.im_z = self.im.shape

    def locate_sample(self):
        sam = []
        for i in range(self.max_cls):
            _xy = np.array(np.where((self.gt == (i + 1)) & (self.mask == 1))).T
            _sam = np.concatenate([_xy, i * np.ones([_xy.shape[0], 1])], axis=-1)
            try:
                sam = np.concatenate([sam, _sam], axis=0)
            except:
                sam = _sam
        self.sample = sam.astype(int)

    def get_patch(self, xy):
        d = self.patch_size // 2
        x = xy[0]
        y = xy[1]
        sam = self.im[:, (x - d):(x + d + 1), (y - d):(y + d + 1)]
        return np.array(sam)

    def train_sample(self):
        train_x, train_y = [], []
        self.locate_sample()
        _samp = self.sample
        for _cls in range(self.max_cls):
            _xy = _samp[_samp[:, 2] == _cls]
            np.random.shuffle(_xy)
            for xy in _xy:
                train_x.append(self.get_patch(xy[:-1]))
                train_y.append(xy[-1])
        train_x, train_y = np.array(train_x), np.array(train_y)
        idx = np.random.permutation(train_x.shape[0])
        train_x = train_x[idx]
        train_y = train_y[idx]
        return train_x, train_y.astype(int)

    def train_unlabeled_sample(self, unlabeled_indicator):
        train_unlabeled_id_x, train_unlabeled_id_y = np.where(unlabeled_indicator == 1)
        r = int(self.patch_size - 1) / 2
        unlabeled_data = np.zeros((train_unlabeled_id_x.shape[0], self.im.shape[0], self.patch_size, self.patch_size))
        for i in range(train_unlabeled_id_x.shape[0]):
            unlabeled_data[i, :, :, :] = self.im[:,
                                         int(train_unlabeled_id_x[i] - r):int(train_unlabeled_id_x[i] + r + 1),
                                         int(train_unlabeled_id_y[i] - r):int(train_unlabeled_id_y[i] + r + 1)]
        unlabeled_y = np.ones(train_unlabeled_id_x.shape[0]) * -1
        return unlabeled_data, unlabeled_y

    def sample_enhancement(self, sample, label):
        a = np.flip(sample, 1)
        b = np.flip(sample, 2)
        c = np.flip(b, 1)
        newsample = np.concatenate((a, b, c, sample), axis=0)
        newlabel = np.concatenate((label, label, label, label), axis=0)
        return newsample, newlabel


@TRAINER.register_module()
class MccSingleModelTrainer_PB(BaseTrainer):
    def __init__(self, trainer, dataset, model, loss_function, optimizer, lr_scheduler, meta):
        super(MccSingleModelTrainer_PB, self).__init__(trainer=trainer, dataset=dataset, model=model,
                                                       loss_function=loss_function,
                                                       optimizer=optimizer, lr_scheduler=lr_scheduler, meta=meta)
        self.build_scalar_recoder()

    def _build_train_dataloader(self):
        train_pf_dataloader = build_dataloader(self.cfg_dataset['train'])
        r = int((self.cfg_trainer['params']['patch_size'] - 1) / 2)

        image = train_pf_dataloader.dataset.pad_im
        image = np.pad(image, ((0, 0), (r, r), (r, r)), mode='constant')
        gt = train_pf_dataloader.dataset.pad_mask
        gt = np.pad(gt, (r, r), mode='constant').astype(int)
        mask = train_pf_dataloader.dataset.train_indicator
        mask = np.pad(mask, (r, r), mode='constant').astype(int)

        max_cls = gt.max()

        hsi_patch_generation = HsiGetPatch(im=image, gt=gt, mask=mask,
                                           patch_size=self.cfg_trainer['params']['patch_size'],
                                           max_cls=max_cls)

        train_x, train_y = hsi_patch_generation.train_sample()
        # train_x, train_y = hsi_patch_generation.sample_enhancement(train_x, train_y)

        dataset = HyperData(torch.from_numpy(train_x).float(), torch.from_numpy(train_y))
        return image, torchDataLoader(dataset=dataset, batch_size=self.cfg_trainer['params']['batch_size_pb'],
                                      shuffle=True)

    def _build_validation_dataloader(self):
        validation_pf_dataloader = build_dataloader(self.cfg_dataset['test'])
        gt = validation_pf_dataloader.dataset.pad_mask
        mask = validation_pf_dataloader.dataset.test_indicator
        return gt, mask

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
        self.gt, self.test_mask = self._build_validation_dataloader()
        self.model = self._build_model().to(self.device)
        self.loss_function = self._build_loss_function()
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()

    def build_scalar_recoder(self):
        self.oa = self._build_scalar_recoder()
        self.aa = self._build_scalar_recoder()
        self.kappa = self._build_scalar_recoder()
        self.loss_recorder = self._build_scalar_recoder()
        self.grad_l2_norm = self._build_scalar_recoder()

    def _train(self, epoch):
        epoch_loss = 0.0
        num_iter = 0
        self.model.train()
        for (data, y) in self.train_dataloader:
            data = data.to(self.device)
            y = y.to(self.device)

            target = self.model(data)

            loss = self.loss_function(target, y, epoch, self.device)

            self.optimizer.zero_grad()
            loss.backward()

            self.clip_recoder_grad_norm(model=self.model, grad_l2_norm=self.grad_l2_norm)

            self.optimizer.step()

            epoch_loss += loss.item()
            num_iter += 1

        self.lr_scheduler.step()
        self.loss_recorder.update_scalar(epoch_loss / num_iter)

    def _validation(self, epoch):
        acc, acc_per_class, acc_cls, IoU, mean_IoU, kappa = mcc_evaluate_fn(image=self.image,
                                                                            gt=self.gt,
                                                                            mask=self.test_mask,
                                                                            model=self.model,
                                                                            patch_size=self.cfg_trainer['params'][
                                                                                'patch_size'],
                                                                            meta=self.meta,
                                                                            device=self.device,
                                                                            path=self.save_path,
                                                                            epoch=epoch)

        self.oa.update_scalar(acc)
        self.aa.update_scalar(acc_cls)
        self.kappa.update_scalar(kappa)

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
                self.oa.update_scalar(0)
                self.aa.update_scalar(0)
                self.kappa.update_scalar(0)

            logging_string = "{} epoch, loss {:.4f}, OA {:.6f}, AA {:.6f}, Kappa {:.6f}".format(
                i + 1,
                self.loss_recorder.scalar[-1],
                self.oa.scalar[-1],
                self.aa.scalar[-1],
                self.kappa.scalar[-1])

            bar.set_description(logging_string)
            logging.info(logging_string)

        self._save_checkpoint()

        self.loss_recorder.save_scalar_npy('loss_npy', self.save_path)
        self.loss_recorder.save_lineplot_fig('Loss', 'loss', self.save_path)
        self.grad_l2_norm.save_scalar_npy('grad_l2_norm_npy', self.save_path)
        self.grad_l2_norm.save_lineplot_fig('Grad L2 Norm', 'grad_l2_norm', self.save_path)
        if validation:
            self.oa.save_scalar_npy('oa_npy', self.save_path)
            self.oa.save_lineplot_fig('OA', 'oa', self.save_path)
            self.aa.save_scalar_npy('aa_npy', self.save_path)
            self.aa.save_lineplot_fig('AA', 'aa', self.save_path)
            self.kappa.save_scalar_npy('kappa_npy', self.save_path)
            self.kappa.save_lineplot_fig('Kappa', 'kappa', self.save_path)


@TRAINER.register_module()
class OsOpenMaxTrainer_PB(BaseTrainer):
    def __init__(self, trainer, dataset, model, loss_function, optimizer, lr_scheduler, meta):
        super(OsOpenMaxTrainer_PB, self).__init__(trainer=trainer, dataset=dataset, model=model,
                                                  loss_function=loss_function,
                                                  optimizer=optimizer, lr_scheduler=lr_scheduler, meta=meta)
        self.build_scalar_recoder()

    def _build_train_dataloader(self):
        train_pf_dataloader = build_dataloader(self.cfg_dataset['train'])
        r = int((self.cfg_trainer['params']['patch_size'] - 1) / 2)

        image = train_pf_dataloader.dataset.pad_im
        image = np.pad(image, ((0, 0), (r, r), (r, r)), mode='constant')
        gt = train_pf_dataloader.dataset.pad_gt
        gt = np.pad(gt, (r, r), mode='constant').astype(int)
        mask = train_pf_dataloader.dataset.pad_mask
        mask = np.pad(mask, (r, r), mode='constant').astype(int)

        max_cls = int(gt.max())

        hsi_patch_generation = HsiGetPatch(im=image, gt=gt, mask=mask,
                                           patch_size=self.cfg_trainer['params']['patch_size'],
                                           max_cls=max_cls)

        train_x, train_y = hsi_patch_generation.train_sample()
        # train_x, train_y = hsi_patch_generation.sample_enhancement(train_x, train_y)

        dataset = HyperData(torch.from_numpy(train_x).float(), torch.from_numpy(train_y))
        return image, torchDataLoader(dataset=dataset, batch_size=self.cfg_trainer['params']['batch_size_pb'],
                                      shuffle=True)

    def _build_validation_dataloader(self):
        validation_pf_dataloader = build_dataloader(self.cfg_dataset['test'])
        gt = validation_pf_dataloader.dataset.pad_gt
        mask = validation_pf_dataloader.dataset.pad_mask
        return gt, mask

    def _build_model(self):
        return build_model(self.cfg_model)

    def _build_loss_function(self):
        pass

    def _build_optimizer(self):
        pass

    def _build_lr_scheduler(self):
        pass

    def _build_components(self):
        self.image, self.train_dataloader = self._build_train_dataloader()
        self.gt, self.test_mask = self._build_validation_dataloader()
        self.model = self._build_model().to(self.device)

    def build_scalar_recoder(self):
        self.close_acc = self._build_scalar_recoder()
        self.close_aa = self._build_scalar_recoder()
        self.close_kappa = self._build_scalar_recoder()
        self.open_acc = self._build_scalar_recoder()
        self.open_aa = self._build_scalar_recoder()
        self.open_kappa = self._build_scalar_recoder()
        self.os_pre = self._build_scalar_recoder()
        self.os_rec = self._build_scalar_recoder()
        self.os_f1 = self._build_scalar_recoder()
        self.os_auc = self._build_scalar_recoder()

    def _train(self, epoch):
        self.model.load_state_dict(torch.load(self.cfg_trainer['params']['checkpoint_path']))
        self.detector = HOCOpenMax(self.model, tailsize=self.cfg_trainer['params']['tailsize'],
                                   alpha=self.cfg_trainer['params']['alpha'],
                                   euclid_weight=self.cfg_trainer['params']['euclid_weight'])
        self.detector.fit(self.train_dataloader, device=self.device)

    def _validation(self, epoch):
        close_acc, close_acc_cls, close_kappa, open_acc, open_acc_cls, open_kappa, os_pre, os_rec, os_f1, auc, fpr, tpr, threshold = os_openmax_evaluate_fn(
            image=self.image,
            gt=self.gt,
            mask=self.test_mask,
            model=self.model,
            detector=self.detector,
            patch_size=self.cfg_trainer['params']['patch_size'],
            meta=self.meta,
            device=self.device,
            path=self.save_path,
            epoch=epoch)

        self.close_acc.update_scalar(close_acc)
        self.close_aa.update_scalar(close_acc_cls)
        self.close_kappa.update_scalar(close_kappa)
        self.open_acc.update_scalar(open_acc)
        self.open_aa.update_scalar(open_acc_cls)
        self.open_kappa.update_scalar(open_kappa)
        self.os_pre.update_scalar(os_pre)
        self.os_rec.update_scalar(os_rec)
        self.os_f1.update_scalar(os_f1)
        self.os_auc.update_scalar(auc)

        auc_roc = dict()
        auc_roc['fpr'] = fpr
        auc_roc['tpr'] = tpr
        auc_roc['threshold'] = threshold
        auc_roc['auc'] = auc
        np.save(os.path.join(self.save_path, 'auc_roc.npy'), auc_roc)

    def _save_checkpoint(self):
        pass

    def run(self, validation=True):
        self._train(1)
        self._validation(1)

        logging_string = "{} epoch, Close_OA {:.6f}, Close_AA {:.6f}, Close_Kappa {:.6f}, Open_OA {:.6f}, Open_AA {:.6f}, Open_Kappa {:.6f}, Os_Pre {:.6f}, Os_Rec {:.6f}, Os_F1 {:.6f}, Os_AUC {:.6f}".format(
            1,
            self.close_acc.scalar[-1],
            self.close_aa.scalar[-1],
            self.close_kappa.scalar[-1],
            self.open_acc.scalar[-1],
            self.open_aa.scalar[-1],
            self.open_kappa.scalar[-1],
            self.os_pre.scalar[-1],
            self.os_rec.scalar[-1],
            self.os_f1.scalar[-1],
            self.os_auc.scalar[-1])

        print(logging_string)
        logging.info(logging_string)


@TRAINER.register_module()
class OsCACLossTrainer_PB(BaseTrainer):
    def __init__(self, trainer, dataset, model, loss_function, optimizer, lr_scheduler, meta):
        super(OsCACLossTrainer_PB, self).__init__(trainer=trainer, dataset=dataset, model=model,
                                                  loss_function=loss_function,
                                                  optimizer=optimizer, lr_scheduler=lr_scheduler, meta=meta)
        self.build_scalar_recoder()

    def _build_train_dataloader(self):
        train_pf_dataloader = build_dataloader(self.cfg_dataset['train'])
        r = int((self.cfg_trainer['params']['patch_size'] - 1) / 2)

        image = train_pf_dataloader.dataset.pad_im
        image = np.pad(image, ((0, 0), (r, r), (r, r)), mode='constant')
        gt = train_pf_dataloader.dataset.pad_gt
        gt = np.pad(gt, (r, r), mode='constant').astype(int)
        mask = train_pf_dataloader.dataset.pad_mask
        mask = np.pad(mask, (r, r), mode='constant').astype(int)

        max_cls = int(gt.max())

        hsi_patch_generation = HsiGetPatch(im=image, gt=gt, mask=mask,
                                           patch_size=self.cfg_trainer['params']['patch_size'],
                                           max_cls=max_cls)

        train_x, train_y = hsi_patch_generation.train_sample()
        # train_x, train_y = hsi_patch_generation.sample_enhancement(train_x, train_y)

        dataset = HyperData(torch.from_numpy(train_x).float(), torch.from_numpy(train_y))
        return image, torchDataLoader(dataset=dataset, batch_size=self.cfg_trainer['params']['batch_size_pb'],
                                      shuffle=True)

    def _build_validation_dataloader(self):
        validation_pf_dataloader = build_dataloader(self.cfg_dataset['test'])
        gt = validation_pf_dataloader.dataset.pad_gt
        mask = validation_pf_dataloader.dataset.pad_mask
        return gt, mask

    def _build_model(self):
        return build_model(self.cfg_model)

    def _build_loss_function(self):
        return CACLoss(n_classes=self.cfg_trainer['params']['n_classes'],
                       magnitude=self.cfg_trainer['params']['magnitude'], alpha=self.cfg_trainer['params']['alpha'])

    def _build_optimizer(self):
        self.cfg_optimizer['params'].update(params=self.model.parameters())
        return build_optimizer(self.cfg_optimizer)

    def _build_lr_scheduler(self):
        self.cfg_lr_scheduler['params'].update(optimizer=self.optimizer)
        return build_lr_scheduler(self.cfg_lr_scheduler)

    def _build_components(self):
        self.image, self.train_dataloader = self._build_train_dataloader()
        self.gt, self.test_mask = self._build_validation_dataloader()
        self.model = self._build_model().to(self.device)
        self.loss_function = self._build_loss_function().to(self.device)
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()

    def build_scalar_recoder(self):
        self.close_acc = self._build_scalar_recoder()
        self.close_aa = self._build_scalar_recoder()
        self.close_kappa = self._build_scalar_recoder()
        self.open_acc = self._build_scalar_recoder()
        self.open_aa = self._build_scalar_recoder()
        self.open_kappa = self._build_scalar_recoder()
        self.os_pre = self._build_scalar_recoder()
        self.os_rec = self._build_scalar_recoder()
        self.os_f1 = self._build_scalar_recoder()
        self.os_auc = self._build_scalar_recoder()
        self.loss_recorder = self._build_scalar_recoder()

    def _train(self, epoch):
        epoch_loss = 0.0
        num_iter = 0
        self.model.train()
        for (data, y) in self.train_dataloader:
            data = data.to(self.device)
            y = y.to(self.device)
            target = self.model(data)

            distances = self.loss_function.distance(target)

            loss = self.loss_function(distances, y)

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()
            num_iter += 1

        self.lr_scheduler.step()
        self.loss_recorder.update_scalar(epoch_loss / num_iter)

    def _validation(self, epoch):
        close_acc, close_acc_cls, close_kappa, open_acc, open_acc_cls, open_kappa, os_pre, os_rec, os_f1, auc, fpr, tpr, threshold = os_cac_evaluate_fn(
            image=self.image,
            gt=self.gt,
            mask=self.test_mask,
            model=self.model,
            patch_size=self.cfg_trainer['params']['patch_size'],
            meta=self.meta,
            device=self.device,
            path=self.save_path,
            loss_function=self.loss_function,
            epoch=epoch)

        self.close_acc.update_scalar(close_acc)
        self.close_aa.update_scalar(close_acc_cls)
        self.close_kappa.update_scalar(close_kappa)
        self.open_acc.update_scalar(open_acc)
        self.open_aa.update_scalar(open_acc_cls)
        self.open_kappa.update_scalar(open_kappa)
        self.os_pre.update_scalar(os_pre)
        self.os_rec.update_scalar(os_rec)
        self.os_f1.update_scalar(os_f1)
        self.os_auc.update_scalar(auc)

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
                self.close_acc.update_scalar(0)
                self.close_aa.update_scalar(0)
                self.close_kappa.update_scalar(0)
                self.open_acc.update_scalar(0)
                self.open_aa.update_scalar(0)
                self.open_kappa.update_scalar(0)
                self.os_pre.update_scalar(0)
                self.os_rec.update_scalar(0)
                self.os_f1.update_scalar(0)
                self.os_auc.update_scalar(0)

            logging_string = "{} epoch, Loss {:.6f},Close_OA {:.6f}, Close_AA {:.6f}, Close_Kappa {:.6f}, Open_OA {:.6f}, Open_AA {:.6f}, Open_Kappa {:.6f}, Os_Pre {:.6f}, Os_Rec {:.6f}, Os_F1 {:.6f}, Os_AUC {:.6f}".format(
                i + 1,
                self.loss_recorder.scalar[-1],
                self.close_acc.scalar[-1],
                self.close_aa.scalar[-1],
                self.close_kappa.scalar[-1],
                self.open_acc.scalar[-1],
                self.open_aa.scalar[-1],
                self.open_kappa.scalar[-1],
                self.os_pre.scalar[-1],
                self.os_rec.scalar[-1],
                self.os_f1.scalar[-1],
                self.os_auc.scalar[-1])

            bar.set_description(logging_string)
            logging.info(logging_string)

        self._save_checkpoint()

        self.loss_recorder.save_scalar_npy('loss_npy', self.save_path)
        self.loss_recorder.save_lineplot_fig('Loss', 'loss', self.save_path)
        if validation:
            self.close_acc.save_scalar_npy('close_oa_npy', self.save_path)
            self.close_acc.save_lineplot_fig('Close_OA', 'close_oa', self.save_path)
            self.close_aa.save_scalar_npy('close_aa_npy', self.save_path)
            self.close_aa.save_lineplot_fig('Close_AA', 'close_aa', self.save_path)
            self.close_kappa.save_scalar_npy('close_kappa_npy', self.save_path)
            self.close_kappa.save_lineplot_fig('Close_Kappa', 'close_kappa', self.save_path)

            self.open_acc.save_scalar_npy('open_oa_npy', self.save_path)
            self.open_acc.save_lineplot_fig('Open_OA', 'open_oa', self.save_path)
            self.open_aa.save_scalar_npy('open_aa_npy', self.save_path)
            self.open_aa.save_lineplot_fig('Open_AA', 'open_aa', self.save_path)
            self.open_kappa.save_scalar_npy('open_kappa_npy', self.save_path)
            self.open_kappa.save_lineplot_fig('Open_Kappa', 'open_kappa', self.save_path)

            self.os_pre.save_scalar_npy('os_pre_npy', self.save_path)
            self.os_pre.save_lineplot_fig('Os_Pre', 'os_pre', self.save_path)
            self.os_rec.save_scalar_npy('os_rec_npy', self.save_path)
            self.os_rec.save_lineplot_fig('Os_Rec', 'os_rec', self.save_path)
            self.os_f1.save_scalar_npy('os_f1_npy', self.save_path)
            self.os_f1.save_lineplot_fig('Os_F1', 'os_f1', self.save_path)
            self.os_auc.save_scalar_npy('os_auc_npy', self.save_path)
            self.os_auc.save_lineplot_fig('Os_AUC', 'os_auc', self.save_path)


# @TRAINER.register_module()
# class OsIILossTrainer_PB(BaseTrainer):
#     def __init__(self, trainer, dataset, model, loss_function, optimizer, lr_scheduler, meta):
#         super(OsIILossTrainer_PB, self).__init__(trainer=trainer, dataset=dataset, model=model,
#                                                  loss_function=loss_function,
#                                                  optimizer=optimizer, lr_scheduler=lr_scheduler, meta=meta)
#         self.build_scalar_recoder()
#
#     def _build_train_dataloader(self):
#         train_pf_dataloader = build_dataloader(self.cfg_dataset['train'])
#         r = int((self.cfg_trainer['params']['patch_size'] - 1) / 2)
#
#         image = train_pf_dataloader.dataset.pad_im
#         image = np.pad(image, ((0, 0), (r, r), (r, r)), mode='constant')
#         gt = train_pf_dataloader.dataset.pad_gt
#         gt = np.pad(gt, (r, r), mode='constant').astype(int)
#         mask = train_pf_dataloader.dataset.pad_mask
#         mask = np.pad(mask, (r, r), mode='constant').astype(int)
#
#         max_cls = int(gt.max())
#
#         hsi_patch_generation = HsiGetPatch(im=image, gt=gt, mask=mask,
#                                            patch_size=self.cfg_trainer['params']['patch_size'],
#                                            max_cls=max_cls)
#
#         train_x, train_y = hsi_patch_generation.train_sample()
#         train_x, train_y = hsi_patch_generation.sample_enhancement(train_x, train_y)
#
#         dataset = HyperData(torch.from_numpy(train_x).float(), torch.from_numpy(train_y))
#         return image, torchDataLoader(dataset=dataset, batch_size=self.cfg_trainer['params']['batch_size_pb'],
#                                       shuffle=True)
#
#     def _build_validation_dataloader(self):
#         validation_pf_dataloader = build_dataloader(self.cfg_dataset['test'])
#         gt = validation_pf_dataloader.dataset.pad_gt
#         mask = validation_pf_dataloader.dataset.pad_mask
#         return gt, mask
#
#     def _build_model(self):
#         return build_model(self.cfg_model)
#
#     def _build_loss_function(self):
#         return IILoss(n_classes=self.cfg_trainer['params']['n_classes'],
#                       n_embedding=self.cfg_trainer['params']['n_embedding'],
#                       alpha=self.cfg_trainer['params']['alpha'])
#
#     def _build_optimizer(self):
#         self.cfg_optimizer['params'].update(params=self.model.parameters())
#         return build_optimizer(self.cfg_optimizer)
#
#     def _build_lr_scheduler(self):
#         self.cfg_lr_scheduler['params'].update(optimizer=self.optimizer)
#         return build_lr_scheduler(self.cfg_lr_scheduler)
#
#     def _build_components(self):
#         self.image, self.train_dataloader = self._build_train_dataloader()
#         self.gt, self.test_mask = self._build_validation_dataloader()
#         self.model = self._build_model().to(self.device)
#         self.loss_function = self._build_loss_function().to(self.device)
#         self.cross_entropy_loss = CrossEntropyLoss().to(self.device)
#         self.optimizer = self._build_optimizer()
#         self.lr_scheduler = self._build_lr_scheduler()
#
#     def build_scalar_recoder(self):
#         self.close_acc = self._build_scalar_recoder()
#         self.close_aa = self._build_scalar_recoder()
#         self.close_kappa = self._build_scalar_recoder()
#         self.open_acc = self._build_scalar_recoder()
#         self.open_aa = self._build_scalar_recoder()
#         self.open_kappa = self._build_scalar_recoder()
#         self.os_pre = self._build_scalar_recoder()
#         self.os_rec = self._build_scalar_recoder()
#         self.os_f1 = self._build_scalar_recoder()
#         self.os_auc = self._build_scalar_recoder()
#         self.loss_recorder = self._build_scalar_recoder()
#
#     def _train(self, epoch):
#         epoch_loss = 0.0
#         num_iter = 0
#         self.model.train()
#         for (data, y) in self.train_dataloader:
#             data = data.to(self.device)
#             y = y.to(self.device)
#             target = self.model(data)
#
#             loss = self.loss_function(target, y) + self.cross_entropy_loss(target, y)
#
#             self.optimizer.zero_grad()
#             loss.backward()
#
#             self.optimizer.step()
#
#             epoch_loss += loss.item()
#             num_iter += 1
#
#         self.lr_scheduler.step()
#         self.loss_recorder.update_scalar(epoch_loss / num_iter)
#
#     def _validation(self, epoch):
#         close_acc, close_acc_cls, close_kappa, open_acc, open_acc_cls, open_kappa, os_pre, os_rec, os_f1, auc, fpr, tpr, threshold = os_ii_evaluate_fn(
#             image=self.image,
#             gt=self.gt,
#             mask=self.test_mask,
#             model=self.model,
#             patch_size=self.cfg_trainer['params']['patch_size'],
#             meta=self.meta,
#             device=self.device,
#             path=self.save_path,
#             loss_function=self.loss_function,
#             epoch=epoch)
#
#         self.close_acc.update_scalar(close_acc)
#         self.close_aa.update_scalar(close_acc_cls)
#         self.close_kappa.update_scalar(close_kappa)
#         self.open_acc.update_scalar(open_acc)
#         self.open_aa.update_scalar(open_acc_cls)
#         self.open_kappa.update_scalar(open_kappa)
#         self.os_pre.update_scalar(os_pre)
#         self.os_rec.update_scalar(os_rec)
#         self.os_f1.update_scalar(os_f1)
#         self.os_auc.update_scalar(auc)
#
#     def _save_checkpoint(self):
#         torch.save(self.model.state_dict(), os.path.join(self.save_path, 'checkpoint.pth'))
#
#     def run(self, validation=True):
#         bar = tqdm(list(range(self.cfg_trainer['params']['max_iters'])))
#         for i in bar:
#             self._train(i)
#
#             if i == self.cfg_trainer['params']['max_iters'] - 1:
#                 validation = True
#             else:
#                 validation = False  ###
#
#             if validation:
#                 self._validation(i)
#             else:
#                 self.close_acc.update_scalar(0)
#                 self.close_aa.update_scalar(0)
#                 self.close_kappa.update_scalar(0)
#                 self.open_acc.update_scalar(0)
#                 self.open_aa.update_scalar(0)
#                 self.open_kappa.update_scalar(0)
#                 self.os_pre.update_scalar(0)
#                 self.os_rec.update_scalar(0)
#                 self.os_f1.update_scalar(0)
#                 self.os_auc.update_scalar(0)
#
#             logging_string = "{} epoch, Loss {:.6f},Close_OA {:.6f}, Close_AA {:.6f}, Close_Kappa {:.6f}, Open_OA {:.6f}, Open_AA {:.6f}, Open_Kappa {:.6f}, Os_Pre {:.6f}, Os_Rec {:.6f}, Os_F1 {:.6f}, Os_AUC {:.6f}".format(
#                 i + 1,
#                 self.loss_recorder.scalar[-1],
#                 self.close_acc.scalar[-1],
#                 self.close_aa.scalar[-1],
#                 self.close_kappa.scalar[-1],
#                 self.open_acc.scalar[-1],
#                 self.open_aa.scalar[-1],
#                 self.open_kappa.scalar[-1],
#                 self.os_pre.scalar[-1],
#                 self.os_rec.scalar[-1],
#                 self.os_f1.scalar[-1],
#                 self.os_auc.scalar[-1])
#
#             bar.set_description(logging_string)
#             logging.info(logging_string)
#
#         self._save_checkpoint()
#
#         self.loss_recorder.save_scalar_npy('loss_npy', self.save_path)
#         self.loss_recorder.save_lineplot_fig('Loss', 'loss', self.save_path)
#         if validation:
#             self.close_acc.save_scalar_npy('close_oa_npy', self.save_path)
#             self.close_acc.save_lineplot_fig('Close_OA', 'close_oa', self.save_path)
#             self.close_aa.save_scalar_npy('close_aa_npy', self.save_path)
#             self.close_aa.save_lineplot_fig('Close_AA', 'close_aa', self.save_path)
#             self.close_kappa.save_scalar_npy('close_kappa_npy', self.save_path)
#             self.close_kappa.save_lineplot_fig('Close_Kappa', 'close_kappa', self.save_path)
#
#             self.open_acc.save_scalar_npy('open_oa_npy', self.save_path)
#             self.open_acc.save_lineplot_fig('Open_OA', 'open_oa', self.save_path)
#             self.open_aa.save_scalar_npy('open_aa_npy', self.save_path)
#             self.open_aa.save_lineplot_fig('Open_AA', 'open_aa', self.save_path)
#             self.open_kappa.save_scalar_npy('open_kappa_npy', self.save_path)
#             self.open_kappa.save_lineplot_fig('Open_Kappa', 'open_kappa', self.save_path)
#
#             self.os_pre.save_scalar_npy('os_pre_npy', self.save_path)
#             self.os_pre.save_lineplot_fig('Os_Pre', 'os_pre', self.save_path)
#             self.os_rec.save_scalar_npy('os_rec_npy', self.save_path)
#             self.os_rec.save_lineplot_fig('Os_Rec', 'os_rec', self.save_path)
#             self.os_f1.save_scalar_npy('os_f1_npy', self.save_path)
#             self.os_f1.save_lineplot_fig('Os_F1', 'os_f1', self.save_path)
#             self.os_auc.save_scalar_npy('os_auc_npy', self.save_path)
#             self.os_auc.save_lineplot_fig('Os_AUC', 'os_auc', self.save_path)


@TRAINER.register_module()
class OsPOODDetectorTrainer_PB(BaseTrainer):
    def __init__(self, trainer, dataset, model, loss_function, optimizer, lr_scheduler, meta):
        super(OsPOODDetectorTrainer_PB, self).__init__(trainer=trainer, dataset=dataset, model=model,
                                                       loss_function=loss_function,
                                                       optimizer=optimizer, lr_scheduler=lr_scheduler, meta=meta)
        self.build_scalar_recoder()

    def _build_train_dataloader(self):
        train_pf_dataloader = build_dataloader(self.cfg_dataset['train'])
        r = int((self.cfg_trainer['params']['patch_size'] - 1) / 2)

        image = train_pf_dataloader.dataset.pad_im
        image = np.pad(image, ((0, 0), (r, r), (r, r)), mode='constant')
        gt = train_pf_dataloader.dataset.pad_gt
        gt = np.pad(gt, (r, r), mode='constant').astype(int)
        mask = train_pf_dataloader.dataset.pad_mask
        mask = np.pad(mask, (r, r), mode='constant').astype(int)

        max_cls = int(gt.max())

        hsi_patch_generation = HsiGetPatch(im=image, gt=gt, mask=mask,
                                           patch_size=self.cfg_trainer['params']['patch_size'],
                                           max_cls=max_cls)

        train_x, train_y = hsi_patch_generation.train_sample()
        # train_x, train_y = hsi_patch_generation.sample_enhancement(train_x, train_y)

        dataset = HyperData(torch.from_numpy(train_x).float(), torch.from_numpy(train_y))
        return image, torchDataLoader(dataset=dataset, batch_size=self.cfg_trainer['params']['batch_size_pb'],
                                      shuffle=True)

    def _build_validation_dataloader(self):
        validation_pf_dataloader = build_dataloader(self.cfg_dataset['test'])
        gt = validation_pf_dataloader.dataset.pad_gt
        mask = validation_pf_dataloader.dataset.pad_mask
        return gt, mask

    def _build_model(self):
        return build_model(self.cfg_model)

    def _build_loss_function(self):
        pass

    def _build_optimizer(self):
        pass

    def _build_lr_scheduler(self):
        pass

    def _build_components(self):
        self.image, self.train_dataloader = self._build_train_dataloader()
        self.gt, self.test_mask = self._build_validation_dataloader()
        self.model = self._build_model().to(self.device)

    def build_scalar_recoder(self):
        self.close_acc = self._build_scalar_recoder()
        self.close_aa = self._build_scalar_recoder()
        self.close_kappa = self._build_scalar_recoder()
        self.open_acc = self._build_scalar_recoder()
        self.open_aa = self._build_scalar_recoder()
        self.open_kappa = self._build_scalar_recoder()
        self.os_pre = self._build_scalar_recoder()
        self.os_rec = self._build_scalar_recoder()
        self.os_f1 = self._build_scalar_recoder()
        self.os_auc = self._build_scalar_recoder()

    def _train(self, epoch):
        self.model.load_state_dict(torch.load(self.cfg_trainer['params']['checkpoint_path']))
        self.model.eval()
        self.detector = detector_init(detector_name=self.cfg_trainer['params']['detector_name'], model=self.model)
        if self.cfg_trainer['params']['detector_name'] == 'MCD':
            self.detector.fit(self.train_dataloader)
        else:
            self.detector.fit(self.train_dataloader, device=self.device)

    def _validation(self, epoch):
        close_acc, close_acc_cls, close_kappa, open_acc, open_acc_cls, open_kappa, os_pre, os_rec, os_f1, auc, fpr, tpr, threshold = os_pytorchood_detector_evaluate_fn(
            image=self.image,
            gt=self.gt,
            mask=self.test_mask,
            model=self.model,
            detector=self.detector,
            patch_size=self.cfg_trainer['params']['patch_size'],
            meta=self.meta,
            device=self.device,
            path=self.save_path,
            epoch=epoch)

        self.close_acc.update_scalar(close_acc)
        self.close_aa.update_scalar(close_acc_cls)
        self.close_kappa.update_scalar(close_kappa)
        self.open_acc.update_scalar(open_acc)
        self.open_aa.update_scalar(open_acc_cls)
        self.open_kappa.update_scalar(open_kappa)
        self.os_pre.update_scalar(os_pre)
        self.os_rec.update_scalar(os_rec)
        self.os_f1.update_scalar(os_f1)
        self.os_auc.update_scalar(auc)

        auc_roc = dict()
        auc_roc['fpr'] = fpr
        auc_roc['tpr'] = tpr
        auc_roc['threshold'] = threshold
        auc_roc['auc'] = auc
        np.save(os.path.join(self.save_path, 'auc_roc.npy'), auc_roc)

    def _save_checkpoint(self):
        pass

    def run(self, validation=True):
        self._train(1)
        self._validation(1)

        logging_string = "{} epoch, Close_OA {:.6f}, Close_AA {:.6f}, Close_Kappa {:.6f}, Open_OA {:.6f}, Open_AA {:.6f}, Open_Kappa {:.6f}, Os_Pre {:.6f}, Os_Rec {:.6f}, Os_F1 {:.6f}, Os_AUC {:.6f}".format(
            1,
            self.close_acc.scalar[-1],
            self.close_aa.scalar[-1],
            self.close_kappa.scalar[-1],
            self.open_acc.scalar[-1],
            self.open_aa.scalar[-1],
            self.open_kappa.scalar[-1],
            self.os_pre.scalar[-1],
            self.os_rec.scalar[-1],
            self.os_f1.scalar[-1],
            self.os_auc.scalar[-1])

        print(logging_string)
        logging.info(logging_string)


@TRAINER.register_module()
class OsPOODSupervisedLossTrainer_PB(BaseTrainer):
    def __init__(self, trainer, dataset, model, loss_function, optimizer, lr_scheduler, meta):
        super(OsPOODSupervisedLossTrainer_PB, self).__init__(trainer=trainer, dataset=dataset, model=model,
                                                             loss_function=loss_function, optimizer=optimizer,
                                                             lr_scheduler=lr_scheduler, meta=meta)
        self.build_scalar_recoder()

    def _build_train_dataloader(self):
        train_pf_dataloader = build_dataloader(self.cfg_dataset['train'])
        r = int((self.cfg_trainer['params']['patch_size'] - 1) / 2)

        image = train_pf_dataloader.dataset.pad_im
        image = np.pad(image, ((0, 0), (r, r), (r, r)), mode='constant')
        gt = train_pf_dataloader.dataset.pad_gt
        gt = np.pad(gt, (r, r), mode='constant').astype(int)
        mask = train_pf_dataloader.dataset.pad_mask
        mask = np.pad(mask, (r, r), mode='constant').astype(int)
        unlabeled_indicator = train_pf_dataloader.dataset.pad_unlabeled_indicator[0]
        unlabeled_indicator = np.pad(unlabeled_indicator, (r, r), mode='constant').astype(int)

        max_cls = int(gt.max())

        hsi_patch_generation = HsiGetPatch(im=image, gt=gt, mask=mask,
                                           patch_size=self.cfg_trainer['params']['patch_size'],
                                           max_cls=max_cls)

        train_positive_x, train_positive_y = hsi_patch_generation.train_sample()
        train_unlabeled_x, train_unlabeled_y = hsi_patch_generation.train_unlabeled_sample(unlabeled_indicator)
        train_x = np.concatenate((train_positive_x, train_unlabeled_x), axis=0)
        train_y = np.concatenate((train_positive_y, train_unlabeled_y)).astype(np.int)
        # train_x, train_y = hsi_patch_generation.sample_enhancement(train_x, train_y)

        dataset = HyperData(torch.from_numpy(train_x).float(), torch.from_numpy(train_y))
        return image, torchDataLoader(dataset=dataset, batch_size=self.cfg_trainer['params']['batch_size_pb'],
                                      shuffle=True)

    def _build_validation_dataloader(self):
        validation_pf_dataloader = build_dataloader(self.cfg_dataset['test'])
        gt = validation_pf_dataloader.dataset.pad_gt
        mask = validation_pf_dataloader.dataset.pad_mask
        return gt, mask

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
        self.gt, self.test_mask = self._build_validation_dataloader()
        self.model = self._build_model().to(self.device)
        self.loss_function = self._build_loss_function().to(self.device)
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()

    def build_scalar_recoder(self):
        self.close_acc = self._build_scalar_recoder()
        self.close_aa = self._build_scalar_recoder()
        self.close_kappa = self._build_scalar_recoder()
        self.open_acc = self._build_scalar_recoder()
        self.open_aa = self._build_scalar_recoder()
        self.open_kappa = self._build_scalar_recoder()
        self.os_pre = self._build_scalar_recoder()
        self.os_rec = self._build_scalar_recoder()
        self.os_f1 = self._build_scalar_recoder()
        self.os_auc = self._build_scalar_recoder()
        self.loss_recorder = self._build_scalar_recoder()

    def _train(self, epoch):
        epoch_loss = 0.0
        num_iter = 0
        self.model.train()
        for (data, y) in self.train_dataloader:
            data = data.to(self.device)
            y = y.to(self.device)
            target = self.model(data)

            loss = self.loss_function(target, y)

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()
            num_iter += 1

        self.lr_scheduler.step()
        self.loss_recorder.update_scalar(epoch_loss / num_iter)

    def _validation(self, epoch):
        self.detector = detector_init(detector_name=self.cfg_trainer['params']['detector_name'], model=self.model)
        close_acc, close_acc_cls, close_kappa, open_acc, open_acc_cls, open_kappa, os_pre, os_rec, os_f1, auc, fpr, tpr, threshold = os_pytorchood_detector_evaluate_fn(
            image=self.image,
            gt=self.gt,
            mask=self.test_mask,
            model=self.model,
            detector=self.detector,
            patch_size=self.cfg_trainer['params']['patch_size'],
            meta=self.meta,
            device=self.device,
            path=self.save_path,
            epoch=epoch)

        self.close_acc.update_scalar(close_acc)
        self.close_aa.update_scalar(close_acc_cls)
        self.close_kappa.update_scalar(close_kappa)
        self.open_acc.update_scalar(open_acc)
        self.open_aa.update_scalar(open_acc_cls)
        self.open_kappa.update_scalar(open_kappa)
        self.os_pre.update_scalar(os_pre)
        self.os_rec.update_scalar(os_rec)
        self.os_f1.update_scalar(os_f1)
        self.os_auc.update_scalar(auc)

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
        self.model.load_state_dict(torch.load(self.cfg_trainer['params']['checkpoint_path']))
        print("Model has been loaded!")
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
                self.close_acc.update_scalar(0)
                self.close_aa.update_scalar(0)
                self.close_kappa.update_scalar(0)
                self.open_acc.update_scalar(0)
                self.open_aa.update_scalar(0)
                self.open_kappa.update_scalar(0)
                self.os_pre.update_scalar(0)
                self.os_rec.update_scalar(0)
                self.os_f1.update_scalar(0)
                self.os_auc.update_scalar(0)

            logging_string = "{} epoch, Loss {:.6f},Close_OA {:.6f}, Close_AA {:.6f}, Close_Kappa {:.6f}, Open_OA {:.6f}, Open_AA {:.6f}, Open_Kappa {:.6f}, Os_Pre {:.6f}, Os_Rec {:.6f}, Os_F1 {:.6f}, Os_AUC {:.6f}".format(
                i + 1,
                self.loss_recorder.scalar[-1],
                self.close_acc.scalar[-1],
                self.close_aa.scalar[-1],
                self.close_kappa.scalar[-1],
                self.open_acc.scalar[-1],
                self.open_aa.scalar[-1],
                self.open_kappa.scalar[-1],
                self.os_pre.scalar[-1],
                self.os_rec.scalar[-1],
                self.os_f1.scalar[-1],
                self.os_auc.scalar[-1])

            bar.set_description(logging_string)
            logging.info(logging_string)

        self._save_checkpoint()

        self.loss_recorder.save_scalar_npy('loss_npy', self.save_path)
        self.loss_recorder.save_lineplot_fig('Loss', 'loss', self.save_path)
        if validation:
            self.close_acc.save_scalar_npy('close_oa_npy', self.save_path)
            self.close_acc.save_lineplot_fig('Close_OA', 'close_oa', self.save_path)
            self.close_aa.save_scalar_npy('close_aa_npy', self.save_path)
            self.close_aa.save_lineplot_fig('Close_AA', 'close_aa', self.save_path)
            self.close_kappa.save_scalar_npy('close_kappa_npy', self.save_path)
            self.close_kappa.save_lineplot_fig('Close_Kappa', 'close_kappa', self.save_path)

            self.open_acc.save_scalar_npy('open_oa_npy', self.save_path)
            self.open_acc.save_lineplot_fig('Open_OA', 'open_oa', self.save_path)
            self.open_aa.save_scalar_npy('open_aa_npy', self.save_path)
            self.open_aa.save_lineplot_fig('Open_AA', 'open_aa', self.save_path)
            self.open_kappa.save_scalar_npy('open_kappa_npy', self.save_path)
            self.open_kappa.save_lineplot_fig('Open_Kappa', 'open_kappa', self.save_path)

            self.os_pre.save_scalar_npy('os_pre_npy', self.save_path)
            self.os_pre.save_lineplot_fig('Os_Pre', 'os_pre', self.save_path)
            self.os_rec.save_scalar_npy('os_rec_npy', self.save_path)
            self.os_rec.save_lineplot_fig('Os_Rec', 'os_rec', self.save_path)
            self.os_f1.save_scalar_npy('os_f1_npy', self.save_path)
            self.os_f1.save_lineplot_fig('Os_F1', 'os_f1', self.save_path)
            self.os_auc.save_scalar_npy('os_auc_npy', self.save_path)
            self.os_auc.save_lineplot_fig('Os_AUC', 'os_auc', self.save_path)


@TRAINER.register_module()
class OsDS3LTrainer_PB(BaseTrainer):
    def __init__(self, trainer, dataset, model, loss_function, optimizer, lr_scheduler, meta):
        super(OsDS3LTrainer_PB, self).__init__(trainer=trainer, dataset=dataset, model=model,
                                               loss_function=loss_function,
                                               optimizer=optimizer, lr_scheduler=lr_scheduler, meta=meta)
        self.build_scalar_recoder()

    def _build_train_dataloader(self):
        train_pf_dataloader = build_dataloader(self.cfg_dataset['train'])
        r = int((self.cfg_trainer['params']['patch_size'] - 1) / 2)

        image = train_pf_dataloader.dataset.pad_im
        image = np.pad(image, ((0, 0), (r, r), (r, r)), mode='constant')
        gt = train_pf_dataloader.dataset.pad_gt
        gt = np.pad(gt, (r, r), mode='constant').astype(int)
        mask = train_pf_dataloader.dataset.pad_mask
        mask = np.pad(mask, (r, r), mode='constant').astype(int)
        unlabeled_indicator = train_pf_dataloader.dataset.pad_unlabeled_indicator[0]
        unlabeled_indicator = np.pad(unlabeled_indicator, (r, r), mode='constant').astype(int)

        max_cls = int(gt.max())

        hsi_patch_generation = HsiGetPatch(im=image, gt=gt, mask=mask,
                                           patch_size=self.cfg_trainer['params']['patch_size'],
                                           max_cls=max_cls)

        train_positive_x, train_positive_y = hsi_patch_generation.train_sample()
        train_unlabeled_x, train_unlabeled_y = hsi_patch_generation.train_unlabeled_sample(unlabeled_indicator)
        train_positive_y = train_positive_y.astype(np.int)
        train_unlabeled_y = train_unlabeled_y.astype(np.int)
        # train_x = np.concatenate((train_positive_x, train_unlabeled_x), axis=0)
        # train_y = np.concatenate((train_positive_y, train_unlabeled_y)).astype(np.int)
        # train_x, train_y = hsi_patch_generation.sample_enhancement(train_x, train_y)

        positive_train_dataset = DS3LHyperData(dataset=torch.from_numpy(train_positive_x).float(),
                                               label=torch.from_numpy(train_positive_y))
        unlabeled_train_dataset = DS3LHyperData(dataset=torch.from_numpy(train_unlabeled_x).float(),
                                                label=torch.from_numpy(train_unlabeled_y))

        positive_train_dataloader = torchDataLoader(dataset=positive_train_dataset,
                                                    batch_size=self.cfg_trainer['params']['batch_size_pb'],
                                                    sampler=DS3LRandomSampler(len(positive_train_dataset),
                                                                              self.cfg_trainer['params']['iterations'] *
                                                                              self.cfg_trainer['params'][
                                                                                  'batch_size_pb']),
                                                    drop_last=True)
        unlabeled_train_dataloader = torchDataLoader(dataset=unlabeled_train_dataset,
                                                     batch_size=self.cfg_trainer['params']['batch_size_pb'],
                                                     sampler=DS3LRandomSampler(len(unlabeled_train_dataset),
                                                                               self.cfg_trainer['params'][
                                                                                   'iterations'] *
                                                                               self.cfg_trainer['params'][
                                                                                   'batch_size_pb']),
                                                     drop_last=True)
        return image, positive_train_dataloader, unlabeled_train_dataloader

    def _build_validation_dataloader(self):
        validation_pf_dataloader = build_dataloader(self.cfg_dataset['test'])
        gt = validation_pf_dataloader.dataset.pad_gt
        mask = validation_pf_dataloader.dataset.pad_mask
        return gt, mask

    def _build_model(self):
        return build_model(self.cfg_model)

    def _build_loss_function(self):
        return build_loss_function(self.cfg_loss_function)

    def _build_optimizer(self):
        self.cfg_optimizer['params'].update(params=self.model.params())
        return build_optimizer(self.cfg_optimizer)

    def _build_lr_scheduler(self):
        pass

    def _build_components(self):
        self.image, self.positive_train_dataloader, self.unlabeled_train_dataloader = self._build_train_dataloader()
        self.gt, self.test_mask = self._build_validation_dataloader()
        self.model = self._build_model().to(self.device)
        self.loss_function = self._build_loss_function().to(self.device)
        self.optimizer = self._build_optimizer()

    def build_scalar_recoder(self):
        self.oa = self._build_scalar_recoder()
        self.aa = self._build_scalar_recoder()
        self.kappa = self._build_scalar_recoder()
        self.loss_recorder = self._build_scalar_recoder()

    def _train(self, epoch):
        wnet = WNet(self.cfg_trainer['params']['n_classes'], 100, 1).to(self.device)
        wnet.train()

        iteration = 0

        optimizer_wnet = torch.optim.Adam(wnet.params(), lr=self.cfg_trainer['params']['lr_wnet'])

        with tqdm(total=self.cfg_trainer['params']['iterations']) as bar:
            for l_data, u_data in zip(self.positive_train_dataloader, self.unlabeled_train_dataloader):
                iteration += 1
                l_images, l_labels, _ = l_data
                u_images, u_labels, idx = u_data

                l_images, l_labels = l_images.to(self.device).float(), l_labels.to(self.device).long()
                u_images, u_labels = u_images.to(self.device).float(), u_labels.to(self.device).long()

                self.model.train()
                meta_net = self._build_model().to(self.device)
                meta_net.load_state_dict(self.model.state_dict())

                # cat labeled and unlabeled data
                labels = torch.cat([l_labels, u_labels], 0)
                labels[-len(u_labels):] = -1  # unlabeled mask
                unlabeled_mask = (labels == -1).float()
                images = torch.cat([l_images, u_images], 0)

                # coefficient for unsupervised loss
                coef = 10.0 * math.exp(-5 * (1 - min(iteration / self.cfg_trainer['params']['warmup'], 1)) ** 2)

                out = meta_net(images)
                ssl_loss = self.loss_function(images, out.detach(), meta_net, unlabeled_mask)

                cost_w = torch.reshape(ssl_loss[len(l_labels):], (len(ssl_loss[len(l_labels):]), 1))

                weight = wnet(out.softmax(1)[len(l_labels):])
                norm = torch.sum(weight)

                cls_loss = F.cross_entropy(out, labels, reduction='none', ignore_index=-1).mean()
                if norm != 0:
                    loss_hat = cls_loss + coef * (torch.sum(cost_w * weight) / norm + ssl_loss[:len(l_labels)].mean())
                else:
                    loss_hat = cls_loss + coef * (torch.sum(cost_w * weight) + ssl_loss[:len(l_labels)].mean())

                meta_net.zero_grad()
                grads = torch.autograd.grad(loss_hat, (meta_net.params()), create_graph=True)
                meta_net.update_params(lr_inner=self.cfg_trainer['params']['meta_lr'], source_params=grads)
                del grads

                # compute upper level objective
                y_g_hat = meta_net(l_images)
                l_g_meta = F.cross_entropy(y_g_hat, l_labels)

                optimizer_wnet.zero_grad()
                l_g_meta.backward()
                optimizer_wnet.step()

                out = self.model(images)

                ssl_loss = self.loss_function(images, out.detach(), self.model, unlabeled_mask)
                cls_loss = F.cross_entropy(out, labels, reduction='none', ignore_index=-1).mean()
                cost_w = torch.reshape(ssl_loss[len(l_labels):], (len(ssl_loss[len(l_labels):]), 1))
                with torch.no_grad():
                    weight = wnet(out.softmax(1)[len(l_labels):])
                    norm = torch.sum(weight)

                if norm != 0:
                    loss = cls_loss + coef * (torch.sum(cost_w * weight) / norm + ssl_loss[:len(l_labels)].mean())
                else:
                    loss = cls_loss + coef * (torch.sum(cost_w * weight) + ssl_loss[:len(l_labels)].mean())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if iteration == self.cfg_trainer['params']['lr_decay_iter']:
                    self.optimizer.param_groups[0]['lr'] *= self.cfg_trainer['params']['lr_decay_factor']

                self.loss_recorder.update_scalar(loss.item())

                bar.set_description("{} step, Loss {:.6f}".format(iteration + 1, self.loss_recorder.scalar[-1]))
                bar.update(1)

    def _validation(self, epoch):
        acc, acc_per_class, acc_cls, IoU, mean_IoU, kappa = mcc_evaluate_fn(image=self.image,
                                                                            gt=self.gt,
                                                                            mask=self.test_mask,
                                                                            model=self.model,
                                                                            patch_size=self.cfg_trainer['params'][
                                                                                'patch_size'],
                                                                            meta=self.meta,
                                                                            device=self.device,
                                                                            path=self.save_path,
                                                                            epoch=epoch)

        self.oa.update_scalar(acc)
        self.aa.update_scalar(acc_cls)
        self.kappa.update_scalar(kappa)

    def _save_checkpoint(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, 'checkpoint.pth'))

    def run(self, validation=True):
        self._train(1)
        if True:
            self._validation(1)
        else:
            self.oa.update_scalar(0)
            self.aa.update_scalar(0)
            self.kappa.update_scalar(0)

        logging_string = "OA {:.6f}, AA {:.6f}, Kappa {:.6f}".format(self.oa.scalar[-1], self.aa.scalar[-1],
                                                                     self.kappa.scalar[-1])
        logging.info(logging_string)

        self._save_checkpoint()

        self.loss_recorder.save_scalar_npy('loss_npy', self.save_path)
        self.loss_recorder.save_lineplot_fig('Loss', 'loss', self.save_path)
