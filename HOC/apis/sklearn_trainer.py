import logging
import os

import numpy as np
import torch

from HOC.apis import BaseTrainer
from HOC.apis.validation import sklearn_evaluate_fn
from HOC.datasets import build_dataloader
from HOC.models import build_model
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
class SklearnTrainer(BaseTrainer):
    def __init__(self, trainer, dataset, model, meta, loss_function=None, optimizer=None, lr_scheduler=None):
        super(SklearnTrainer, self).__init__(trainer=trainer, dataset=dataset, model=model,
                                             loss_function=loss_function,
                                             optimizer=optimizer, lr_scheduler=lr_scheduler, meta=meta)
        self.build_scalar_recoder()

    def _build_train_dataloader(self):
        train_pf_dataloader = build_dataloader(self.cfg_dataset['train'])

        image = train_pf_dataloader.dataset.im
        positive_train_indicator = train_pf_dataloader.dataset.positive_train_indicator
        unlabeled_train_indicator = train_pf_dataloader.dataset.unlabeled_train_indicator

        train_positive_id_x, train_positive_id_y = np.where(positive_train_indicator == 1)
        train_unlabeled_id_x, train_unlabeled_id_y = np.where(unlabeled_train_indicator == 1)

        positive_train_data = _get_patch_data(image, train_positive_id_x, train_positive_id_y, patch_size=1)
        unlabeled_train_data = _get_patch_data(image, train_unlabeled_id_x, train_unlabeled_id_y, patch_size=1)

        return image, positive_train_data, unlabeled_train_data

    def _build_validation_dataloader(self):
        validation_pf_dataloader = build_dataloader(self.cfg_dataset['test'])
        positive_validation_indicator = validation_pf_dataloader.dataset.positive_test_indicator
        negative_validation_indicator = validation_pf_dataloader.dataset.negative_test_indicator
        return positive_validation_indicator, negative_validation_indicator

    def _build_model(self):
        return build_model(self.cfg_model)

    def _build_loss_function(self):
        pass

    def _build_optimizer(self):
        pass

    def _build_lr_scheduler(self):
        pass

    def _build_components(self):
        self.image, self.positive_train_data, self.unlabeled_train_data = self._build_train_dataloader()
        self.positive_validation_indicator, self.negative_validation_indicator = self._build_validation_dataloader()
        self.model = self._build_model()

    def build_scalar_recoder(self):
        self.f1_recorder = self._build_scalar_recoder()
        self.pre_recorder = self._build_scalar_recoder()
        self.rec_recorder = self._build_scalar_recoder()

    def _train(self, epoch):
        self.model.fit(self.positive_train_data, self.unlabeled_train_data)

    def _validation(self, epoch):
        auc, fpr, tpr, threshold, pre, rec, f1 = sklearn_evaluate_fn(image=self.image,
                                                                     positive_test_indicator=self.positive_validation_indicator,
                                                                     negative_test_indicator=self.negative_validation_indicator,
                                                                     cls=self.cfg_dataset['train']['params']['ccls'],
                                                                     model=self.model,
                                                                     meta=self.meta,
                                                                     path=self.save_path,
                                                                     epoch=epoch)

        self.pre_recorder.update_scalar(pre)
        self.rec_recorder.update_scalar(rec)
        self.f1_recorder.update_scalar(f1)

        auc_roc = dict()
        auc_roc['fpr'] = fpr
        auc_roc['tpr'] = tpr
        auc_roc['threshold'] = threshold
        auc_roc['auc'] = auc
        np.save(os.path.join(self.save_path, 'auc_roc.npy'), auc_roc)

    def _save_checkpoint(self):
        pass

    def run(self, validation=True):
        self._train(epoch=0)

        if validation:
            self._validation(epoch=0)
        else:
            self.pre_recorder.update_scalar(0)
            self.rec_recorder.update_scalar(0)
            self.f1_recorder.update_scalar(0)

        logging_string = "{} epoch, loss {:.4f}, p_loss {:.4f}, u_loss {:.4f}, P {:.6f}, R {:.6f}, F1 {:.6f}".format(
            0,
            0,
            0,
            0,
            self.pre_recorder.scalar[-1],
            self.rec_recorder.scalar[-1],
            self.f1_recorder.scalar[-1])

        print(logging_string)
        logging.info(logging_string)

        if validation:
            self.f1_recorder.save_scalar_npy('f1_npy', self.save_path)
            self.f1_recorder.save_lineplot_fig('F1-score', 'f1-score', self.save_path)
            self.rec_recorder.save_scalar_npy('recall_npy', self.save_path)
            self.rec_recorder.save_lineplot_fig('Recall', 'recall', self.save_path)
            self.pre_recorder.save_scalar_npy('precision_npy', self.save_path)
            self.pre_recorder.save_lineplot_fig('Precision', 'precision', self.save_path)
