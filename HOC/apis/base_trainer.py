from abc import ABC, abstractmethod
import os
import shutil

import torch

from HOC.utils.logging_tool import basic_logging
from HOC.utils.scalar_recorder import ScalarRecorder


class BaseTrainer(ABC):
    def __init__(self, trainer, dataset, model, loss_function, optimizer, lr_scheduler, meta):
        self.cfg_trainer = trainer
        self.cfg_dataset = dataset
        self.cfg_model = model
        self.cfg_loss_function = loss_function
        self.cfg_optimizer = optimizer
        self.cfg_lr_scheduler = lr_scheduler
        self.meta = meta

        self.recoder = dict()
        self._build_device()
        self._build_components()
        self._build_log()
        self._copy_config()

    def _build_scalar_recoder(self):
        return ScalarRecorder()

    def _build_device(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _build_log(self):
        if self.cfg_trainer['type'] == 'SklearnTrainer':
            folder_name = os.path.join(os.path.abspath('.'),
                                       self.meta['save_path'],
                                       self.cfg_dataset['train']['type'],
                                       self.cfg_trainer['type'],
                                       self.cfg_model['type'],
                                       str(self.cfg_dataset['train']['params']['ccls']))
        elif self.cfg_trainer['type'] in ['MccSingleModelTrainer', 'OsMccSingleModelTrainer',
                                          'MccSingleModelTrainer_PB', 'OsCACLossTrainer_PB',
                                          'OsIILossTrainer_PB', 'OsSelfCalibrationTrainer', 'OsDS3LTrainer_PB']:
            folder_name = os.path.join(os.path.abspath('.'),
                                       self.meta['save_path'],
                                       self.cfg_dataset['train']['type'],
                                       self.cfg_trainer['type'],
                                       self.cfg_loss_function['type'],
                                       self.cfg_model['type'])
        elif self.cfg_trainer['type'] in ['OsOpenMaxTrainer_PB', 'OsPOODDetectorTrainer_PB']:
            folder_name = os.path.join(os.path.abspath('.'),
                                       self.meta['save_path'],
                                       self.cfg_dataset['train']['type'],
                                       self.cfg_trainer['type'],
                                       self.cfg_model['type'],
                                       self.cfg_trainer['params']['detector_name'])
        elif self.cfg_trainer['type'] in ['OsPOODSupervisedLossTrainer_PB']:
            folder_name = os.path.join(os.path.abspath('.'),
                                       self.meta['save_path'],
                                       self.cfg_dataset['train']['type'],
                                       self.cfg_trainer['type'],
                                       self.cfg_loss_function['type'],
                                       self.cfg_trainer['params']['detector_name'],
                                       self.cfg_model['type'])
        else:
            folder_name = os.path.join(os.path.abspath('.'),
                                       self.meta['save_path'],
                                       self.cfg_dataset['train']['type'],
                                       self.cfg_trainer['type'],
                                       self.cfg_loss_function['type'],
                                       self.cfg_model['type'],
                                       str(self.cfg_dataset['train']['params']['ccls']))

        if self.cfg_model['type'] == 'SMPModelGn':
            folder_name = os.path.join(folder_name,
                                       self.cfg_model['params']['model_network'],
                                       self.cfg_model['params']['encoder_name'], )

        self.save_path = basic_logging(folder_name)
        print("The save path is:", self.save_path)

    def _copy_config(self):
        config_name = self.meta['config_path'].split('/')[-1]
        shutil.copyfile(self.meta['config_path'], os.path.join(self.save_path, config_name))

    def clip_recoder_grad_norm(self, model, grad_l2_norm):
        if not (self.cfg_trainer['params']['clip_grad'] is None):
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg_trainer['params']['clip_grad'])

        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0) for p in model.parameters()]), 2.0)
        grad_l2_norm.update_scalar(total_norm.item())

    @abstractmethod
    def _build_train_dataloader(self):
        pass

    @abstractmethod
    def _build_model(self):
        pass

    @abstractmethod
    def _build_loss_function(self):
        pass

    @abstractmethod
    def _build_optimizer(self):
        pass

    @abstractmethod
    def _build_lr_scheduler(self):
        pass

    @abstractmethod
    def _build_components(self):
        pass

    @abstractmethod
    def _train(self, epoch):
        pass

    @abstractmethod
    def _validation(self, epoch):
        pass

    @abstractmethod
    def _save_checkpoint(self):
        pass

    @abstractmethod
    def run(self, validation=True):
        pass
