import argparse
import sys
import os

sys.path.append(os.path.abspath('.'))

import torch
import torch.onnx
from torch2trt_dynamic import torch2trt_dynamic

from HOC.apis.build_trainer import build_from_cfg
from HOC.apis.validation import fcn_evaluate_fn
from HOC.datasets.build_dataloader import HOCDataLoader
from HOC.utils import read_config, DATASETS, MODELS


def Argparse():
    parser = argparse.ArgumentParser(description='HOC Inference')
    parser.add_argument('-c', '--cfg', type=str,
                        default='/home/zhw2021/code/HOneCls/configs/cfg_demos/trt/trt_model_convert.py',
                        help='File path of config')
    return parser.parse_args()


if __name__ == "__main__":
    args = Argparse()
    config = read_config(args.cfg)

    model = build_from_cfg(config['model'], MODELS).cuda()
    model.load_state_dict(torch.load(config['torch2trt']['params']['checkpoints_path']))
    model.eval()
    analog_input = torch.ones(config['torch2trt']['params']['opt_shape_param'][0][1]).cuda()
    print('optimize start')
    model_trt = torch2trt_dynamic(model, [analog_input], fp16_mode=config['torch2trt']['params']['fp16_mode'],
                                  opt_shape_param=config['torch2trt']['params']['opt_shape_param'],
                                  max_workspace_size=20000)
    print('optimize over')
    torch.save(model_trt.state_dict(), config['torch2trt']['params']['trt_model_save_path'])
    print('trt model has saved!')
