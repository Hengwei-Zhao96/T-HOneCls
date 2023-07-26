import argparse
import sys
import os

sys.path.append(os.path.abspath('.'))

import torch

from HOC.apis.build_trainer import build_from_cfg
from HOC.apis.validation import fcn_inference_fn,fcn_evaluate_fn
from HOC.datasets.build_dataloader import HOCDataLoader
from HOC.utils import read_config, DATASETS, MODELS


def Argparse():
    parser = argparse.ArgumentParser(description='HOC Inference')
    parser.add_argument('-c', '--cfg', type=str,
                        default='/home/zhw2021/code/HOneCls/configs/cfg_demos/inference.py',
                        help='File path of config')
    return parser.parse_args()


if __name__ == "__main__":
    args = Argparse()
    config = read_config(args.cfg)

    test_pf_dataloader = HOCDataLoader(build_from_cfg(config['dataset']['test'], DATASETS))
    model = build_from_cfg(config['model'], MODELS).cuda()
    model.load_state_dict(torch.load(config['inference']['params']['checkpoint_path']))
    print('Model has be loaded!')

    auc_s, fpr_s, tpr_s, threshold_s, pre_s, rec_s, f1_s =fcn_evaluate_fn(
        model=model,
        test_dataloader=test_pf_dataloader,
        meta=config['meta'],
        cls=config['dataset']['test']['params']['ccls'],
        device='cuda:0',
        path=config['inference']['params']['save_path'],
        epoch=0)
    print(f1_s)
