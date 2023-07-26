for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/FocalLoss/1.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/FocalLoss/2.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/FocalLoss/3.py'
#CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/FocalLoss/5.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/FocalLoss/6.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/FocalLoss/14.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/FocalLoss/16.py'
done