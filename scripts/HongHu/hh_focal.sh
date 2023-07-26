for repeat in 1 2 3 4 5
do
#CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/FocalLoss/4.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/FocalLoss/6.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/FocalLoss/7.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/FocalLoss/9.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/FocalLoss/10.py'
#CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/FocalLoss/11.py'
#CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/FocalLoss/18.py'
#CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/FocalLoss/19.py'
done