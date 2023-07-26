for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/LongKou/FocalLoss/1.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/LongKou/FocalLoss/3.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/LongKou/FocalLoss/4.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/LongKou/FocalLoss/6.py'
#CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/LongKou/FocalLoss/7.py'
done