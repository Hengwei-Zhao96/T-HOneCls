for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/MAE/4.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/MAE/6.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/MAE/7.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/MAE/9.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/MAE/10.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/MAE/11.py'
#CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/MAE/18.py'
#CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/MAE/19.py'
done