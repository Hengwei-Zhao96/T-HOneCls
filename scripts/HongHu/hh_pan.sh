for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HongHu/PAN/4.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HongHu/PAN/6.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HongHu/PAN/7.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HongHu/PAN/9.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HongHu/PAN/10.py'
#CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HongHu/PAN/11.py'
#CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HongHu/PAN/18.py'
#CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HongHu/PAN/19.py'
done