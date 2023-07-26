for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HongHu/CSPU/4.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HongHu/CSPU/6.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HongHu/CSPU/7.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HongHu/CSPU/9.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HongHu/CSPU/10.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HongHu/CSPU/11.py'
#CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HongHu/CSPU/18.py'
#CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HongHu/CSPU/19.py'
done