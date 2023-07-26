for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/AsyPU/4.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/AsyPU/6.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/AsyPU/7.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/AsyPU/9.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/AsyPU/10.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/AsyPU/11.py'
#CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/AsyPU/18.py'
#CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/AsyPU/19.py'
done