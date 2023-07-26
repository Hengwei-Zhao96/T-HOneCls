for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/TCE/TCE-o=6/4.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/TCE/TCE-o=6/6.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/TCE/TCE-o=6/7.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/TCE/TCE-o=6/9.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/TCE/TCE-o=6/10.py'
#CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/TCE/TCE-o=6/11.py'
#CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/TCE/TCE-o=6/18.py'
#CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/TCE/TCE-o=6/19.py'
done