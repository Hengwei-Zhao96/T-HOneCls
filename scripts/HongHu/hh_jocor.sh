for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HongHu/JoCoR/4.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HongHu/JoCoR/6.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HongHu/JoCoR/7.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HongHu/JoCoR/9.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HongHu/JoCoR/10.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HongHu/JoCoR/11.py'
#CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HongHu/JoCoR/18.py'
#CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HongHu/JoCoR/19.py'
done