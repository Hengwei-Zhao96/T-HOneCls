for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HongHu/CoTeaching/4.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HongHu/CoTeaching/6.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HongHu/CoTeaching/7.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HongHu/CoTeaching/9.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HongHu/CoTeaching/10.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HongHu/CoTeaching/11.py'
#CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HongHu/CoTeaching/18.py'
#CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HongHu/CoTeaching/19.py'
done