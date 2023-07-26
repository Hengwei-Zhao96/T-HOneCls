for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HongHu/T-HOneCls/T-HOneCls-beta=0/4.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HongHu/T-HOneCls/T-HOneCls-beta=0/6.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HongHu/T-HOneCls/T-HOneCls-beta=0/7.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HongHu/T-HOneCls/T-HOneCls-beta=0/9.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HongHu/T-HOneCls/T-HOneCls-beta=0/10.py'
#CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HongHu/T-HOneCls/T-OneCls-o=2/11.py'
#CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HongHu/T-HOneCls/T-OneCls-o=2/18.py'
#CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HongHu/T-HOneCls/T-OneCls-o=2/19.py'
done