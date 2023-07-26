for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/T-HOneCls/T-HOneCls-beta=0/1.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/T-HOneCls/T-HOneCls-beta=0/2.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/T-HOneCls/T-HOneCls-beta=0/3.py'
#CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/T-HOneCls/T-HOneCls-beta=0/5.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/T-HOneCls/T-HOneCls-beta=0/6.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/T-HOneCls/T-HOneCls-beta=0/14.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/T-HOneCls/T-HOneCls-beta=0/16.py'
done