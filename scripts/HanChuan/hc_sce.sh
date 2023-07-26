for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/SCE/1.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/SCE/2.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/SCE/3.py'
#CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/SCE/5.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/SCE/6.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/SCE/14.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/SCE/16.py'
done