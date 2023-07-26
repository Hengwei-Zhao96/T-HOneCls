for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/PAN/1.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/PAN/2.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/PAN/3.py'
#CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/PAN/5.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/PAN/6.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/PAN/14.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/PAN/16.py'
done