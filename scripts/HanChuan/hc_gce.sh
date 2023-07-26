for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/GCE/GCE-q=0.3/1.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/GCE/GCE-q=0.3/2.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/GCE/GCE-q=0.3/3.py'
#CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/GCE/GCE-q=0.3/5.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/GCE/GCE-q=0.3/6.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/GCE/GCE-q=0.3/14.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/GCE/GCE-q=0.3/16.py'
done