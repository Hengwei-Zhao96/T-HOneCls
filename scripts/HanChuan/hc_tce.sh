for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/TCE/TCE-o=6/1.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/TCE/TCE-o=6/2.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/TCE/TCE-o=6/3.py'
#CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/TCE/TCE-o=6/5.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/TCE/TCE-o=6/6.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/TCE/TCE-o=6/14.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/TCE/TCE-o=6/16.py'
done