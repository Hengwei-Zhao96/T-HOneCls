for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/AsyPU/1.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/AsyPU/2.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/AsyPU/3.py'
#CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/AsyPU/5.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/AsyPU/6.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/AsyPU/14.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/AsyPU/16.py'
done