for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HanChuan/CSPU/1.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HanChuan/CSPU/2.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HanChuan/CSPU/3.py'
#CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HanChuan/CSPU/5.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HanChuan/CSPU/6.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HanChuan/CSPU/14.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/HanChuan/CSPU/16.py'
done