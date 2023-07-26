for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/MSE/1.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/MSE/2.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/MSE/3.py'
#CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/MSE/5.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/MSE/6.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/MSE/14.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/MSE/16.py'
done