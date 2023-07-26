for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/ImbalancedPU/1.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/ImbalancedPU/2.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/ImbalancedPU/3.py'
#CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/ImbalancedPU/5.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/ImbalancedPU/6.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/ImbalancedPU/14.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/HanChuan/ImbalancedPU/16.py'
done