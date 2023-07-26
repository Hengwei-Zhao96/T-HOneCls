for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/LongKou/ImbalancedPU/1.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/LongKou/ImbalancedPU/3.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/LongKou/ImbalancedPU/4.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/LongKou/ImbalancedPU/6.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/LongKou/ImbalancedPU/7.py'
done