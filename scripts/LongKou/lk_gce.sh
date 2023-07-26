for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/LongKou/GCE/GCE-q=0.3/1.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/LongKou/GCE/GCE-q=0.3/3.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/LongKou/GCE/GCE-q=0.3/4.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/LongKou/GCE/GCE-q=0.3/6.py'
#CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/LongKou/GCE/GCE-q=0.3/7.py'
done