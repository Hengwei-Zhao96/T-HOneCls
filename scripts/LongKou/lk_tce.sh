for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/LongKou/TCE/TCE-o=6/1.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/LongKou/TCE/TCE-o=6/3.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/LongKou/TCE/TCE-o=6/4.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/LongKou/TCE/TCE-o=6/6.py'
#CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/LongKou/TCE/TCE-o=6/7.py'
done