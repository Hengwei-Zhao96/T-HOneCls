for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/LongKou/T-HOneCls/T-HOneCls-beta=0/1.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/LongKou/T-HOneCls/T-HOneCls-beta=0/3.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/LongKou/T-HOneCls/T-HOneCls-beta=0/4.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/LongKou/T-HOneCls/T-HOneCls-beta=0/6.py'
#CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/LongKou/T-HOneCls/T-HOneCls-beta=0/7.py'
done