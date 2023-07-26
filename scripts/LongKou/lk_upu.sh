for repeat in 1
do
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/LongKou/uPU/1.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/LongKou/uPU/3.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/LongKou/uPU/4.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/LongKou/uPU/6.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/LongKou/uPU/7.py'
done