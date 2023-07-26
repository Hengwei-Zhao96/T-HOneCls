for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/LongKou/CoTeaching/1.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/LongKou/CoTeaching/3.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/LongKou/CoTeaching/4.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/LongKou/CoTeaching/6.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/LongKou/CoTeaching/7.py'
done