for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/Var+Self/2.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/Var+Self/4.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/Var+Self/4lk.py'
done