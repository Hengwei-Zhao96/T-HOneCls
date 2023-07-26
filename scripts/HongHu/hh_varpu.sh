for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/Training_samples_testing/4_vpu_40000.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/VarPU/6.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/VarPU/7.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/VarPU/9.py'
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/VarPU/10.py'
#CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/VarPU/11.py'
#CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/VarPU/18.py'
#CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/HongHu/VarPU/19.py'
done