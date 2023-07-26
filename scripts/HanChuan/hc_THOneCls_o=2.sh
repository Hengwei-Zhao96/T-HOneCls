for repeat in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/T-HOneCls/T-HOneCls-o=2/2.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/T-HOneCls/T-HOneCls-o=2/2.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/T-HOneCls/T-HOneCls-o=2/3.py'
##CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/T-HOneCls/T-HOneCls-o=2/5.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/T-HOneCls/T-HOneCls-o=2/6.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/T-HOneCls/T-HOneCls-o=2/14.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/HanChuan/T-HOneCls/T-HOneCls-o=2/16.py'
done