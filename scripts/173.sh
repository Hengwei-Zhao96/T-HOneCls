CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/173/T-HOneCls/Airplane.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/173/T-HOneCls/Forest.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/173/T-HOneCls/Park.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/173/T-HOneCls/Snow.py'

python tools/train.py -c 'configs/173/PXL/Airplane.py'
python tools/train.py -c 'configs/173/PXL/Forest.py'
python tools/train.py -c 'configs/173/PXL/Park.py'
python tools/train.py -c 'configs/173/PXL/Snow.py'

python tools/class_prior_estimation.py -c 'configs/173/KMPE/Airplane.py'
python tools/class_prior_estimation.py -c 'configs/173/KMPE/Forest.py'
python tools/class_prior_estimation.py -c 'configs/173/KMPE/Park.py'
python tools/class_prior_estimation.py -c 'configs/173/KMPE/Snow.py'

CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/173/CNNnnPU/Airplane.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/173/CNNnnPU/Forest.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/173/CNNnnPU/Park.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/173/CNNnnPU/Snow.py'

CUDA_VISIBLE_DEVICES=0 python tools/train.py -c 'configs/173/vPU/Airplane.py'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c 'configs/173/vPU/Forest.py'
CUDA_VISIBLE_DEVICES=2 python tools/train.py -c 'configs/173/vPU/Park.py'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -c 'configs/173/vPU/Snow.py'

python tools/train.py -c 'configs/173/OCSVM/Airplane.py'
python tools/train.py -c 'configs/173/OCSVM/Forest.py'
python tools/train.py -c 'configs/173/OCSVM/Park.py'
python tools/train.py -c 'configs/173/OCSVM/Snow.py'

CUDA_VISIBLE_DEVICES=0 python tools/inference.py -c 'configs/173/T-HOneCls/173inference_forest.py'
CUDA_VISIBLE_DEVICES=0 python tools/inference.py -c 'configs/173/T-HOneCls/173inference_park.py'
CUDA_VISIBLE_DEVICES=0 python tools/inference.py -c 'configs/173/T-HOneCls/173inference_plane.py'
CUDA_VISIBLE_DEVICES=0 python tools/inference.py -c 'configs/173/T-HOneCls/173inference_snow.py'