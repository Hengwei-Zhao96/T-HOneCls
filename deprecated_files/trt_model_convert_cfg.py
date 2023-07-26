config = dict(
    dataset=dict(
        test=dict(
            type='HongHuDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-HongHu/data',
                gt_path='/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-HongHu/gt',
                train_flage=False,
                num_positive_train_samples=100,
                sub_minibatch=10,
                ccls=4,
                ratio=40
            )
        )
    ),
    model=dict(
        type='SimpleFreeOCNet',
        params=dict(
            in_channels=270,
            num_classes=1,
            block_channels=(64, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        )
    ),
    torch2trt=dict(
        type='torch2trt_dynamic',
        params=dict(
            checkpoints_path='/home/zhw2021/code/HOneCls/log/HongHuDataset/SelfCalibrationTrainer/TaylorVarPULossPf/SimpleFreeOCNet/4/2022-08-20 18-20-30/model_t/checkpoint_t.pth',
            opt_shape_param=[[
                [1, 270, 100, 100],
                [1, 270, 688, 480],
                [1, 270, 700, 700]
            ]],
            trt_model_save_path='./model_trt.pth',
            fp16_mode=True,

        )
    ),
    meta=dict(
        image_size=(678, 465),
        palette=[
            [0, 0, 0],
            [255, 0, 0],
            [255, 255, 255],
            [176, 48, 96],
            [255, 255, 0],
            [255, 127, 80],
            [0, 255, 0],
            [0, 205, 0],
            [0, 139, 0],
            [127, 255, 212],
            [160, 32, 240],
            [216, 191, 216],
            [0, 0, 255],
            [0, 0, 139],
            [218, 112, 214],
            [160, 82, 45],
            [0, 255, 255],
            [255, 165, 0],
            [127, 255, 0],
            [139, 139, 0],
            [0, 139, 139],
            [205, 181, 205],
            [238, 154, 0]],
    )
)
