config = dict(
    dataset=dict(
        train=dict(
            type='IndianPinesDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/HSI/Indian_Pines/Indian_pines_corrected.mat',
                gt_path='/home/zhw2021/code/HOneCls/Data/HSI/Indian_Pines/Indian_pines_gt.mat',
                train_flage=True,
                num_positive_train_samples=100,
                sub_minibatch=10,
                ccls=11,
                ratio=40
            )
        ),
        test=dict(
            type='IndianPinesDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/HSI/Indian_Pines/Indian_pines_corrected.mat',
                gt_path='/home/zhw2021/code/HOneCls/Data/HSI/Indian_Pines/Indian_pines_gt.mat',
                train_flage=False,
                num_positive_train_samples=100,
                sub_minibatch=10,
                ccls=11,
                ratio=40
            )
        )
    ),
    model=dict(
        type='FreeOCNet',
        params=dict(
            in_channels=200,
            num_classes=1,
            block_channels=(64, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        )
    ),
    loss_function=dict(
        type='TaylorVarPULossPf',
        params=dict(
            order=2,
        ),
    ),
    optimizer=dict(
        type='SGD',
        params=dict(
            lr=0.0001,
            momentum=0.9,
            weight_decay=0.0001
        ),
    ),
    lr_scheduler=dict(
        type='ExponentialLR',
        params=dict(
            gamma=0.995),
    ),
    trainer=dict(
        type='SelfCalibrationTrainer',
        params=dict(
            max_iters=150,
            clip_grad=6,
            beta=0.5,
            ema_model_alpha=0.99
        ),
    ),
    meta=dict(
        save_path='Log/T-HOneCls',
        image_size=(145, 145),
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
