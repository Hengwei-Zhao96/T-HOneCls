config = dict(
    dataset=dict(
        train=dict(
            type='HanChuanDataset',
            params=dict(
                image_path='./Data/UAVData/WHU-Hi-HanChuan/WHU-Hi-HanChuan',
                gt_path='./Data/UAVData/WHU-Hi-HanChuan/Train100',
                train_flage=True,
                num_positive_train_samples=100,
                sub_minibatch=10,
                ccls=1,
                ratio=40
            )
        ),
        test=dict(
            type='HanChuanDataset',
            params=dict(
                image_path='./Data/UAVData/WHU-Hi-HanChuan/WHU-Hi-HanChuan',
                gt_path='./Data/UAVData/WHU-Hi-HanChuan/Test100',
                train_flage=False,
                num_positive_train_samples=100,
                sub_minibatch=10,
                ccls=1,
                ratio=40
            )
        )
    ),
    model=dict(
        type='FreeOCNet',
        params=dict(
            in_channels=274,
            num_classes=1,
            block_channels=(64, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        )
    ),
    loss_function=dict(
        type='VarPULossPf',
        params=dict(
        ),
    ),
    optimizer=dict(
        type='SGD',
        params=dict(
            lr=0.0002,
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
        type='SingleModelTrainer',
        params=dict(
            max_iters=170,
            clip_grad=None,
        ),
    ),
    meta=dict(
        save_path='Log/T-HOneCls',
        image_size=(1217, 303),
        palette=[
            [0, 0, 0],
            [176, 48, 96],
            [0, 255, 255],
            [255, 0, 255],
            [160, 32, 240],
            [127, 255, 212],
            [127, 255, 0],
            [0, 205, 0],
            [0, 255, 0],
            [0, 139, 0],
            [255, 0, 0],
            [216, 191, 216],
            [255, 127, 80],
            [160, 82, 45],
            [255, 255, 255],
            [218, 112, 214],
            [0, 0, 255]],
    ),
)
