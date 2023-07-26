config = dict(
    dataset=dict(
        train=dict(
            type='LongKouDataset',
            params=dict(
                image_path='./Data/UAVData/WHU-Hi-LongKou/WHU-Hi-LongKou',
                gt_path='./Data/UAVData/WHU-Hi-LongKou/LKTrain100',
                train_flage=True,
                num_positive_train_samples=100,
                sub_minibatch=10,
                ccls=3,
                ratio=40
            )
        ),
        test=dict(
            type='LongKouDataset',
            params=dict(
                image_path='./Data/UAVData/WHU-Hi-LongKou/WHU-Hi-LongKou',
                gt_path='./Data/UAVData/WHU-Hi-LongKou/LKTest100',
                train_flage=False,
                num_positive_train_samples=100,
                sub_minibatch=10,
                ccls=3,
                ratio=40
            )
        )
    ),
    model=dict(
        type='FreeOCNet',
        params=dict(
            in_channels=270,
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
            lr=0.0001,
            momentum=0.9,
            weight_decay=0.0001
        ),
    ),
    lr_scheduler=dict(
        type='ExponentialLR',
        params=dict(
            gamma=0.99),
    ),
    trainer=dict(
        type='SingleModelTrainer',
        params=dict(
            max_iters=150,
            clip_grad=None,
        ),
    ),
    meta=dict(
        save_path='Log/T-HOneCls',
        image_size=(550, 400),
        palette=[
            [0, 0, 0],
            [255, 0, 0],
            [238, 154, 0],
            [255, 255, 0],
            [0, 255, 0],
            [0, 255, 255],
            [0, 139, 139],
            [0, 0, 255],
            [255, 255, 255],
            [160, 32, 240]],
    ),
)
