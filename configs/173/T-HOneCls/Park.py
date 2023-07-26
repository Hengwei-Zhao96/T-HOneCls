config = dict(
    dataset=dict(
        train=dict(
            type='OneSevenThreeDataset',
            params=dict(
                image_path='./Data/173Data/Park/park.dat',
                # image_path='./Data/173Data/Park/noise.dat',
                gt_path='./Data/173Data/Park/gt',
                train_flage=True,
                num_positive_train_samples=50,
                sub_minibatch=10,
                ccls=1,
                ratio=400
            )
        ),
        test=dict(
            type='OneSevenThreeDataset',
            params=dict(
                image_path='./Data/173Data/Park/park.dat',
                # image_path='./Data/173Data/Park/noise.dat',
                gt_path='./Data/173Data/Park/gt',
                train_flage=False,
                num_positive_train_samples=50,
                sub_minibatch=10,
                ccls=1,
                ratio=400
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
    loss_function=dict(
        type='TaylorVarPULossPf',
        params=dict(
            order=1,
        ),
    ),
    optimizer=dict(
        type='SGD',
        params=dict(
            lr=0.0003,
            momentum=0.9,
            weight_decay=0.0001
        ),
    ),
    lr_scheduler=dict(
        type='ExponentialLR',
        params=dict(
            gamma=1),
    ),
    trainer=dict(
        type='SelfCalibrationTrainer',
        params=dict(
            max_iters=200,
            clip_grad=6,
            beta=0.5,
            ema_model_alpha=0.99
        ),
    ),
    meta=dict(
        save_path='Log/173',
        image_size=(498, 714),
        palette=[
            [0, 0, 0],
            [255, 0, 0]],
    )
)
