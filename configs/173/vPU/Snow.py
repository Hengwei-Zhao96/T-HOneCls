config = dict(
    dataset=dict(
        train=dict(
            type='OneSevenThreeDataset',
            params=dict(
                image_path='./Data/173Data/Snow/snow.dat',
                # image_path='./Data/173Data/Snow/noise.dat',
                gt_path='./Data/173Data/Snow/Snow_unlabeled_gt',
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
                image_path='./Data/173Data/Snow/snow.dat',
                # image_path='./Data/173Data/Snow/noise.dat',
                gt_path='./Data/173Data/Snow/Snow_unlabeled_gt',
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
            in_channels=36,
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
        type='SingleModelTrainer',
        params=dict(
            max_iters=20,
            clip_grad=6
        ),
    ),
    meta=dict(
        save_path='Log/173',
        image_size=(624, 848),
        palette=[
            [0, 0, 0],
            [255, 0, 0]],
    )
)
