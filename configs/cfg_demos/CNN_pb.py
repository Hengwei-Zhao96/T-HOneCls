config = dict(
    dataset=dict(
        train=dict(
            type='HongHuDataset',
            params=dict(
                image_path='./Data/UAVData/WHU-Hi-HongHu/data',
                gt_path='./Data/UAVData/WHU-Hi-HongHu/gt',
                train_flage=True,
                num_positive_train_samples=100,
                sub_minibatch=1,
                cls=4,
                ratio=40
            )
        ),
        test=dict(
            type='HongHuDataset',
            params=dict(
                image_path='./Data/UAVData/WHU-Hi-HongHu/data',
                gt_path='./Data/UAVData/WHU-Hi-HongHu/gt',
                train_flage=False,
                num_positive_train_samples=100,
                sub_minibatch=1,
                cls=4,
                ratio=40
            )
        )
    ),
    model=dict(
        type='FreeNetEncoder',
        params=dict(
            in_channels=270,
            out_channels=1,
            patch_size=15,
        )
    ),
    loss_function=dict(
        type='NnPULossPb',
        params=dict(
            prior=0.370233702337023,
            beta=None,
            gamma=None
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
            gamma=1),
    ),
    trainer=dict(
        type='SingleModelTrainer_PB',
        params=dict(
            max_iters=5,
            clip_grad=6,
            patch_size=15,
            batch_size_pb=1024,
        ),
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
