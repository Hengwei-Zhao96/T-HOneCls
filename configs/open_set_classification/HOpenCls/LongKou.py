config = dict(
    dataset=dict(
        train=dict(
            type='OsMccLongKouDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-LongKou/WHU-Hi-LongKou',
                gt_path='/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-LongKou/Train100_new_label.npy',
                train_flage=True,
                num_classes=6,
                num_train_samples_per_class=100,
                sub_minibatch=5,
                num_unlabeled_samples=4000
            )
        ),
        test=dict(
            type='OsMccLongKouDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-LongKou/WHU-Hi-LongKou',
                gt_path='/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-LongKou/Test100_new_open_label.npy',
                train_flage=False,
                num_classes=7,
                num_train_samples_per_class=100,
                sub_minibatch=5,
                num_unlabeled_samples=4000
            )
        )
    ),
    model=dict(
        type='OSFreeNet',
        params=dict(
            in_channels=270,
            num_classes=6,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        )
    ),
    loss_function=dict(
        type='OSCELossPf',
        params=dict(
            order=1,
            num_classes=6,
        ),
    ),
    optimizer=dict(
        type='SGD',
        params=dict(
            lr=0.0001,
            momentum=0.9,
            weight_decay=0.001
        ),
    ),
    lr_scheduler=dict(
        type='ExponentialLR',
        params=dict(
            gamma=1,
        ),
    ),
    trainer=dict(
        type='OsSelfCalibrationTrainer',
        params=dict(
            max_iters=50,
            clip_grad=None,
            beta=1,
            ema_model_alpha=0.99
        ),
    ),
    meta=dict(
        save_path='./Log/OpenSet',
        image_size=(550, 400),
        palette_class_mapping={1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 0},
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
            [160, 32, 240]
        ],
    )
)
