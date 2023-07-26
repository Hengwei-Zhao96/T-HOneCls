config = dict(
    dataset=dict(
        train=dict(
            type='MccLongKouDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-LongKou/WHU-Hi-LongKou',
                gt_path='/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-LongKou/Train100_new_label.npy',
                train_flage=True,
                num_classes=6,
                num_train_samples_per_class=100,
                sub_minibatch=20
            )
        ),
        test=dict(
            type='MccLongKouDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-LongKou/WHU-Hi-LongKou',
                gt_path='/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-LongKou/Train100_new_label.npy',
                train_flage=False,
                num_classes=6,
                num_train_samples_per_class=100,
                sub_minibatch=20
            )
        )
    ),
    model=dict(
        type='FreeNetEncoder',
        params=dict(
            in_channels=270,
            out_channels=6,
            patch_size=9,
        )
    ),
    loss_function=dict(
        type='CELossPb',
        params=dict(),
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
        type='MccSingleModelTrainer_PB',
        params=dict(
            max_iters=300,
            clip_grad=None,
            patch_size=9,
            batch_size_pb=256,
        ),
    ),
    meta=dict(
        save_path='./Log/MCC_for_Open_Set',
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
            [160, 32, 240]
        ],
    )
)
