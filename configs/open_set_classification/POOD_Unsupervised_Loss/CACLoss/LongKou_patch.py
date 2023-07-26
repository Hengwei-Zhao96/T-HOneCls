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
        type='FreeNetEncoder',
        params=dict(
            in_channels=270,
            out_channels=6,
            patch_size=9,
        )
    ),
    loss_function=dict(
        type='CACLoss',
        params=dict(),
    ),
    optimizer=dict(
        type='Adam',
        params=dict(
            lr=0.001,
            # momentum=0.9,
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
        type='OsCACLossTrainer_PB',
        params=dict(
            max_iters=500,
            n_classes=6,
            magnitude=5,
            alpha=2,
            patch_size=9,
            batch_size_pb=256,
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
