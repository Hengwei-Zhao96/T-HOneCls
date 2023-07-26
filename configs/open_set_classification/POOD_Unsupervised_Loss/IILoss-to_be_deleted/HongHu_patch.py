config = dict(
    dataset=dict(
        train=dict(
            type='OsMccHongHuDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-HongHu/WHU-Hi-HongHu',
                gt_path='/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-HongHu/Train100_new_label.npy',
                train_flage=True,
                num_classes=17,
                num_train_samples_per_class=100,
                sub_minibatch=5,
                num_unlabeled_samples=4000
            )
        ),
        test=dict(
            type='OsMccHongHuDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-HongHu/WHU-Hi-HongHu',
                gt_path='/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-HongHu/Test100_new_open_label.npy',
                train_flage=False,
                num_classes=18,
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
            out_channels=17,
            patch_size=9,
        )
    ),
    loss_function=dict(
        type='IILoss',
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
        type='OsIILossTrainer_PB',
        params=dict(
            max_iters=500,
            n_classes=17,
            n_embedding=17,
            alpha=2,
            patch_size=9,
            batch_size_pb=256,
        ),
    ),
    meta=dict(
        save_path='./Log/OpenSet',
        image_size=(940, 475),
        palette_class_mapping={1: 4, 2: 6, 3: 7, 4: 8, 5: 9, 6: 10, 7: 11, 8: 12, 9: 13, 10: 14, 11: 15, 12: 16, 13: 17,
                               14: 18, 15: 19, 16: 20, 17: 21, 18: 0},
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
            [238, 154, 0]
        ],
    )
)
