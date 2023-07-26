config = dict(
    dataset=dict(
        train=dict(
            type='MccHanChuanDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-HanChuan/WHU-Hi-HanChuan',
                gt_path='/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-HanChuan/Train100_new_label.npy',
                train_flage=True,
                num_classes=6,
                num_train_samples_per_class=100,
                sub_minibatch=20
            )
        ),
        test=dict(
            type='MccHanChuanDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-HanChuan/WHU-Hi-HanChuan',
                gt_path='/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-HanChuan/Train100_new_label.npy',
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
            in_channels=274,
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
            [0, 0, 255]
        ],
    )
)
