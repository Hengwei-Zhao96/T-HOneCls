config = dict(
    dataset=dict(
        train=dict(
            type='OsMccHanChuanDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-HanChuan/WHU-Hi-HanChuan',
                gt_path='/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-HanChuan/Train100_new_label.npy',
                train_flage=True,
                num_classes=6,
                num_train_samples_per_class=100,
                sub_minibatch=5,
                num_unlabeled_samples=4000
            )
        ),
        test=dict(
            type='OsMccHanChuanDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-HanChuan/WHU-Hi-HanChuan',
                gt_path='/home/zhw2021/code/HOneCls/Data/UAVData/WHU-Hi-HanChuan/Test100_new_open_label.npy',
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
            in_channels=274,
            out_channels=6,
            patch_size=9,
        )
    ),
    loss_function=dict(
        type='None',
        params=dict(),
    ),
    optimizer=dict(
        type='None',
        params=dict(),
    ),
    lr_scheduler=dict(
        type='None',
        params=dict(),
    ),
    trainer=dict(
        type='OsOpenMaxTrainer_PB',
        params=dict(
            detector_name='OpenMax',
            checkpoint_path="/home/zhw2021/code/HOneCls/Log/MCC_for_Open_Set/MccHanChuanDataset/MccSingleModelTrainer_PB/CELossPb/FreeNetEncoder/2023-06-15 14-02-29/checkpoint.pth",
            tailsize=25,
            alpha=5,
            euclid_weight=0.5,
            patch_size=9,
            batch_size_pb=256,
        ),
    ),
    meta=dict(
        save_path='./Log/OpenSet',
        image_size=(1217, 303),
        palette_class_mapping={1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 0},
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
