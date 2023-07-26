config = dict(
    dataset=dict(
        train=dict(
            type='OsMccIndianPinesDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/HSI/Indian_Pines/Indian_pines_corrected.mat',
                gt_path='/home/zhw2021/code/HOneCls/Data/HSI/Indian_Pines/new_indian_gt17.npy',
                train_flage=True,
                num_classes=8,
                num_train_samples_per_class=100,
                sub_minibatch=5,
                num_unlabeled_samples=4000
            )
        ),
        test=dict(
            type='OsMccIndianPinesDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/HSI/Indian_Pines/Indian_pines_corrected.mat',
                gt_path='/home/zhw2021/code/HOneCls/Data/HSI/Indian_Pines/new_indian_gt17.npy',
                train_flage=False,
                num_classes=9,
                num_train_samples_per_class=100,
                sub_minibatch=5,
                num_unlabeled_samples=4000
            )
        )
    ),
    model=dict(
        type='FreeNetEncoder',
        params=dict(
            in_channels=200,
            out_channels=8,
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
        type='OsPOODDetectorTrainer_PB',
        params=dict(
            detector_name='EnergyBased',
            # MCD, MaxSoftmax, Mahalanobis, EnergyBased, Entropy, MaxLogit, ODIN, KLMatching
            checkpoint_path="/home/zhw2021/code/HOneCls/Log/MCC_for_Open_Set/MccIndianPinesDataset/MccSingleModelTrainer_PB/CELossPb/FreeNetEncoder/2023-06-15 13-36-35/checkpoint.pth",
            patch_size=9,
            batch_size_pb=256,
        ),
    ),
    meta=dict(
        save_path='./Log/OpenSet',
        image_size=(145, 145),
        palette_class_mapping={1: 2, 2: 3, 3: 5, 4: 8, 5: 10, 6: 11, 7: 12, 8: 14, 9: 0},
        palette=[
            [0, 0, 0],
            [255, 252, 134],
            [0, 55, 243],
            [255, 93, 0],
            [0, 251, 132],
            [255, 58, 252],
            [74, 50, 255],
            [0, 173, 255],
            [0, 250, 0],
            [174, 173, 81],
            [162, 84, 158],
            [84, 176, 255],
            [55, 91, 112],
            [101, 189, 60],
            [143, 70, 44],
            [108, 252, 171],
            [255, 252, 0]
        ],
    )
)
