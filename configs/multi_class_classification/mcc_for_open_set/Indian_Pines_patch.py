config = dict(
    dataset=dict(
        train=dict(
            type='MccIndianPinesDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/HSI/Indian_Pines/Indian_pines_corrected.mat',
                gt_path='/home/zhw2021/code/HOneCls/Data/HSI/Indian_Pines/new_mcc_indian_gt17.npy',
                train_flage=True,
                num_classes=8,
                num_train_samples_per_class=100,
                sub_minibatch=5
            )
        ),
        test=dict(
            type='MccIndianPinesDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/HSI/Indian_Pines/Indian_pines_corrected.mat',
                gt_path='/home/zhw2021/code/HOneCls/Data/HSI/Indian_Pines/new_mcc_indian_gt17.npy',
                train_flage=False,
                num_classes=8,
                num_train_samples_per_class=100,
                sub_minibatch=5
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
        type='CELossPb',
        params=dict(
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
        image_size=(145, 145),
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
