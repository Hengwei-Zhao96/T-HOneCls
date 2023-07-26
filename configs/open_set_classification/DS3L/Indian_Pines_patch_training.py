config = dict(
    dataset=dict(
        train=dict(
            type='OsMccIndianPinesDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/HSI/Indian_Pines/Indian_pines_corrected.mat',
                gt_path='/home/zhw2021/code/HOneCls/Data/HSI/Indian_Pines/new_mcc_indian_gt17.npy',
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
                gt_path='/home/zhw2021/code/HOneCls/Data/HSI/Indian_Pines/new_mcc_indian_gt17.npy',
                train_flage=False,
                num_classes=8,
                num_train_samples_per_class=100,
                sub_minibatch=5,
                num_unlabeled_samples=4000
            )
        )
    ),
    model=dict(
        type='MetaFreeNetEncoder',
        params=dict(
            in_channels=200,
            out_channels=8,
            patch_size=9,
        )
    ),
    loss_function=dict(
        type='DS3L_MSE_Loss',
        params=dict(),
    ),
    optimizer=dict(
        type='SGD',
        params=dict(
            lr=0.001
        ),
    ),
    lr_scheduler=dict(
        type='None',
        params=dict(),
    ),
    trainer=dict(
        type='OsDS3LTrainer_PB',
        params=dict(
            iterations=5000,
            n_classes=8,
            patch_size=9,
            batch_size_pb=256,
            lr_wnet=6e-5,  # this parameter need to be carefully tuned for different settings
            warmup=5000,
            meta_lr=0.001,
            lr_decay_iter=400000,
            lr_decay_factor=0.2,
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
