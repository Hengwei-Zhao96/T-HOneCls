config = dict(
    dataset=dict(
        train=dict(
            type='OsMccSalinasDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/HSI/Salinas/Salinas_corrected.mat',
                gt_path='/home/zhw2021/code/HOneCls/Data/HSI/Salinas/new_mcc_salinas_gt17.npy',
                train_flage=True,
                num_classes=16,
                num_train_samples_per_class=100,
                sub_minibatch=5,
                num_unlabeled_samples=4000
            )
        ),
        test=dict(
            type='OsMccSalinasDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/HSI/Salinas/Salinas_corrected.mat',
                gt_path='/home/zhw2021/code/HOneCls/Data/HSI/Salinas/new_mcc_salinas_gt17.npy',
                train_flage=False,
                num_classes=16,
                num_train_samples_per_class=100,
                sub_minibatch=5,
                num_unlabeled_samples=4000
            )
        )
    ),
    model=dict(
        type='MetaFreeNetEncoder',
        params=dict(
            in_channels=204,
            out_channels=16,
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
            n_classes=16,
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
        image_size=(512, 217),
        palette_class_mapping={1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13,
                               14: 14, 15: 15, 16: 16, 17: 0},
        palette=[
            [0, 0, 0],
            [220, 184, 9],
            [3, 0, 154],
            [255, 0, 0],
            [255, 52, 155],
            [255, 102, 255],
            [0, 0, 255],
            [236, 129, 1],
            [0, 255, 0],
            [131, 131, 0],
            [153, 0, 153],
            [0, 247, 241],
            [0, 153, 153],
            [0, 153, 0],
            [138, 95, 45],
            [103, 254, 203],
            [246, 239, 0]
        ],
    )
)
