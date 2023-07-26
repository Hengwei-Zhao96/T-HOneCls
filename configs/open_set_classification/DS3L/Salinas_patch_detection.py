config = dict(
    dataset=dict(
        train=dict(
            type='OsMccSalinasDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/HSI/Salinas/Salinas_corrected.mat',
                gt_path='/home/zhw2021/code/HOneCls/Data/HSI/Salinas/new_salinas_gt17.npy',
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
                gt_path='/home/zhw2021/code/HOneCls/Data/HSI/Salinas/new_salinas_gt17.npy',
                train_flage=False,
                num_classes=17,
                num_train_samples_per_class=100,
                sub_minibatch=5,
                num_unlabeled_samples=4000
            )
        )
    ),
    model=dict(
        type='FreeNetEncoder',
        params=dict(
            in_channels=204,
            out_channels=16,
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
            checkpoint_path="/home/zhw2021/code/HOneCls/tools/Log/OpenSet/OsMccSalinasDataset/OsDS3LTrainer_PB/DS3L_MSE_Loss/MetaFreeNetEncoder/2023-06-29 09-06-50/checkpoint.pth",
            patch_size=9,
            batch_size_pb=256,
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
