config = dict(
    dataset=dict(
        train=dict(
            type='OsMccPaviaUDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/HSI/Pavia_University/PaviaU.mat',
                gt_path='/home/zhw2021/code/HOneCls/Data/HSI/Pavia_University/new_mcc_paviaU_gt15.npy',
                train_flage=True,
                num_classes=9,
                num_train_samples_per_class=100,
                sub_minibatch=5,
                num_unlabeled_samples=4000
            )
        ),
        test=dict(
            type='OsMccPaviaUDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/HSI/Pavia_University/PaviaU.mat',
                gt_path='/home/zhw2021/code/HOneCls/Data/HSI/Pavia_University/new_mcc_paviaU_gt15.npy',
                train_flage=False,
                num_classes=9,
                num_train_samples_per_class=100,
                sub_minibatch=5,
                num_unlabeled_samples=4000
            )
        )
    ),
    model=dict(
        type='MetaFreeNetEncoder',
        params=dict(
            in_channels=103,
            out_channels=9,
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
            n_classes=9,
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
        image_size=(610, 340),
        palette_class_mapping={1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 0},
        palette=[
            [0, 0, 0],
            [192, 192, 192],
            [0, 255, 1],
            [0, 255, 255],
            [0, 128, 1],
            [255, 0, 254],
            [165, 82, 40],
            [129, 0, 127],
            [255, 0, 0],
            [255, 255, 0]
        ],
    )
)
