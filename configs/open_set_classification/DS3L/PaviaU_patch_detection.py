config = dict(
    dataset=dict(
        train=dict(
            type='OsMccPaviaUDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/HSI/Pavia_University/PaviaU.mat',
                gt_path='/home/zhw2021/code/HOneCls/Data/HSI/Pavia_University/new_paviaU_gt15.npy',
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
                gt_path='/home/zhw2021/code/HOneCls/Data/HSI/Pavia_University/new_paviaU_gt15.npy',
                train_flage=False,
                num_classes=10,
                num_train_samples_per_class=100,
                sub_minibatch=5,
                num_unlabeled_samples=4000
            )
        )
    ),
    model=dict(
        type='FreeNetEncoder',
        params=dict(
            in_channels=103,
            out_channels=9,
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
            checkpoint_path="/home/zhw2021/code/HOneCls/tools/Log/OpenSet/OsMccPaviaUDataset/OsDS3LTrainer_PB/DS3L_MSE_Loss/MetaFreeNetEncoder/2023-06-29 08-39-36/checkpoint.pth",
            patch_size=9,
            batch_size_pb=256,
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