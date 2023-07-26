config = dict(
    dataset=dict(
        train=dict(
            type='MccPaviaUDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/HSI/Pavia_University/PaviaU.mat',
                gt_path='/home/zhw2021/code/HOneCls/Data/HSI/Pavia_University/new_mcc_paviaU_gt15.npy',
                train_flage=True,
                num_classes=9,
                num_train_samples_per_class=100,
                sub_minibatch=20
            )
        ),
        test=dict(
            type='MccPaviaUDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/HSI/Pavia_University/PaviaU.mat',
                gt_path='/home/zhw2021/code/HOneCls/Data/HSI/Pavia_University/new_mcc_paviaU_gt15.npy',
                train_flage=False,
                num_classes=9,
                num_train_samples_per_class=100,
                sub_minibatch=20
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
        image_size=(610, 340),
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
