config = dict(
    dataset=dict(
        train=dict(
            type='MccSalinasDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/HSI/Salinas/Salinas_corrected.mat',
                gt_path='/home/zhw2021/code/HOneCls/Data/HSI/Salinas/new_mcc_salinas_gt17.npy',
                train_flage=True,
                num_classes=16,
                num_train_samples_per_class=100,
                sub_minibatch=5
            )
        ),
        test=dict(
            type='MccSalinasDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/HSI/Salinas/Salinas_corrected.mat',
                gt_path='/home/zhw2021/code/HOneCls/Data/HSI/Salinas/new_mcc_salinas_gt17.npy',
                train_flage=False,
                num_classes=16,
                num_train_samples_per_class=100,
                sub_minibatch=5
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
        image_size=(512, 217),
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
            [246, 239, 0]],
    )
)
