config = dict(
    dataset=dict(
        train=dict(
            type='MccSalinasDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/HSI/Salinas/Salinas_corrected.mat',
                gt_path='/home/zhw2021/code/HOneCls/Data/HSI/Salinas/Salinas_gt.mat',
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
                gt_path='/home/zhw2021/code/HOneCls/Data/HSI/Salinas/Salinas_gt.mat',
                train_flage=False,
                num_classes=16,
                num_train_samples_per_class=100,
                sub_minibatch=5
            )
        )
    ),
    model=dict(
        type='FreeOCNet',
        params=dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        )
    ),
    loss_function=dict(
        type='CELossPf',
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
        type='PolynomialLR',
        params=dict(
            power=0.9,
            total_iters=1000,
        ),
    ),
    trainer=dict(
        type='MccSingleModelTrainer',
        params=dict(
            max_iters=100
        ),
    ),
    meta=dict(
        save_path='./Logtest/FPGA',
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
