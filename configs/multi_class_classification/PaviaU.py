config = dict(
    dataset=dict(
        train=dict(
            type='MccPaviaUDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/HSI/Pavia_University/PaviaU.mat',
                gt_path='/home/zhw2021/code/HOneCls/Data/HSI/Pavia_University/PaviaU_gt.mat',
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
                gt_path='/home/zhw2021/code/HOneCls/Data/HSI/Pavia_University/PaviaU_gt.mat',
                train_flage=False,
                num_classes=9,
                num_train_samples_per_class=100,
                sub_minibatch=20
            )
        )
    ),
    model=dict(
        type='FreeOCNet',
        params=dict(
            in_channels=103,
            num_classes=9,
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
            lr=0.001,
            momentum=0.9,
            weight_decay=0.001
        ),
    ),
    lr_scheduler=dict(
        type='PolynomialLR',
        params=dict(
            power=1,
            total_iters=200,
        ),
    ),
    trainer=dict(
        type='MccSingleModelTrainer',
        params=dict(
            max_iters=200
        ),
    ),
    meta=dict(
        save_path='./Log/MCC/FPGA',
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
