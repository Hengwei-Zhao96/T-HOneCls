config = dict(
    dataset=dict(
        train=dict(
            type='Mcc2013GRSSDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/HSI/2013DFC/2013_IEEE_GRSS_DF_Contest_CASI.tif',
                gt_path='/home/zhw2021/code/HOneCls/Data/HSI/2013DFC/grd.tif',
                train_flage=True,
                num_classes=15,
                num_train_samples_per_class=200,
                sub_minibatch=20
            )
        ),
        test=dict(
            type='Mcc2013GRSSDataset',
            params=dict(
                image_path='/home/zhw2021/code/HOneCls/Data/HSI/2013DFC/2013_IEEE_GRSS_DF_Contest_CASI.tif',
                gt_path='/home/zhw2021/code/HOneCls/Data/HSI/2013DFC/testlabels.tif',
                train_flage=False,
                num_classes=15,
                num_train_samples_per_class=200,
                sub_minibatch=20
            )
        )
    ),
    model=dict(
        type='FreeNet',
        params=dict(
            in_channels=144,
            num_classes=15,
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
            max_iters=1000
        ),
    ),
    meta=dict(
        save_path='./Logtest/FPGA',
        image_size=(349, 1905),
        palette=[
            [0, 0, 0],
            [139, 67, 45],
            [0, 0, 255],
            [255, 100, 0],
            [0, 255, 123],
            [164, 75, 155],
            [101, 173, 255],
            [118, 254, 172],
            [60, 91, 112],
            [255, 255, 0],
            [255, 255, 125],
            [255, 0, 255],
            [100, 0, 255],
            [0, 172, 254],
            [0, 255, 0],
            [171, 175, 80]],
    )
)
