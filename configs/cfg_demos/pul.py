config = dict(
    dataset=dict(
        train=dict(
            type='HongHuDataset',
            params=dict(
                image_path='./Data/UAVData/WHU-Hi-HongHu/data',
                gt_path='./Data/UAVData/WHU-Hi-HongHu/gt',
                train_flage=True,
                num_positive_train_samples=100,
                sub_minibatch=10,
                ccls=4,
                ratio=40
            )
        ),
        test=dict(
            type='HongHuDataset',
            params=dict(
                image_path='./Data/UAVData/WHU-Hi-HongHu/data',
                gt_path='./Data/UAVData/WHU-Hi-HongHu/gt',
                train_flage=False,
                num_positive_train_samples=100,
                sub_minibatch=10,
                ccls=4,
                ratio=40
            )
        )
    ),
    model=dict(
        type='PUL',
        params=dict(),
    ),

    trainer=dict(
        type='SklearnTrainer',
        params=dict(),
    ),
    meta=dict(
        image_size=(678, 465),
        palette=[
            [0, 0, 0],
            [255, 0, 0],
            [255, 255, 255],
            [176, 48, 96],
            [255, 255, 0],
            [255, 127, 80],
            [0, 255, 0],
            [0, 205, 0],
            [0, 139, 0],
            [127, 255, 212],
            [160, 32, 240],
            [216, 191, 216],
            [0, 0, 255],
            [0, 0, 139],
            [218, 112, 214],
            [160, 82, 45],
            [0, 255, 255],
            [255, 165, 0],
            [127, 255, 0],
            [139, 139, 0],
            [0, 139, 139],
            [205, 181, 205],
            [238, 154, 0]],
    )
)
