config = dict(
    dataset=dict(
        train=dict(
            type='OneSevenThreeDataset',
            params=dict(
                image_path='./Data/173Data/Airplane/plane',
                # image_path='./Data/173Data/Airplane/noise.dat',
                gt_path='./Data/173Data/Airplane/gt',
                train_flage=True,
                num_positive_train_samples=50,
                sub_minibatch=10,
                ccls=1,
                ratio=400
            )
        ),
        test=dict(
            type='OneSevenThreeDataset',
            params=dict(
                image_path='./Data/173Data/Airplane/plane',
                # image_path='./Data/173Data/Airplane/noise.dat',
                gt_path='./Data/173Data/Airplane/gt',
                train_flage=False,
                num_positive_train_samples=50,
                sub_minibatch=10,
                ccls=1,
                ratio=400
            )
        )
    ),

    meta=dict()
)
