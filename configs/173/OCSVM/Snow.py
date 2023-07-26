config = dict(
    dataset=dict(
        train=dict(
            type='OneSevenThreeDataset',
            params=dict(
                image_path='./Data/173Data/Snow/snow.dat',
                # image_path='./Data/173Data/Snow/noise.dat',
                gt_path='./Data/173Data/Snow/Snow_unlabeled_gt',
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
                image_path='./Data/173Data/Snow/snow.dat',
                # image_path='./Data/173Data/Snow/noise.dat',
                gt_path='./Data/173Data/Snow/Snow_unlabeled_gt',
                train_flage=False,
                num_positive_train_samples=50,
                sub_minibatch=10,
                ccls=1,
                ratio=400
            )
        )
    ),
    model=dict(
        type='OCSVM',
        params=dict(kernel='rbf'),
    ),

    trainer=dict(
        type='SklearnTrainer',
        params=dict(),
    ),
    meta=dict(
        save_path='Log/173',
        image_size=(624, 848),
        palette=[
            [0, 0, 0],
            [255, 0, 0]],
    )
)
