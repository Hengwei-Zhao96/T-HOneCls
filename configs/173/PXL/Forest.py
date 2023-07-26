config = dict(
    dataset=dict(
        train=dict(
            type='OneSevenThreeDataset',
            params=dict(
                image_path='./Data/173Data/Forest/forest.dat',
                # image_path='./Data/173Data/Forest/noise.dat',
                gt_path='./Data/173Data/Forest/Forest_unlabeled_gt',
                train_flage=True,
                num_positive_train_samples=100,
                sub_minibatch=10,
                ccls=1,
                ratio=400
            )
        ),
        test=dict(
            type='OneSevenThreeDataset',
            params=dict(
                image_path='./Data/173Data/Forest/forest.dat',
                # image_path='./Data/173Data/Forest/noise.dat',
                gt_path='./Data/173Data/Forest/Forest_unlabeled_gt',
                train_flage=False,
                num_positive_train_samples=100,
                sub_minibatch=10,
                ccls=1,
                ratio=400
            )
        )
    ),
    model=dict(
        type='PBL',
        params=dict(),
    ),

    trainer=dict(
        type='SklearnTrainer',
        params=dict(),
    ),
    meta=dict(
        save_path='Log/173',
        image_size=(190, 300),
        palette=[
            [0, 0, 0],
            [255, 0, 0]],
    )
)
