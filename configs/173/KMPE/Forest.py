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

    meta=dict()
)
