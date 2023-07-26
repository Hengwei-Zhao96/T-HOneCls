config = dict(
    dataset=dict(
        test=dict(
            type='OneSevenThreeDataset',
            params=dict(
                image_path='./Data/173Data/Park/park.dat',
                # image_path='./Data/173Data/Park/noise.dat',
                gt_path='./Data/173Data/Park/gt',
                train_flage=False,
                num_positive_train_samples=50,
                sub_minibatch=10,
                ccls=1,
                ratio=400
            )
        )
    ),
    model=dict(
        type='SimpleFreeOCNet',
        params=dict(
            in_channels=270,
            num_classes=1,
            block_channels=(64, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        )
    ),
    inference=dict(
        type='TRT',
        params=dict(
            save_path='./test',
            checkpoint_path="/data1/zhw2021/Log/173/OneSevenThreeDataset/SelfCalibrationTrainer/TaylorVarPULossPf/SimpleFreeOCNet/1/2022-09-24 14-31-56/model_t/checkpoint_t.pth",
        )
    ),
    meta=dict(
        image_size=(498, 714),
        palette=[
            [0, 0, 0],
            [255, 0, 0]],
    )
)
