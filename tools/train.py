import argparse
import sys
import os

sys.path.append(os.path.abspath('.'))

from HOC.apis import build_trainer
from HOC.utils import read_config
from HOC.utils import set_random_seed


def Argparse():
    parser = argparse.ArgumentParser(description='HOC Training')
    parser.add_argument('-c', '--cfg', type=str, help='File path of config')
    parser.add_argument('-r', '--random_seed', type=int, default=None, help='Random seed')
    return parser.parse_args()


if __name__ == "__main__":
    args = Argparse()
    config = read_config(args.cfg)
    random_seed = args.random_seed

    if random_seed is not None:
        set_random_seed(random_seed)
        config['meta'].update(random_seed=random_seed)

    trainer = build_trainer(config)(**config)
    trainer.run(validation=True)
