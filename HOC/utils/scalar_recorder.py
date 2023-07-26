import os

import matplotlib.pylab as plt
import numpy as np
from pandas import DataFrame
import seaborn as sns

plt.switch_backend('agg')


class ScalarRecorder:
    def __init__(self):
        self.scalar = []

    def update_scalar(self, scalar):
        self.scalar.append(scalar)

    def save_kde_fig(self, saved_path):
        scalar = np.asarray(self.scalar)
        scalar_mean = scalar.mean()
        scalar_std = scalar.std()
        plt.cla()
        sns.kdeplot(scalar)
        plt.savefig(os.path.join(saved_path, 'kde_mean_' + str(scalar_mean) + '_std_' + str(scalar_std) + '.png'))

    def save_lineplot_fig(self, y_name, fig_name, saved_path):
        index_scalar = zip(list(range(0, len(self.scalar))), np.asarray(self.scalar))
        data = DataFrame(index_scalar, columns=['Index', y_name])
        plt.cla()
        sns.lineplot(x='Index', y=y_name, data=data)
        plt.savefig(os.path.join(saved_path, fig_name + '.png'))

    def save_scalar_npy(self, name, save_path):
        npy_scalar = np.asarray(self.scalar)
        np.save(os.path.join(save_path, name + '.npy'), npy_scalar)

    def get_mean(self):
        scalar = np.asarray(self.scalar)
        scalar_mean = scalar.mean()
        return scalar_mean
