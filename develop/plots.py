#  Copyright (c) 2020. Robin Thibaut, Ghent University
from os.path import join as jp

import matplotlib.pyplot as plt
import numpy as np

from develop.sgps import datread


def plot_coordinates(self):
    try:
        plt.plot(self.raw_data[:, 0], self.raw_data[:, 1], 'ko')
    except:
        pass
    try:
        plt.xticks(np.cumsum(self.along_r) + self.xo - self.dx, labels=[])
        plt.yticks(np.cumsum(self.along_c) + self.yo - self.dy, labels=[])
    except:
        pass

    plt.grid('blue')
    plt.show()


def plot_2d(self, save=False):
    """Rudimentary 2D plot"""
    matrix = datread(jp(self.res_dir, 'results.grid'), start=3)
    matrix = np.where(matrix == -9966699, np.nan, matrix)
    matrix = matrix.reshape((self.nrow, self.ncol))
    extent = (self.xo, self.x_lim, self.yo, self.y_lim)
    plt.imshow(np.flipud(matrix), cmap='coolwarm', extent=extent)
    plt.plot(self.raw_data[:, 0], self.raw_data[:, 1], 'k+', markersize=1, alpha=.7)
    plt.colorbar()
    if save:
        plt.savefig(jp(self.res_dir, 'results.png'), bbox_inches='tight', dpi=300)
    plt.show()
