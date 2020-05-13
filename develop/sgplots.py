#  Copyright (c) 2020. Robin Thibaut, Ghent University
from os.path import join as jp

import matplotlib.pyplot as plt
import numpy as np

from develop.sgps import datread


class Plots:

    def __init__(self, model):
        self.model = model
        self.name = self.model.model_name

    def plot_coordinates(self):
        try:
            plt.plot(self.model.point_set.raw_data[:, 0], self.model.point_set.raw_data[:, 1], 'ko')
        except:
            pass
        try:
            plt.xticks(np.cumsum(self.model.dis.along_r) + self.model.dis.xo - self.model.dis.dx, labels=[])
            plt.yticks(np.cumsum(self.model.dis.along_c) + self.model.dis.yo - self.model.dis.dy, labels=[])
        except:
            pass

        plt.grid('blue')
        plt.show()

    def plot_2d(self, res_file=None, save=False):
        """Rudimentary 2D plot"""
        if res_file is None:
            res_file = jp(self.model.res_dir, 'results.grid')
        matrix = datread(res_file, start=3)
        matrix = np.where(matrix == -9966699, np.nan, matrix)
        matrix = matrix.reshape((self.model.dis.nrow, self.model.dis.ncol))
        extent = (self.model.dis.xo, self.model.dis.x_lim, self.model.dis.yo, self.model.dis.y_lim)
        plt.imshow(np.flipud(matrix), cmap='coolwarm', extent=extent)
        plt.plot(self.model.point_set.raw_data[:, 0], self.model.point_set.raw_data[:, 1], 'k+', markersize=1, alpha=.7)
        plt.colorbar()
        if save:
            plt.savefig(jp(self.model.res_dir, 'results.png'), bbox_inches='tight', dpi=300)
        plt.show()
