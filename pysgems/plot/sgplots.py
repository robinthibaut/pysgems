#  Copyright (c) 2022. Robin Thibaut, Ghent University
from os.path import join as jp

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from pysgems.base.packbase import Package
from pysgems.io.sgio import datread


class Plots(Package):
    def __init__(self, project):
        Package.__init__(self, project)
        self.name = self.parent.project_name

    def plot_coordinates(self):
        """plot the coordinates"""
        try:
            # Scatter plot (2D)
            plt.plot(
                self.parent.point_set.raw_data[:, 0],
                self.parent.point_set.raw_data[:, 1],
                "ko",
            )
        except Exception as e:
            logger.info(e)
            pass
        try:
            plt.xticks(
                np.cumsum(self.parent.dis.along_r)
                + self.parent.dis.xo
                - self.parent.dis.dx,
                labels=[],
            )
            plt.yticks(
                np.cumsum(self.parent.dis.along_c)
                + self.parent.dis.yo
                - self.parent.dis.dy,
                labels=[],
            )
        except Exception as e:
            logger.info(e)
            pass

        plt.grid("blue")
        plt.show()

    def plot_2d(self, name, res_file=None, save=False, show=True):
        """Rudimentary 2D plot

        :param name: name of the plot
        :type name: str
        :param res_file: name of the file to plot
        :param save: save the plot
        """
        if res_file is None:
            res_file = jp(self.parent.res_dir, "results.grid")
        matrix = datread(res_file, start=3)
        matrix = np.where(matrix == -9966699, np.nan, matrix)
        matrix = matrix.reshape((self.parent.dis.nrow, self.parent.dis.ncol))
        extent = (
            self.parent.dis.xo,
            self.parent.dis.x_lim,
            self.parent.dis.yo,
            self.parent.dis.y_lim,
        )
        plt.imshow(np.flipud(matrix), cmap="coolwarm", extent=extent)
        plt.plot(
            self.parent.point_set.raw_data[:, 0],
            self.parent.point_set.raw_data[:, 1],
            "k+",
            markersize=1,
            alpha=0.7,
        )
        plt.colorbar()
        if save:
            plt.savefig(jp(self.parent.res_dir, name), bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        plt.close()
