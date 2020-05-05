import numpy as np
import time
import os
from os.path import join as jp
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def clockwork(func):
    def wrapper(*args, **kwargs):
        startTime = time.time()
        func(*args, **kwargs)
        endTime = time.time()
        print("time: ", endTime - startTime, " seconds")

    return wrapper


def datread(file=None, start=0, end=-1):
    """Reads space separated dat file"""
    with open(file, 'r') as fr:
        lines = np.copy(fr.readlines())
        try:
            op = np.array([list(map(float, line.split())) for line in lines[start:end]])
        except ValueError:
            op = [line.split() for line in lines[start:end]]
    return op


def joinlist(j, mylist):
    """
    Function that joins an array of numbers with j as separator. For example, joinlist('^', [1,2]) returns 1^2
    """
    gp = j.join(map(str, mylist))

    return gp


def blocks_from_rc(rows, columns, xo, yo):
    """

    :param rows: array of x-widths along a row
    :param columns: array of y-widths along a column
    :param xo:
    :param yo:
    :return: generator of (cell node number, block vertices coordinates, block center)
    """
    nrow = len(rows)
    ncol = len(columns)
    delr = rows
    delc = columns
    r_sum = np.cumsum(delr) + yo
    c_sum = np.cumsum(delc) + xo

    def get_node(i, j):
        """
        Get node index to fixed hard data
        :param i: row number
        :param j: column number
        :return: node number
        """
        return int(i * ncol + j)

    for c in range(nrow):
        for n in range(ncol):
            b = [[c_sum[n] - delc[n], r_sum[c] - delr[c]],
                 [c_sum[n] - delc[n], r_sum[c]],
                 [c_sum[n], r_sum[c]],
                 [c_sum[n], r_sum[c] - delr[c]]]
            yield get_node(c, n), np.array(b), np.mean(b, axis=0)


@clockwork
def my_node(xy, rows, columns, xo, yo):
    """
    Given a point coordinate xy [x, y], computes its node number by computing the euclidean distance of each cell
    center.
    :param xy:
    :param rows: array of x-widths along a row
    :param columns: array of y-widths along a column
    :param xo: x origin
    :param yo: y origin
    :return:
    """

    rn = np.array(xy)

    dmin = np.min([rows.min(), columns.min()]) / 2
    blocks = blocks_from_rc(rows, columns, xo, yo)
    vmin = np.inf
    cell = None
    for b in blocks:
        c = b[2]
        dc = np.linalg.norm(rn - c)  # Euclidean distance
        if dc <= dmin:  # If point is inside cell
            return b[0]
        elif dc < vmin:
            vmin = dc
            cell = b[0]

    return cell


class Sgems:

    def __init__(self, file_name, dx, dy, xo=None, yo=None, x_lim=None, y_lim=None):

        # Directories
        self.cwd = os.getcwd()
        self.data_dir = jp(self.cwd, 'dataset')
        self.res_dir = jp(self.cwd, 'results')
        self.file_name = file_name

        # Data
        self.dataframe, self.project_name, self.columns = self.loader()
        self.xy = np.vstack((self.dataframe[:, 0], self.dataframe[:, 1])).T  # X, Y coordinates

        # Grid geometry
        self.dx = dx  # Block x-dimension
        self.dy = dy  # Block y-dimension
        self.dz = 0  # Block z-dimension

        self.xo, self.yo, self.x_lim, self.y_lim, self.nrow, self.ncol, self.nlay, self.along_r, self.along_c \
            = self.grid(dx, dy, xo, yo, x_lim, y_lim)

        self.plot_coordinates()

    # Load sgems dataset
    def loader(self):
        project_info = datread(self.file_name, end=2)
        project_name = project_info[0][0].lower()
        n_features = int(project_info[1][0])
        head = datread(self.file_name, start=2, end=2+n_features)
        columns_name = [h[0].lower() for h in head]
        data = datread(self.file_name, start=2+n_features)
        return data, project_name, columns_name

    def load_dataframe(self):
        self.dataframe, self.project_name, self.columns = self.loader()
        self.xy = np.vstack((self.dataframe[:, 0], self.dataframe[:, 0])).T  # X, Y coordinates

    def grid(self, dx, dy, xo=None, yo=None, x_lim=None, y_lim=None):

        if x_lim is None and y_lim is None:
            x_lim, y_lim = np.round(np.max(self.xy, axis=0)) + np.array([dx, dy]) * 4  # X max, Y max
        if xo is None and yo is None:
            xo, yo = np.round(np.min(self.xy, axis=0)) - np.array([dx, dy]) * 4  # X min, Y min

        nrow = int((y_lim - yo) // dy)  # Number of rows
        ncol = int((x_lim - xo) // dx)  # Number of columns
        nlay = 1  # Number of layers
        along_r = np.ones(ncol) * dx  # Size of each cell along y-dimension - rows
        along_c = np.ones(nrow) * dy  # Size of each cell along x-dimension - columns

        return xo, yo,  x_lim, y_lim, nrow, ncol, nlay, along_r, along_c

    def plot_coordinates(self):
        plt.plot(self.dataframe[:, 0], self.dataframe[:, 1], 'ko')
        plt.xticks(np.cumsum(self.along_r) + self.xo - self.dx, labels=[])
        plt.yticks(np.cumsum(self.along_c) + self.yo - self.dy, labels=[])
        plt.grid('blue')
        plt.show()
