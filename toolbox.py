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
        self.node_file = jp(self.data_dir, 'fnodes.txt')

        # Data
        self.dataframe, self.project_name, self.columns = self.loader()
        self.xy = np.vstack((self.dataframe[:, 0], self.dataframe[:, 1])).T  # X, Y coordinates
        self.nodata = -999

        # Grid geometry
        self.dx = dx  # Block x-dimension
        self.dy = dy  # Block y-dimension
        self.dz = 0  # Block z-dimension
        self.xo, self.yo, self.x_lim, self.y_lim, self.nrow, self.ncol, self.nlay, self.along_r, self.along_c \
            = self.grid(dx, dy, xo, yo, x_lim, y_lim)

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

    def compute_nodes(self):
        """
        Determines node location for each data point.
        :return: nodes number
        It is necessary to know the node number to assign the hard data property to the sgems grid
        """
        nodes = np.array([my_node(c, self.along_c, self.along_r, self.xo, self.yo) for c in self.xy])
        np.save(jp(self.data_dir, 'nodes'), nodes)  # Save to nodes to avoid recomputing each time

        return nodes

    def get_nodes(self):
        try:
            d_nodes = np.load(jp(self.data_dir, 'nodes.npy'))
        except FileNotFoundError:
            d_nodes = self.compute_nodes()

        return d_nodes

    def cleanup(self):
        """
        Removes no-data rows from data frame and compute the mean of data points sharing the same cell.
        :return: Filtered list of each attribute
        """
        data_nodes = self.get_nodes()
        unique_nodes = list(set(data_nodes))

        fn = []
        for h in self.columns[2:]:  # For each feature
            # fixed nodes = [[node i, value i]....]
            fixed_nodes = np.array([[data_nodes[dn], self.dataframe[h][dn]] for dn in range(len(data_nodes))])
            # Deletes points where val == nodata
            hard_data = np.delete(fixed_nodes, np.where(fixed_nodes[:, 1] == self.nodata), axis=0)
            # If data points share the same cell, compute their mean and assign the value to the cell
            for n in unique_nodes:
                where = np.where(hard_data[:, 0] == n)[0]
                if len(where) > 1:  # If more than 1 point per cell
                    mean = np.mean(hard_data[where, 1])
                    hard_data[where, 1] = mean

            fn.append(hard_data.tolist())

        return fn

    # Save node list to load it into sgems later
    def export_node_idx(self):
        if not os.path.isfile(self.node_file):
            hard = self.cleanup()
            with open(jp(self.data_dir, self.node_file), 'w') as nd:
                nd.write(repr(hard))
