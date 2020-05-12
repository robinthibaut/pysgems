#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
import shutil
import time
from os.path import join as jp

import numpy as np
from shapely.geometry import Point, Polygon


def blocks_from_rc(rows, columns, layers, xo=0, yo=0, zo=0):
    """
    Yields blocks defining grid cells
    :param rows: array of x-widths along a row
    :param columns: array of y-widths along a column
    :param layers: array of z-widths along a column
    :param xo: x origin
    :param yo: y origin
    :param zo: z origin
    :return: generator of (cell node number, block vertices coordinates, block center)
    """
    nrow = len(rows)
    ncol = len(columns)
    nlay = len(layers)
    delr = rows
    delc = columns
    dell = layers
    r_sum = np.cumsum(delr) + yo
    c_sum = np.cumsum(delc) + xo
    l_sum = np.cumsum(delc) + zo

    def get_node(r, c, h):
        """
        Get node index to fix hard data
        :param r: row number
        :param c: column number
        :param h: layer number
        :return: node number
        """
        nrc = nrow * ncol
        return int((h * nrc) + (r * ncol) + c)

    for k in range(nlay):
        for i in range(nrow):
            for j in range(ncol):
                b = [
                    [c_sum[j] - delc[j], r_sum[i] - delr[i], l_sum[k] - dell[k]],
                    [c_sum[j], r_sum[i] - delr[i], l_sum[k] - dell[k]],
                    [c_sum[j] - delc[j], r_sum[i], l_sum[k] - dell[k]],
                    [c_sum[j], r_sum[i], l_sum[k] - dell[k]],

                    [c_sum[j] - delc[j], r_sum[i] - delr[i], l_sum[k]],
                    [c_sum[j], r_sum[i] - delr[i], l_sum[k]],
                    [c_sum[j] - delc[j], r_sum[i], l_sum[k]],
                    [c_sum[j], r_sum[i], l_sum[k]]
                ]
                yield get_node(i, j, k), np.array(b), np.mean(b, axis=0)


class Discretize:

    def __init__(self,
                 model,
                 dx=1,
                 dy=1,
                 dz=1,
                 xo=0,
                 yo=0,
                 zo=0,
                 x_lim=1,
                 y_lim=1,
                 z_lim=1,
                 nodata=-9966699):

        self.model = model
        self.model.dis = self
        self.nodata = nodata
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.xo = xo
        self.yo = yo
        self.zo = zo
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.z_lim = z_lim
        self.nrow = None
        self.ncol = None
        self.nlay = None
        self.along_r = None
        self.along_c = None
        self.along_l = None

    def generate_grid(self,
                      xo=None,
                      yo=None,
                      zo=None,
                      x_lim=None,
                      y_lim=None,
                      z_lim=None):
        """
        Constructs the grid geometry. The user can not control directly the number of rows and columns
        but instead inputs the cell size in x and y dimensions.
        xo, yo, x_lim, y_lim, defining the bounding box of the grid, are None by default, and are computed
        based on the data points distribution.
        :param xo:
        :param yo:
        :param zo:
        :param x_lim:
        :param y_lim:
        :param z_lim:
        """

        if self.dy > 0:
            nrow = int((y_lim - yo) // self.dy)  # Number of rows
        else:
            nrow = 1
        if self.dx > 0:
            ncol = int((x_lim - xo) // self.dx)  # Number of columns
        else:
            ncol = 1
        if self.dz > 0:
            nlay = int((z_lim - zo) // self.dz)  # Number of layers
        else:
            nlay = 1

        along_r = np.ones(ncol) * self.dx  # Size of each cell along y-dimension - rows
        along_c = np.ones(nrow) * self.dy  # Size of each cell along x-dimension - columns
        along_l = np.ones(nlay) * self.dz  # Size of each cell along x-dimension - columns

        self.xo = xo
        self.yo = yo
        self.zo = zo
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.z_lim = z_lim
        self.nrow = nrow
        self.ncol = ncol
        self.nlay = nlay
        self.along_r = along_r
        self.along_c = along_c
        self.along_l = along_l

        self.model.dis = self

    def my_node(self, xyz):
        """
        Given a point coordinate xy [x, y], computes its node number by computing the euclidean distance of each cell
        center.
        :param xyz:  x, y, z coordinate of data point
        :return:
        """
        start = time.time()
        rn = np.array(xyz)
        # first check if point is within the grid
        p_xy = Point([rn[0], rn[1]])
        p_xz = Point([rn[0], rn[2]])
        poly_xy = Polygon([(self.xo, self.yo), (self.x_lim, self.yo), (self.x_lim, self.y_lim), (self.xo, self.y_lim)])
        poly_xz = Polygon([(self.xo, self.zo), (self.x_lim, self.zo), (self.x_lim, self.z_lim), (self.xo, self.z_lim)])

        if self.dz == 0:
            crit = p_xy.within(poly_xy)
        else:
            crit = p_xy.within(poly_xy) and p_xz.within(poly_xz)

        if crit:
            if self.dz > 0:
                dmin = np.min([self.dx, self.dy, self.dz]) / 2
            else:
                dmin = np.min([self.dx, self.dy]) / 2

            blocks = blocks_from_rc(self.along_c, self.along_r, self.along_l,
                                    self.xo, self.yo, self.zo)
            vmin = np.inf
            cell = None
            for b in blocks:
                c = b[2]
                dc = np.linalg.norm(rn - c)  # Euclidean distance
                if dc <= dmin:  # If point is inside cell
                    print('found 1 node in {} s'.format(time.time() - start))
                    return b[0]
                if dc < vmin:
                    vmin = dc
                    cell = b[0]
            print('found 1 node in {} s'.format(time.time() - start))
            return cell
        else:
            return self.nodata

    def compute_nodes(self, xyz, node_file):
        """
        Determines node location for each data point.
        It is necessary to know the node number to assign the hard data property to the sgems grid.
        :param xyz: Data points x, y, z coordinates
        :param node_file: file path where data points cells will be stored
        """

        nodes = np.array([self.my_node(c) for c in xyz])

        np.save(node_file, nodes)  # Save to nodes to avoid recomputing each time

    def get_nodes(self, xyz, node_file, dis_file):
        """

        :param xyz: Data points 3D coordinates array
        :param node_file: File path to the data points nodes files
        :param dis_file: File path to the discretization file

        """

        npar = np.array([self.dx, self.dy, self.dz,
                         self.xo, self.yo, self.zo,
                         self.x_lim, self.y_lim, self.z_lim,
                         self.nrow, self.ncol, self.nlay])

        if os.path.isfile(dis_file):  # Check previous grid info
            pdis = np.loadtxt(dis_file)
            # If different, recompute data points node by deleting previous node file
            if not np.array_equal(pdis, npar):
                print('New grid found')
                try:
                    os.remove(node_file)
                except FileNotFoundError:
                    pass
                finally:
                    np.savetxt(dis_file, npar)
                    self.compute_nodes(xyz, node_file)
            else:
                print('Using previous grid')
                try:
                    np.load(node_file)
                except FileNotFoundError:
                    self.compute_nodes(xyz, node_file)
        else:
            np.savetxt(dis_file, npar)
            self.compute_nodes(xyz, node_file)

    def print_hard_data(self, subdata, node_file, output_dir):
        """
        Removes no-data rows from data frame and compute the mean of data points sharing the same cell.
        Export the list of shape (n features, m nodes, 2) containing the node of each point data with the corresponding
        value, for each feature.

        :param subdata: Pandas dataframe whose columns are the values of features to save as hard data.
        :param node_file: File path to the nodes files
        :param output_dir: Directory where hard data lists will be saved
        :return: Filtered list of each attribute
        """
        data_nodes = np.load(node_file)
        unique_nodes = list(set(data_nodes))

        for h in subdata:  # For each feature
            # fixed nodes = [[node i, value i]....]
            fixed_nodes = np.array([[data_nodes[dn], subdata[h][dn]] for dn in range(len(data_nodes))])
            # Deletes points where val == nodata
            hard_data = np.delete(fixed_nodes, np.where(fixed_nodes == self.nodata)[0], axis=0)
            # If data points share the same cell, compute their mean and assign the value to the cell
            for n in unique_nodes:
                where = np.where(hard_data[:, 0] == n)[0]
                if len(where) > 1:  # If more than 1 point per cell
                    mean = np.mean(hard_data[where, 1])
                    hard_data[where, 1] = mean

            fn = hard_data.tolist()

            # For each, feature X, saves a file X.hard
            cell_values_name = jp(os.path.dirname(node_file), '{}.hard'.format(h))
            with open(cell_values_name, 'w') as nd:
                nd.write(repr(fn))
            shutil.copyfile(cell_values_name,
                            cell_values_name.replace(os.path.dirname(cell_values_name), output_dir))

