#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
import shutil
import time
from os.path import join as jp

import numpy as np

from pysgems.base.packbase import Package


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


class Discretize(Package):

    def __init__(self,
                 project,
                 dx=1,
                 dy=1,
                 dz=0,
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
        """

        Package.__init__(self, project)

        self.cell_file = None

        self.dx = dx
        self.dy = dy
        self.dz = dz
        if self.parent.point_set is not None:
            if self.parent.point_set.dimension == 2:
                self.dz = 0

        # Grid origins
        if xo is None:
            if self.parent.point_set is not None:
                xs = self.parent.point_set.dataframe['x']
                xo = np.min(xs) - 4*self.dx
            else:
                xo = 0

        if yo is None:
            if self.parent.point_set is not None:
                ys = self.parent.point_set.dataframe['y']
                yo = np.min(ys) - 4*self.dy
            else:
                yo = 0

        if zo is None:
            if self.parent.point_set is not None:
                zs = self.parent.point_set.dataframe['z']
                zo = np.min(zs) - 4*self.dz
            else:
                zo = 0

        # Grid limits
        if x_lim is None:
            if self.parent.point_set is not None:
                xs = self.parent.point_set.dataframe['x']
                x_lim = np.max(xs) + 4*self.dx
            else:
                x_lim = 1

        if y_lim is None:
            if self.parent.point_set is not None:
                ys = self.parent.point_set.dataframe['y']
                y_lim = np.max(ys) + 4*self.dy
            else:
                y_lim = 1

        if z_lim is None:
            if self.parent.point_set is not None:
                zs = self.parent.point_set.dataframe['z']
                z_lim = np.max(zs) + 4*self.dz
            else:
                x_lim = 1

        # Cell dimensions
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

        setattr(self.parent, 'dis', self)

    def my_cell(self, xyz):
        """
        Given a point coordinate xy [x, y], computes its cell number by computing the euclidean distance of each cell
        center.
        :param xyz:  x, y, z coordinate of data point
        :return:
        """
        start = time.time()
        rn = np.array(xyz)
        # first check if point is within the grid

        crit = 0
        if np.all(rn >= np.array([self.xo, self.yo, self.zo])) and \
                np.all(rn <= np.array([self.x_lim, self.y_lim, self.z_lim])):
            crit = 1

        if crit:  # if point inside
            if self.parent.point_set.dimension == 3:  # if 3D
                dmin = np.min([self.dx, self.dy, self.dz]) / 2  # minimum distance under which a point is in a cell
            else:  # if 2D
                dmin = np.min([self.dx, self.dy]) / 2  # minimum distance under which a point is in a cell

            blocks = blocks_from_rc(self.along_c, self.along_r, self.along_l,
                                    self.xo, self.yo, self.zo)  # cell generator

            # mapping data points to cells:
            # slow but memory-effective method
            vmin = np.inf
            cell = None
            for b in blocks:
                c = b[2]
                dc = np.linalg.norm(rn - c)  # Euclidean distance
                if dc <= dmin:  # If point is inside cell
                    print('found 1 cell id in {} s'.format(time.time() - start))
                    return b[0]
                if dc < vmin:
                    vmin = dc
                    cell = b[0]
            print('found 1 cell id in {} s'.format(time.time() - start))
            return cell
        else:
            return self.parent.nodata

    def compute_cells(self, xyz):
        """
        Determines cell location for each data point.
        It is necessary to know the cell number to assign the hard data property to the sgems grid.
        :param xyz: Data points x, y, z coordinates
        """

        nodes = np.array([self.my_cell(c) for c in xyz])

        np.save(self.cell_file, nodes)  # Save to nodes to avoid recomputing each time

    def write_hard_data(self, subdata, dis_file=None, cell_file=None, output_dir=None):
        """
        Removes no-data rows from data frame and compute the mean of data points sharing the same cell.
        Export the list of shape (n features, m cells, 2) containing the node of each point data with the corresponding
        value, for each feature.

        :param dis_file:
        :param cell_file:
        :param subdata: Pandas dataframe whose columns are the values of features to save as hard data.
        :param output_dir: Directory where hard data lists will be saved
        :return: Filtered list of each attribute
        """

        if cell_file is not None:
            self.cell_file = cell_file

        if self.cell_file is None:
            self.cell_file = jp(self.parent.res_dir, 'cells.npy')

        if output_dir is None:
            output_dir = self.parent.res_dir

        if dis_file is None:
            dis_file = jp(os.path.dirname(self.cell_file), 'grid.dis')

        xyz = subdata[['x', 'y', 'z']].to_numpy()

        npar = np.array([self.dx, self.dy, self.dz,
                         self.xo, self.yo, self.zo,
                         self.x_lim, self.y_lim, self.z_lim,
                         self.nrow, self.ncol, self.nlay])

        if os.path.isfile(dis_file):  # Check previous grid info
            pdis = np.loadtxt(dis_file)
            # If different, recompute data points cell by deleting previous cell file
            if not np.array_equal(pdis, npar):
                print('New grid found')
                try:
                    os.remove(self.cell_file)
                except FileNotFoundError:
                    pass
                finally:
                    np.savetxt(dis_file, npar)
                    self.compute_cells(xyz)
            else:
                print('Using previous grid')
                if not os.path.exists(self.cell_file):
                    self.compute_cells(xyz)
        else:
            np.savetxt(dis_file, npar)
            self.compute_cells(xyz)

        data_nodes = np.load(self.cell_file)
        unique_nodes = list(set(data_nodes))

        h = subdata.columns.values[-1]
        hd = []
        # fixed nodes = [[node i, value i]....]
        fixed_nodes = np.array([[data_nodes[dn], subdata[h][dn]] for dn in range(len(data_nodes))])
        # Deletes points where val == nodata
        hard_data = np.delete(fixed_nodes, np.where(fixed_nodes == self.parent.nodata)[0], axis=0)
        # If data points share the same cell, compute their mean and assign the value to the cell
        for n in unique_nodes:
            where = np.where(hard_data[:, 0] == n)[0]
            if len(where) > 1:  # If more than 1 point per cell
                mean = np.mean(hard_data[where, 1])
                hard_data[where, 1] = mean

            fn = hard_data.tolist()
            # For each, feature X, saves a file X.hard
            cell_values_name = jp(os.path.dirname(self.cell_file), '{}.hard'.format(h))
            with open(cell_values_name, 'w') as nd:
                nd.write(repr(fn))
            try:
                shutil.copyfile(cell_values_name,
                                cell_values_name.replace(os.path.dirname(cell_values_name), output_dir))
            except shutil.SameFileError:
                pass

        setattr(self.parent, 'hard_data', hd.append(h))

