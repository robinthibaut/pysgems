#  Copyright (c) 2020. Robin Thibaut, Ghent University
import os
import shutil
import time

import numpy as np
from shapely.geometry import Point, Polygon


class Cells:
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


    def compute_nodes(self):
        """
        Determines node location for each data point.
        It is necessary to know the node number to assign the hard data property to the sgems grid.
        :return: nodes number
        """
        nodes = np.array([self.my_node(c) for c in self.xyz])

        np.save(self.node_file, nodes)  # Save to nodes to avoid recomputing each time


    def get_nodes(self):
        npar = np.array([self.dx, self.dy, self.dz,
                         self.xo, self.yo, self.zo,
                         self.x_lim, self.y_lim, self.z_lim,
                         self.nrow, self.ncol, self.nlay])

        if os.path.isfile(self.dis_file):  # Check previous grid info
            pdis = np.loadtxt(self.dis_file)
            # If different, recompute data points node by deleting previous node file
            if not np.array_equal(pdis, npar):
                print('New grid found')
                try:
                    os.remove(self.node_file)
                    os.remove(self.node_value_file)
                except FileNotFoundError:
                    pass
                finally:
                    np.savetxt(self.dis_file, npar)
                    self.compute_nodes()
            else:
                print('Using previous grid')
                try:
                    np.load(self.node_file)
                except FileNotFoundError:
                    self.compute_nodes()
        else:
            np.savetxt(self.dis_file, npar)
            self.compute_nodes()

        self.export_node_idx()


    def nodes_cleanup(self, features):
        """
        Removes no-data rows from data frame and compute the mean of data points sharing the same cell.
        :param features: str or list(str) of features name to save
        :return: Filtered list of each attribute
        """
        data_nodes = np.load(self.node_file)
        unique_nodes = list(set(data_nodes))

        if not isinstance(features, list):
            features = [features]
        fn = []
        for h in features:  # For each feature
            # fixed nodes = [[node i, value i]....]
            fixed_nodes = np.array([[data_nodes[dn], self.dataframe[h][dn]] for dn in range(len(data_nodes))])
            # Deletes points where val == nodata
            hard_data = np.delete(fixed_nodes, np.where(fixed_nodes == self.nodata)[0], axis=0)
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
        """
        Export the list of shape (n features, m nodes, 2) containing the node of each point data with the corresponding
        value, for each feature
        """
        self.hard_data_objects = self.columns[3:]
        if not os.path.isfile(self.node_value_file):
            hard = self.nodes_cleanup(features=self.hard_data_objects)
            with open(self.node_value_file, 'w') as nd:
                nd.write(repr(hard))
            shutil.copyfile(self.node_value_file,
                            self.node_value_file.replace(os.path.dirname(self.node_value_file), self.res_dir))
