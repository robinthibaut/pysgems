#  Copyright (c) 2020. Robin Thibaut, Ghent University
import os
from os.path import join as jp

import develop.data_ops
import develop.grid_ops
import develop.script_ops


class Sgems:

    def __init__(self,
                 cwd='',
                 data_dir='',
                 file_name='',
                 res_dir=None,
                 dx=1, dy=1, dz=1,
                 xo=None, yo=None, zo=None,
                 x_lim=None, y_lim=None, z_lim=None,
                 nodata=-999):

        # Directories
        if not cwd:
            self.cwd = os.path.dirname(os.getcwd())  # Main directory
        else:
            self.cwd = cwd
        self.algo_dir = jp(self.cwd, 'algorithms')  # algorithms directory
        self.data_dir = data_dir  # data directory
        self.res_dir = res_dir  # result dir initiated when modifying xml file if none given
        self.file_name = file_name  # data file name

        # Data
        self.nodata = nodata
        self.data = develop.data_ops.Operations()
        if file_name:
            self.file_path = jp(self.data_dir, file_name)
            self.data.load_dataframe()
            self.node_file = jp(os.path.dirname(self.file_path), 'nodes.npy')  # nodes files
            self.node_value_file = jp(os.path.dirname(self.file_path), 'fnodes.txt')
            self.dis_file = jp(os.path.dirname(self.file_path), 'dis.info')

        # Grid geometry - use self.generate_grid() to update values
        self.bounding_box = None
        self.dis = develop.grid_ops.Discretize(dx=dx, dy=dy, dz=dz,
                                               xo=xo, yo=yo, zo=zo,
                                               x_lim=x_lim, y_lim=y_lim, z_lim=z_lim,
                                               dataframe=self.data.dataframe, nodata=self.nodata)
        self.dis.generate_grid()

        # Algorithm XML
        self.auto_update = False  # Experimental feature to auto fill XML and saving binary files

        self.xml = develop.script_ops.XML(cwd=self.cwd,
                                          res_dir=self.res_dir,
                                          project_name=self.data.project_name,
                                          columns=self.data.columns,
                                          auto_update=self.auto_update,
                                          algo_dir=self.algo_dir)


