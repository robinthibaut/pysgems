#  Copyright (c) 2020. Robin Thibaut, Ghent University

from os.path import join as jp


class Sgems:

    def __init__(self, model_name='sgems_test', model_wd='', **kwargs):

        self.model_name = model_name
        self.model_wd = model_wd

        self.algo_dir = jp(self.model_wd, 'algorithms')  # algorithms directory

        self.dis = None
        # self.data_dir = data_dir  # data directory
        # self.res_dir = res_dir  # result dir initiated when modifying xml file if none given
        # self.file_name = file_name  # data file name
        #
        # # Data
        # self.nodata = nodata
        # self.data = develop.data_ops.Operations()
        # if file_name:
        #     self.file_path = jp(self.data_dir, file_name)
        #     self.data.load_dataframe()
        #
        # # Grid geometry - use self.generate_grid() to update values
        # self.dis = develop.grid_ops.Discretize
        #
        # # Algorithm XML
        # self.auto_update = False  # Experimental feature to auto fill XML and saving binary files
        #
        # self.xml = develop.script_ops.XML(cwd=self.cwd,
        #                                   res_dir=self.res_dir,
        #                                   project_name=self.data.project_name,
        #                                   columns=self.data.columns,
        #                                   auto_update=self.auto_update,
        #                                   algo_dir=self.algo_dir)
        #
        # self.command = develop.sys_ops.Commmands
