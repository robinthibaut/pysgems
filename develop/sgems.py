#  Copyright (c) 2020. Robin Thibaut, Ghent University
import os
from os.path import join as jp


class Sgems:

    def __init__(self, model_name='sgems_test', model_wd='', res_dir='', script_dir='', **kwargs):

        self.model_name = model_name
        self.model_wd = model_wd
        self.res_dir = res_dir

        self.dis = None  # Discretization instance
        self.point_set = None  # Point set manager instance
        self.algo = None  # XML manipulation instance

        if not script_dir:
            dir_path = os.path.abspath(__file__ + "/../")
            self.template_file = jp(dir_path, 'script_templates/script_template.py')  # Python template file path

    def write_command(self):
        """
        Write python script that sgems will run.
        """

        run_algo_flag = ''  # This empty str will replace the # in front of the commands meant to execute sgems
        # within its python environment
        try:
            name = self.root.find('algorithm').attrib['name']  # Algorithm name
            with open(self.op_file) as alx:  # Remove unwanted \n
                algo_xml = alx.read().strip('\n')

        except AttributeError or FileNotFoundError:
            name = 'None'
            algo_xml = 'None'
            run_algo_flag = '#'  # If no algorithm loaded, then just loads the data

        sgrid = [self.ncol, self.nrow, self.nlay,
                 self.dx, self.dy, self.dz,
                 self.xo, self.yo, self.zo]  # Grid information
        grid = joinlist('::', sgrid)  # Grid in sgems format

        sgems_files = [sf + '.sgems' for sf in self.object_file_names]

        # The list below is the list of flags that will be replaced in the sgems python script
        params = [[run_algo_flag, '#'],
                  [self.res_dir.replace('\\', '//'), 'RES_DIR'],  # for sgems convention...
                  [grid, 'GRID'],
                  [self.project_name, 'PROJECT_NAME'],
                  [str(self.hard_data_objects), 'FEATURES_LIST'],
                  ['results', 'FEATURE_OUTPUT'],  # results.grid = output file
                  [name, 'ALGORITHM_NAME'],
                  [name, 'PROPERTY_NAME'],
                  [algo_xml, 'ALGORITHM_XML'],
                  [str(sgems_files), 'OBJECT_FILES'],
                  [self.node_value_file.replace('\\', '//'), 'NODES_VALUES_FILE']]

        with open(self.template_file) as sst:
            template = sst.read()
        for i in range(len(params)):  # Replaces the parameters
            template = template.replace(params[i][1], params[i][0])

        with open(jp(self.res_dir, 'simusgems.py'), 'w') as sstw:  # Write sgems python file
            sstw.write(template)
