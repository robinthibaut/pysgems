#  Copyright (c) 2020. Robin Thibaut, Ghent University
import os
import subprocess
import time
import uuid
from os.path import join as jp

from pysgems.utils.sgutils import joinlist


class Sgems:

    def __init__(self, project_name='sgems_test', project_wd='', res_dir='', script_dir='', nodata=-9966699, **kwargs):

        self.project_name = project_name

        self.project_wd = project_wd
        if not self.project_wd:
            self.project_wd = os.getcwd()

        self.res_dir = res_dir
        # result directory generated according to project and algorithm name
        if self.res_dir is None:
            # Generate result directory if none is given
            self.res_dir = jp(self.project_wd, 'results', '_'.join([self.project_name, uuid.uuid1().hex]))
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)

        self.dis = None  # Discretization instance
        self.point_set = None  # Point set manager instance
        self.algo = None  # XML manipulation instance
        self.nodata = nodata

        self.object_file_names = []  # List of features name needed for the algorithm
        self.command_name = ''

        if not script_dir:
            dir_path = os.path.abspath(__file__ + "/../../")
            self.template_file = jp(dir_path, 'script_templates/script_template.py')  # Python template file path

    def write_command(self):
        """
        Write python script that sgems will run.
        """

        self.command_name = jp(self.res_dir, '{}_commands.py'.format(self.project_name))

        run_algo_flag = ''  # This empty str will replace the # in front of the commands meant to execute sgems
        # within its python environment
        try:
            name = self.algo.root.find('algorithm').attrib['name']  # Algorithm name
            name_op = name
            if name == 'sgsim':
                name_op += '__real0'
            with open(self.algo.op_file) as alx:  # Remove unwanted \n
                algo_xml = alx.read().strip('\n')

        except AttributeError or FileNotFoundError:
            name = 'None'
            name_op = name
            algo_xml = 'None'
            run_algo_flag = '#'  # If no algorithm loaded, then just loads the data

        sgrid = [self.dis.ncol, self.dis.nrow, self.dis.nlay,
                 self.dis.dx, self.dis.dy, self.dis.dz,
                 self.dis.xo, self.dis.yo, self.dis.zo]  # Grid information
        grid = joinlist('::', sgrid)  # Grid in sgems format

        sgems_files = [sf + '.sgems' for sf in self.object_file_names]

        # The list below is the list of flags that will be replaced in the sgems python script
        params = [[run_algo_flag, '#~'],
                  [self.res_dir.replace('\\', '//'), 'RES_DIR'],  # for sgems convention...
                  [grid, 'GRID'],
                  [self.project_name, 'PROJECT_NAME'],
                  ['results', 'FEATURE_OUTPUT'],  # results.grid = output file
                  [name, 'ALGORITHM_NAME'],
                  [name_op, 'PROPERTY_NAME'],
                  [algo_xml, 'ALGORITHM_XML'],
                  [str(sgems_files), 'OBJECT_FILES']]

        with open(self.template_file) as sst:
            template = sst.read()
        for i in range(len(params)):  # Replaces the parameters
            template = template.replace(params[i][1], params[i][0])

        with open(self.command_name, 'w') as sstw:  # Write sgems python file
            sstw.write(template)

    def script_file(self):
        """Create script file"""
        run_script = jp(self.res_dir, 'sgems.script')
        rscpt = open(run_script, 'w')
        rscpt.write(' '.join(['RunScript', self.command_name]))
        rscpt.close()

    def bat_file(self):
        """Create bat file"""
        if not os.path.isfile(jp(self.res_dir, 'sgems.script')):
            self.script_file()

        batch = jp(self.res_dir, 'RunSgems.bat')
        bat = open(batch, 'w')
        bat.write(' '.join(['cd', self.res_dir, '\n']))
        bat.write(' '.join(['sgems', 'sgems.script']))
        bat.close()

    def run(self):
        """Call bat file, run sgems"""
        batch = jp(self.res_dir, 'RunSgems.bat')
        if not os.path.isfile(batch):
            self.bat_file()
        start = time.time()

        subprocess.call([batch])  # Opens the BAT file
        print('ran algorithm in {} s'.format(time.time() - start))
