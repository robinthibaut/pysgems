#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
import subprocess
import time
from os.path import join as jp


class Commmands:

    def __init__(self, res_dir):
        self.res_dir = res_dir

    def script_file(self):
        """Create script file"""
        run_script = jp(self.res_dir, 'sgems.script')
        rscpt = open(run_script, 'w')
        rscpt.write(' '.join(['RunScript', jp(self.res_dir, 'simusgems.py')]))
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
