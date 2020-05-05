import os
from os.path import join as jp
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import toolbox

# Directories
cwd = os.getcwd()
data_dir = jp(cwd, 'dataset')
res_dir = jp(cwd, 'results')
f_name = jp(data_dir, 'Dataset_Log_WithoutOutlier_WithoutDouble(LowerThan30m)_Without-4.txt')

sgems = toolbox.Sgems(file_name=f_name, dx=5000, dy=5000)
sgems.plot_coordinates()
sgems.export_node_idx()

sgrid = [sgems.ncol, sgems.nrow, sgems.nlay,
         sgems.dx, sgems.dy, sgems.dz,
         sgems.xo, sgems.yo, 0]  # Grid information

grid = toolbox.joinlist('::', sgrid)

params = [[sgems.res_dir.replace('\\', '//'), 'RES_DIR'],
          [grid, 'GRID'],
          [sgems.project_name, 'PROJECT_NAME'],
          [sgems.node_value_file.replace('\\', '//'), 'NODES_VALUES_FILE']]

with open('simusgems_template.py') as sst:
    template = sst.read()

for i in range(len(params)):  # Replaces the parameters
    template = template.replace(params[i][1], params[i][0])

with open('simusgems_test.py', 'w') as sstw:
    sstw.write(template)
