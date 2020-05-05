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

sgems = toolbox.Sgems(file_name=f_name, dx=1000, dy=1000)
sgems.plot_coordinates()
sgems.export_node_idx()
