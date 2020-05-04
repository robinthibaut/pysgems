import os
from os.path import join as jp

import numpy as np
import pandas as pd

import toolbox

# Directories
cwd = os.getcwd()
data_dir = jp(cwd, 'dataset')
res_dir = jp(cwd, 'results')

# Load dataset
file_name = jp(data_dir, 'Dataset_Log_WithoutOutlier_WithoutDouble(LowerThan30m)_Without-4.txt')
head = toolbox.datread(file_name, start=2, end=16)
columns = [h[0].lower() for h in head]
data = toolbox.datread(file_name, start=16)

dataframe = pd.DataFrame(data=data, columns=columns)

# %%  Define grid geometry

xy = np.vstack((dataframe.x, dataframe.y)).T  # X, Y coordinates

x_lim, y_lim = np.round(np.max(xy, axis=0))  # X max, Y max
xo, yo = np.round(np.min(xy, axis=0))  # X min, Y min
zo = 0
dx = 200  # Block x-dimension
dy = 200  # Block y-dimension
dz = 0  # Block z-dimension
nrow = int((y_lim - yo) // dy)  # Number of rows
ncol = int((x_lim - xo) // dx)  # Number of columns
nlay = 1  # Number of layers
along_r = np.ones(ncol) * dx  # Size of each cell along y-dimension - rows
along_c = np.ones(nrow) * dy  # Size of each cell along x-dimension - columns


def compute_node(coord):

    nodes = toolbox.my_node(coord, along_c, along_r, xo, yo)

    return nodes

