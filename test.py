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

# Load dataset
file_name = jp(data_dir, 'Dataset_Log_WithoutOutlier_WithoutDouble(LowerThan30m)_Without-4.txt')
head = toolbox.datread(file_name, start=2, end=16)
columns = [h[0].lower() for h in head]
data = toolbox.datread(file_name, start=16)

dataframe = pd.DataFrame(data=data, columns=columns)


# %%  Define grid geometry

xy = np.vstack((dataframe.x, dataframe.y)).T  # X, Y coordinates

dx = 1000  # Block x-dimension
dy = 1000  # Block y-dimension
dz = 0  # Block z-dimension
x_lim, y_lim = np.round(np.max(xy, axis=0)) + np.array([dx, dy])*4  # X max, Y max
xo, yo = np.round(np.min(xy, axis=0)) - np.array([dx, dy])*4  # X min, Y min
zo = 0
nrow = int((y_lim - yo) // dy)  # Number of rows
ncol = int((x_lim - xo) // dx)  # Number of columns
nlay = 1  # Number of layers
along_r = np.ones(ncol) * dx  # Size of each cell along y-dimension - rows
along_c = np.ones(nrow) * dy  # Size of each cell along x-dimension - columns
# TODO: eliminate points outside bounding box
# TODO: delete nodes files if different discretization
# TODO: dirmakers
# %% Plot points coordinates


def plot_coordinates():
    plt.plot(dataframe.x, dataframe.y, 'ko')
    plt.xticks(np.cumsum(along_r)+xo-dx, labels=[])
    plt.yticks(np.cumsum(along_c)+yo-dy, labels=[])
    plt.grid('blue')
    plt.show()


plot_coordinates()


def compute_nodes():
    """
    Determines node location for each data point.
    :return: nodes number
    It is necessary to know the node number to assign the hard data property to the sgems grid
    """
    nodes = np.array([toolbox.my_node(c, along_c, along_r, xo, yo) for c in xy])
    np.save(jp(data_dir, 'nodes'), nodes)  # Save to nodes to avoid recomputing each time

    return nodes


def get_nodes():
    try:
        d_nodes = np.load(jp(data_dir, 'nodes.npy'))
    except FileNotFoundError:
        d_nodes = compute_nodes()

    return d_nodes


data_nodes = get_nodes()


def cleanup():
    """
    Removes no-data rows from data frame and compute the mean of data points sharing the same cell.
    :return: Filtered list of each attribute
    """
    unique_nodes = list(set(data_nodes))
    nodata = -999
    fn = []
    for h in columns[2:]:  # For each feature
        # fixed nodes = [[node i, value i]....]
        fixed_nodes = np.array([[data_nodes[dn], dataframe[h][dn]] for dn in range(len(data_nodes))])
        # Deletes points where val == nodata
        hard_data = np.delete(fixed_nodes, np.where(fixed_nodes[:, 1] == nodata), axis=0)
        # If data points share the same cell, compute their mean and assign the value to the cell
        for n in unique_nodes:
            where = np.where(hard_data[:, 0] == n)[0]
            if len(where) > 1:  # If more than 1 point per cell
                mean = np.mean(hard_data[where, 1])
                hard_data[where, 1] = mean

        fn.append(hard_data.tolist())

    return fn


hard = cleanup()
# Save node list to load it into sgems later
node_file = jp(data_dir, 'fnodes.txt')
with open(jp(data_dir, node_file), 'w') as nd:
    nd.write(repr(hard))

sgrid = [ncol, nrow, nlay,
         dx, dy, dz,
         xo, yo, zo]  # Grid information

grid = toolbox.joinlist('::', sgrid)  # Grid in sgems format

ncells = int(nrow*ncol*nlay)

segp = toolbox.joinlist(' ', [150000, 70000, 40000, 0, 0, 0])  # Search ellipsoid geometry

range_max = 150000
range_med = 70000
range_min = 40000

params = [[res_dir.replace('\\', '//'), 'OPFOLDER'],
          ['test', 'NAME'],
          ['cokriging_test', 'OUTPUT'],
          [grid, 'GRID'],
          [segp, 'SEARCHELLIPSOID'],
          [range_max, 'RMAX'],
          [range_med, 'RMED'],
          [range_min, 'RMIN'],
          [ncells, 'NCELLS'],
          [node_file.replace('\\', '//'), 'FNODES']]

template = """\
import sgems as statistical_simulation\n\
import os

os.chdir("OPFOLDER")
statistical_simulation.execute('DeleteObjects NAME')\n\
statistical_simulation.execute('DeleteObjects hd')\n\
statistical_simulation.execute('DeleteObjects finished')\n\
#\n\
statistical_simulation.execute('NewCartesianGrid  NAME::GRID')\n\
#\n\
properties = ['ca', 'mg', 'nh4', 'cl', 'hco3', 'so4', 'no3', 'no2', 'f', 'i', 'fetot', 'po4']\n\
nodata = -9966699\n\
statistical_simulation.execute('NewCartesianGrid  hd::GRID')\n\

with open("FNODES") as nf:\n\
    fn = eval(nf.read())\n\
for p in range(len(properties)):\n\
    hard_data = [nodata for i in range(NCELLS)]\n\
    for n in fn[p]:\n\
        hard_data[int(n[0])] = n[1]\n\
    statistical_simulation.set_property('hd', properties[p], hard_data)\n\
    
#\n\
#\n\
#\n\
statistical_simulation.execute('RunGeostatAlgorithm  cokriging::/GeostatParamUtils/XML::<parameters>  <algorithm name="cokriging" />     \
<Primary_Harddata_Grid value="hd" region=""  />     \
<Primary_Variable  value="f"  />     \
<Assign_Hard_Data  value="1"  />     \
<Secondary_Harddata_Grid value="hd" region=""  />     \
<Secondary_Variable  value="hco3"  />     \
<Min_Conditioning_Data  value="3" />     \
<Max_Conditioning_Data  value="20" />     \
<Search_Ellipsoid_1  value="40000 40000 0  0 0 0" />    \
<AdvancedSearch_1  use_advanced_search="0"></AdvancedSearch_1>    \
<Min_Conditioning_Data_2  value="3" />     \
<Max_Conditioning_Data_2  value="20" />     \
<Search_Ellipsoid_2  value="5500 5500 0  0 0 0" />    \
<Variogram_C11  nugget="0.074" structures_count="1"  >    \
<structure_1  contribution="0.065"  type="Spherical"   >      \
<ranges max="40000"  medium="40000"  min="0"   />      \
<angles x="0"  y="0"  z="0"   />    </structure_1>  </Variogram_C11>    \
<Variogram_C12  nugget="0.005" structures_count="1"  >    \
<structure_1  contribution="0.0159"  type="Spherical"   >      \
<ranges max="47400"  medium="47400"  min="0"   />      \
<angles x="0"  y="0"  z="0"   />    </structure_1>  \
</Variogram_C12>    <Variogram_C22  nugget="0.047" structures_count="1"  >    \
<structure_1  contribution="0.02"  type="Spherical"   >      \
<ranges max="5500"  medium="5500"  min="0"   />      \
<angles x="0"  y="0"  z="0"   />    \
</structure_1>  </Variogram_C22>    <Correl_Z1Z2  value="" />     <Var_Z2  value="" />     <MM2_Correl_Z1Z2  value="" />     \
<MM2_Variogram_C22  nugget="0" structures_count="1"  >    <structure_1  contribution="0"  type="Spherical"   >      \
<ranges max="0"  medium="0"  min="0"   />      <angles x="0"  y="0"  z="0"   />    \
</structure_1>  </MM2_Variogram_C22>    \
<Grid_Name value="NAME" region=""  />     \
<Property_Name  value="OUTPUT" />     \
<Kriging_Type  value="Simple Kriging (SK)"  />     \
<SK_Means  value="-0.11269 2.28098206" />     \
<Cokriging_Type  value="Full Cokriging"  />   </parameters> ')\n\

statistical_simulation.execute('SaveGeostatGrid  test::OUTPUT.grid::gslib::0::OUTPUT')\n\

"""

for i in range(len(params)):  # Replaces the parameters
    template = template.replace(params[i][1], str(params[i][0]))

# FILE NAME
file_name = jp(res_dir, 'simusgems.py')
sgf = open(file_name, 'w')
sgf.write(template)
sgf.close()

# Create script file
run_script = jp(res_dir, 'statistical_simulation.script')
rscpt = open(run_script, 'w')
rscpt.write(' '.join(['RunScript', file_name]))
rscpt.close()

# Create BAT file
batch = jp(res_dir, 'RunSgems.bat')
bat = open(batch, 'w')
bat.write(' '.join(['cd', res_dir, '\n']))
bat.write(' '.join(['sgems', 'statistical_simulation.script']))
bat.close()

subprocess.call([batch])  # Opens the BAT file


