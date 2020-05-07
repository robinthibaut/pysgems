import os
from os.path import join as join_path
import struct
import numpy as np
import toolbox

# Define working directory
cwd = os.getcwd()
# Define datasets directory
data_dir = join_path(cwd, 'datasets', 'test')
# Define file name
f_name = 'sgems_dataset.dat'
# Initialize problem
sgems = toolbox.Sgems(data_dir=data_dir, file_name=f_name, dx=5, dy=5)
# Compute data points node and export file
sgems.export_node_idx()
# Display point coordinates and grid
sgems.plot_coordinates()
# Load your algorithm xml file in the 'algorithms' folder
algo_name = sgems.xml_reader('kriging')
# Show xml structure tree
sgems.show_tree()
# Modify xml below:
sgems.xml_update('Hard_Data', {'grid': 'sgems', 'property': 'Au'})
sgems.xml_update('Grid_Name', {'value': 'sgems', 'region': ''})
# Write python script
# sgems.write_command()
# Run sgems
# sgems.run()


subframe1 = sgems.dataframe[['x', 'y', 'Ag']]
subframe2 = sgems.dataframe[['x', 'y', 'As']]

toolbox.write_point_set('ag.sgems', subframe1)
toolbox.write_point_set('as.sgems', subframe2)