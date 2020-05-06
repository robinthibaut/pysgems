import os
from os.path import join as join_path
import toolbox

# Define working directory
cwd = os.getcwd()
# Define datasets directory
data_dir = join_path(cwd, 'datasets', 'test')
# Define file name
f_name = 'sgems_dataset.dat'
# Initialize problem
sgems = toolbox.Sgems(data_dir=data_dir, file_name=f_name, dx=1, dy=1)
# Compute data points node and export file
sgems.export_node_idx()
# Display point coordinates and grid
sgems.plot_coordinates()
# Load your algorithm xml file in the 'algorithms' folder
algo_name = sgems.xml_reader('kriging')
# Show xml structure tree
sgems.show_tree()
# Modify xml below:
sgems.xml_update('Hard_Data', {'grid': 'sgems', 'property': 'au'})
# Write python script
sgems.write_command()
# Run sgems
sgems.run()
