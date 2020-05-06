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
sgems = toolbox.Sgems(data_dir=data_dir, file_name=f_name, dx=2, dy=2)
# Display point coordinates and grid
sgems.plot_coordinates()
# Compute data points node and export file
sgems.export_node_idx()
# Load your algorithm xml file in the 'algorithms' folder
algo_name = sgems.xml_reader('cokriging')
# Show xml structure tree
sgems.show_tree()
# Modify xml below:

# Write python script
sgems.write_command()
# Run sgems
sgems.run()
