import os
from os.path import join as join_path
import toolbox

# Define working directory
cwd = os.getcwd()
# Define datasets directory
data_dir = join_path(cwd, 'datasets', 'benin')
# Define file name
f_name = 'Dataset_Log_WithoutOutlier_WithoutDouble(LowerThan30m)_Without-4.txt'
# Initialize problem
sgems = toolbox.Sgems(dx=5000, dy=5000, data_dir=data_dir, file_name=f_name)
sgems.generate_grid()
# Display point coordinates and grid
sgems.plot_coordinates()
# Load your algorithm xml file in the 'algorithms' folder
algo_name = sgems.xml_reader('cokriging')
# Show xml structure tree
sgems.show_tree()
# Modify xml below:
# sgems.xml_update('Hard_Data', {'grid': 'ag_grid', 'region': '', 'property': 'ag'})
# Write python script
sgems.write_command()
# Run sgems
sgems.run()

