import os
from os.path import join as join_path

import toolbox

# Define working directory
cwd = os.getcwd()
# Define datasets directory
data_dir = join_path(cwd, 'datasets', 'demo')
# Define file name
f_name = 'sgems_dataset.dat'
# Initialize problem, define dx and dy and indicate dataset path
sgems = toolbox.Sgems(dx=2, dy=2, data_dir=data_dir, file_name=f_name)
# Generate grid
sgems.generate_grid()
# Display point coordinates and grid
sgems.plot_coordinates()
# Which feature are available
print(sgems.columns)
# Load your algorithm xml file in the 'algorithms' folder
algo_name = sgems.xml_reader('kriging')
# Show xml structure tree
sgems.show_tree()
# Modify xml below:
# By default, the feature grid name of feature X is called 'X_grid'.
sgems.xml_update('Hard_Data', 'grid', 'ag_grid')
sgems.xml_update('Hard_Data', 'property', 'ag')
# Write datasets
sgems.make_data('ag')
# Write python script
sgems.write_command()
# Run sgems
sgems.run()
# Plot 2D results
sgems.plot_2d(save=True)
