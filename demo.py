import os
from os.path import join as join_path

import toolbox

# Define working directory
cwd = os.getcwd()
# Define datasets directory
data_dir = join_path(cwd, 'datasets', 'demo')
# Define file name
f_name = 'sgems_dataset.txt'
# Initialize problem
sgems = toolbox.Sgems(dx=2, dy=2, data_dir=data_dir, file_name=f_name)
# Generate grid
sgems.generate_grid()
# Display point coordinates and grid
sgems.plot_coordinates()
# Load your algorithm xml file in the 'algorithms' folder
algo_name = sgems.xml_reader('kriging')
# Show xml structure tree
sgems.show_tree()
# Modify xml below:
sgems.xml_update('Primary_Harddata_Grid', 'value', 'f_grid')
sgems.xml_update('Primary_Variable', 'value', 'f')
sgems.xml_update('Secondary_Harddata_Grid', 'value', 'hco3_grid')
sgems.xml_update('Secondary_Variable', 'value', 'hco3')
# Write datasets
sgems.make_data(['f', 'hco3'])
# Write python script
sgems.write_command()
# Run sgems
sgems.run()
# Plot 2D results
sgems.plot_2d(save=True)
