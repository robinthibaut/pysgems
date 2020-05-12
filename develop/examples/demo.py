#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
from os.path import join as join_path

from develop import sgems


def main():
    # %% Define working directory
    cwd = os.getcwd()
    # %% Define datasets directory
    data_dir = join_path(cwd, 'datasets', 'demo')
    # %% Define file name
    f_name = 'sgems_dataset.dat'
    # %% Initialize problem, define dx and dy and indicate dataset path
    sg = sgems.Sgems(dx=2, dy=2, data_dir=data_dir, file_name=f_name)
    sg.res_dir = join_path(cwd, 'results', 'demo')
    # %% Generate grid. Grid dimensions are automatically generated based on the data points
    # unless specified otherwise
    # sgems.generate_grid()
    # %% Display point coordinates and grid
    sg.plot_coordinates()
    # %% Which feature are available
    print(sg.dis.columns)
    # %% Load your algorithm xml file in the 'algorithms' folder. A result folder will automatically be generated at
    # this time if no such folder is already defined.
    sg.xml.xml_reader('kriging')
    # %% Show xml structure tree
    sg.xml.show_tree()
    # %% Modify xml below:
    # By default, the feature grid name of feature X is called 'X_grid'.
    # sgems.xml_update(path, attribute, new value)
    sg.xml.xml_update('Hard_Data', 'grid', 'ag_grid')
    sg.xml.xml_update('Hard_Data', 'property', 'ag')
    # %% Write datasets of needed features
    # sgems.export_01(['f1', 'f2'...'fn'])
    sg.data.export_01('ag')
    # %% Write python script
    sg.write_command()
    # %% Run sgems
    sg.run()
    # Plot 2D results
    sg.plot_2d(save=True)


if __name__ == '__main__':
    main()
