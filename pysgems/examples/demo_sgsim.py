#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
from os.path import join as join_path

from pysgems.algo.sgalgo import XML
from pysgems.dis.sgdis import Discretize
from pysgems.io.sgio import PointSet
from pysgems.plot.sgplots import Plots
from pysgems.sgems import sg


def main():
    # %% Initiate sgems pjt
    cwd = os.getcwd()  # Working directory
    rdir = join_path(cwd, 'results', 'demo_sgsim')  # Results directory
    pjt = sg.Sgems(project_name='sgsim_test', project_wd=cwd, res_dir=rdir)

    # %% Load hard data point set
    data_dir = join_path(cwd, 'datasets', 'demo_sgsim')
    dataset = 'sgsim_hard_data.eas'
    file_path = join_path(data_dir, dataset)

    hd = PointSet(project=pjt, pointset_path=file_path)
    hd.export_01()  # Exports in binary

    # %% Generate grid. Grid dimensions can automatically be generated based on the data points
    # unless specified otherwise, but cell dimensions dx, dy, (dz) must be specified
    ds = Discretize(project=pjt, dx=10, dy=10, xo=0, yo=0, x_lim=1500, y_lim=1000)

    # %% Display point coordinates and grid
    pl = Plots(project=pjt)
    pl.plot_coordinates()

    # %% Which feature are available
    print(pjt.point_set.columns)

    # %% Load your algorithm xml file in the 'algorithms' folder.
    algo_dir = join_path(os.path.dirname(cwd), 'algorithms')
    al = XML(project=pjt, algo_dir=algo_dir)
    al.xml_reader('sgsim')

    # %% Show xml structure tree
    al.show_tree()

    # %% Modify xml below:
    # By default, the feature grid name of feature X is called 'X_grid'.
    # sgems.xml_update(path, attribute, new value)
    al.xml_update('Assign_Hard_Data', 'value', '1')
    al.xml_update('Hard_Data', new_attribute_dict={'grid': 'hd_grid', 'property': 'hd'})

    # %% Write python script
    pjt.write_command()

    # %% Run sgems
    pjt.run()
    # Plot 2D results
    pl.plot_2d(save=True)


if __name__ == '__main__':
    main()
