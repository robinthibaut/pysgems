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
    rdir = join_path(cwd, 'results', 'demo_kriging')  # Results directory
    pjt = sg.Sgems(project_name='sgems_test', project_wd=cwd, res_dir=rdir)

    # %% Load data point set
    data_dir = join_path(cwd, 'datasets', 'demo_kriging')
    dataset = 'sgems_dataset.eas'
    file_path = join_path(data_dir, dataset)

    ps = PointSet(project=pjt, pointset_path=file_path)

    # %% Generate grid. Grid dimensions can automatically be generated based on the data points
    # unless specified otherwise, but cell dimensions dx, dy, (dz) must be specified
    ds = Discretize(project=pjt, dx=5, dy=5)

    # %% Display point coordinates and grid
    pl = Plots(project=pjt)
    pl.plot_coordinates()

    # %% Which feature are available
    print(pjt.point_set.columns)

    # %% Load your algorithm xml file in the 'algorithms' folder.
    algo_dir = join_path(os.path.dirname(cwd), 'algorithms')
    al = XML(project=pjt, algo_dir=algo_dir)
    al.xml_reader('kriging')

    # %% Show xml structure tree
    al.show_tree()

    # %% Modify xml below:
    # By default, the feature grid name of feature X is called 'X_grid'.
    # sgems.xml_update(path, attribute, new value)
    al.xml_update('Hard_Data', 'grid', 'ag_grid')
    al.xml_update('Hard_Data', 'property', 'ag')

    # %% Write binary datasets of needed features
    # sgems.export_01(['f1', 'f2'...'fn'])
    ps.export_01('ag')

    # %% Write python script
    pjt.write_command()

    # %% Run sgems
    pjt.run()
    # Plot 2D results
    pl.plot_2d(save=True)


if __name__ == '__main__':
    main()
