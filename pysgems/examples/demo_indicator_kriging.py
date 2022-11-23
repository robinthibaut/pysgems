#  Copyright (c) 2022. Robin Thibaut, Ghent University

import os
from os.path import join as join_path

from pysgems.algo.sgalgo import XML
from pysgems.dis.sgdis import Discretize
from pysgems.examples.demo_indicator_kriging_mapping import PlotInterpolationMaps
from pysgems.io.sgio import PointSet
from pysgems.plot.sgplots import Plots
from pysgems.sgems import sg


def main():
    # define parameters, calculated outside this project.
    parameter_list = [
        "level_0",
        "level_1",
        "level_2",
        "level_3",
        "level_4",
        "level_5",
        "level_6",
        "level_7",
    ]
    threshold_list = [
        "threshold_0",
        "threshold_1",
        "threshold_2",
        "threshold_3",
        "threshold_4",
        "threshold_5",
        "threshold_6",
        "threshold_7",
        "threshold_8",
        "threshold_9",
        "threshold_10",
    ]
    log_values = True
    level_FIK = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "20"]
    list_levels = [
        "0.6931471805599453",
        "1.0986122886681098",
        "1.3862943611198906",
        "1.6094379124341003",
        "1.8534197411589592",
        "2.4022458416478507",
        "2.616824937969143",
        "2.9092952853407605",
    ]

    # %% Initiate sgems pjt
    cwd = os.getcwd()  # Working directory
    rdir = join_path(cwd, "results", "demo_indicator_kriging")  # Results directory
    algo_dir = join_path(os.path.dirname(cwd), "algorithms")
    algo_XMLs = [
        join_path(algo_dir, "full_indicator_kriging"),
        join_path(algo_dir, "full_indicator_kriging_postKriging"),
    ]
    pjt = sg.Sgems(
        project_name="sgems_FIK",
        project_wd=cwd,
        res_dir=rdir,
        parameters=parameter_list,
        kriging_type="FIK",
        algo_XML_list=algo_XMLs,
    )

    # %% Load data point set
    data_dir = join_path(cwd, "datasets", "demo_indicator_kriging")
    dataset = "sgems_dataset_full_indicator_kriging.eas"
    file_path = join_path(data_dir, dataset)
    print(file_path)
    inputfile = join_path(data_dir, "VMM_PFOA_PFOS_EFSA4_Drinkwater20_2d.csv")

    ps = PointSet(project=pjt, pointset_path=file_path)

    # %% Generate grid. Grid dimensions can automatically be generated based on the data points
    # unless specified otherwise, but cell dimensions dx, dy, (dz) must be specified
    dx_input = 100
    dy_input = 100
    x0_input = 20000
    y0_input = 150000
    x_lim_input = 260000
    y_lim_input = 250000

    ds = Discretize(
        project=pjt,
        dx=dx_input,
        dy=dy_input,
        xo=x0_input,
        yo=y0_input,
        x_lim=x_lim_input,
        y_lim=y_lim_input,
    )

    # %% Display point coordinates and grid
    pl = Plots(project=pjt)
    # pl.plot_coordinates()

    # %% Which feature are available
    # print(pjt.point_set.columns)

    # %% Load your algorithm xml file in the 'algorithms' folder.
    al = XML(project=pjt, algo_dir=algo_dir)
    al.xml_reader(algo_XMLs[0])

    # %% Show xml structure tree
    al.show_tree()

    # %% Modify xml below:
    # By default, the feature grid name of feature X is called 'X_grid'.
    # sgems.xml_update(path, attribute, new value)

    # %% Write binary datasets of needed features
    # sgems.export_01(['f1', 'f2'...'fn'])

    ps.export_01(parameter_list)

    # %% Write python script
    pjt.write_command()

    # %% Run sgems
    pjt.run()

    # plot the results
    result_file_kriging = join_path(rdir, "results.grid")
    outputfolder = join_path(rdir, "plots")
    names = ["ConditionalMean", "ConditionalVariance"] + parameter_list + threshold_list
    pim = PlotInterpolationMaps()
    pim.main(
        result_file_kriging,
        outputfolder,
        dx_input,
        dy_input,
        x0_input,
        y0_input,
        x_lim_input,
        y_lim_input,
        log_values,
        names,
        typekriging="FIK",
        levels_pK=level_FIK,
        indicator_levels=list_levels,
        original_data=inputfile,
        parameter="PFOA",
        data_loc=file_path,
    )


if __name__ == "__main__":
    main()
