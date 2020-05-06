import os
from os.path import join as join_path
import toolbox

cwd = os.getcwd()
data_dir = join_path(cwd, 'datasets', 'benin')
f_name = 'Dataset_Log_WithoutOutlier_WithoutDouble(LowerThan30m)_Without-4.txt'
sgems = toolbox.Sgems(data_dir=data_dir, file_name=f_name, dx=1000, dy=1000)
sgems.plot_coordinates()
sgems.export_node_idx()
algo_name = sgems.xml_reader('cokriging')
sgems.show_tree()
sgems.write_command()
sgems.run()
