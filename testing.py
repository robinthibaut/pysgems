import toolbox


f_name = 'Dataset_Log_WithoutOutlier_WithoutDouble(LowerThan30m)_Without-4.txt'
sgems = toolbox.Sgems(file_name=f_name, dx=5000, dy=5000)
sgems.plot_coordinates()
sgems.export_node_idx()
algo_name = sgems.xml_reader('cokriging')
sgems.show_tree()
sgems.write_command()
sgems.run()
