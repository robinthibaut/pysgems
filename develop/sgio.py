#  Copyright (c) 2020. Robin Thibaut, Ghent University
import struct
from os.path import join as jp

import numpy as np
import pandas as pd


def datread(file=None, start=0, end=None):
    # end must be set to None and NOT -1
    """Reads space separated dat file"""
    with open(file, 'r') as fr:
        lines = np.copy(fr.readlines())[start:end]
        try:
            op = np.array([list(map(float, line.split())) for line in lines])
        except ValueError:
            op = [line.split() for line in lines]
    return op


def write_point_set(file_name, sub_dataframe, nodata=-999):
    # TODO: build similar method to save grid files.
    """
    Function to write sgems binary point set files.

    The Simulacre_input_filter class is a filter that can read the default file
    format of GsTLAppli. The format is a binary format, with big endian byte
    order. Following are a description of the file formats for the pointset and
    the cartesian grid objects. All file formats begin with magic number
    0xB211175D, a string indicating the type of object stored in the file, the
    name of the object, and a version number (Q_INT32). The rest is specific
    to the object stored:

     - point-set:
        a Q_UINT32 indicating the number of points in the object.
        a Q_UINT32 indicating the number of properties in the object
        strings containing the names of the properties
        the x,y,z coordinates of each point, as floats
        all the property values, one property at a time, in the order specified
        by the strings of names, as floats. For each property there are as many
        values as points in the point-set.

     - cartesian grid:
        3 Q_UINT32 indicating the number of cells in the x,y,z directions
        3 floats for the dimensions of a single cell
        3 floats for the origin of the grid
        a Q_UINT32 indicating the number of properties
        all the property values, one property at a time, in the order specified
        by the strings of names, as floats. For each property, there are nx*ny*nz
        values (nx,ny,nz are the number of cells in the x,y,z directions).

    :param nodata: nodata value, rows containing this value are omitted.
    :param file_name:
    :param sub_dataframe: Sub-frame of the feature to be exported [x, y, feature value]
    :return:
    """

    # First, rows with no data occurrence are popped
    sub_dataframe = sub_dataframe[(sub_dataframe != nodata).all(axis=1)]

    xyz = np.vstack(
        (sub_dataframe['x'],
         sub_dataframe['y'],
         np.zeros(len(sub_dataframe)))  # 0 column for z
    ).T  # We need X Y Z coordinates even if working in 2D

    pp = sub_dataframe.columns[-1]  # Get name of the property

    grid_name = '{}_grid'.format(pp)

    ext = '.sgems'
    if ext not in file_name:
        file_name += ext

    with open(file_name, 'wb') as wb:
        wb.write(struct.pack('i', int(1.561792946e+9)))  # Magic number
        wb.write(struct.pack('>i', 10))  # Length of 'Point_set' + 1

    with open(file_name, 'a') as wb:
        wb.write('Point_set')  # Type of file

    with open(file_name, 'ab') as wb:
        wb.write(struct.pack('>b', 0))  # Signed character 0 after str

    with open(file_name, 'ab') as wb:
        wb.write(struct.pack('>i', len(grid_name) + 1))  # Length of 'grid' + 1

    with open(file_name, 'a') as wb:
        wb.write(grid_name)  # Name of the grid on which points are saved

    with open(file_name, 'ab') as wb:
        wb.write(struct.pack('>b', 0))  # Signed character 0 after str

    with open(file_name, 'ab') as wb:
        wb.write(struct.pack('>i', 100))  # version number
        wb.write(struct.pack('>i', len(xyz)))  # n data points
        wb.write(struct.pack('>i', 1))  # n property

    with open(file_name, 'ab') as wb:
        wb.write(struct.pack('>i', len(pp) + 1))  # Length of property name + 1

    with open(file_name, 'a') as wb:
        wb.write(pp)  # Property name

    with open(file_name, 'ab') as wb:
        wb.write(struct.pack('>b', 0))  # Signed character 0 after str

    with open(file_name, 'ab') as wb:
        for c in xyz:
            ttt = struct.pack('>fff', c[0], c[1], c[2])  # Coordinates x, y, z
            wb.write(ttt)

    with open(file_name, 'ab') as wb:
        for v in sub_dataframe[pp]:
            wb.write(struct.pack('>f', v))  # Values


class PointSet:

    def __init__(self,
                 model,
                 pointset_path=None):

        self.model = model
        self.object_file_names = []
        self.project_name = self.model.model_name
        self.file_path = pointset_path
        self.res_dir = self.model.res_dir

        self.raw_data, self.project_name, self.columns = self.loader()
        self.dataframe = pd.DataFrame(data=self.raw_data, columns=self.columns)
        try:
            self.xyz = self.dataframe[['x', 'y', 'z']].to_numpy()
        except KeyError:  # Assumes 2D dataset
            self.dataframe.insert(2, 'z', np.zeros(self.dataframe.shape[0]))
            self.columns = list(self.dataframe.columns.values)
            self.xyz = self.dataframe[['x', 'y', 'z']].to_numpy()
            self.dz = 0

        self.model.point_set = self

    # Load sgems dataset
    def loader(self):
        """Parse dataset in GSLIB format"""
        project_info = datread(self.file_path, end=2)  # Name, n features
        project_name = project_info[0][0].lower()  # Project name (lowered)
        n_features = int(project_info[1][0])  # Number of features len([x, y, f1, f2... fn])
        head = datread(self.file_path, start=2, end=2 + n_features)  # Name of features (x, y, z, f1...)
        columns_name = [h[0].lower() for h in head]  # Column names (lowered)
        data = datread(self.file_path, start=2 + n_features)  # Raw data
        return data, project_name, columns_name

    def load_dataframe(self):
        """
        Loads sgems data set.
        Assumes that x, y, z are the first three columns and are labeled as such.
        """
        self.raw_data, self.project_name, self.columns = self.loader()
        self.dataframe = pd.DataFrame(data=self.raw_data, columns=self.columns)
        try:
            self.xyz = self.dataframe[['x', 'y', 'z']].to_numpy()
        except KeyError:  # Assumes 2D dataset
            self.dataframe.insert(2, 'z', np.zeros(self.dataframe.shape[0]))
            self.columns = list(self.dataframe.columns.values)
            self.xyz = self.dataframe[['x', 'y', 'z']].to_numpy()
            self.dz = 0

        self.model.point_set = self

    def export_01(self, features=None):
        """
        Gives a list of point set names to be saved in sgems binary format and saves them to the result directory
        :param features: Names of features to export
        """

        if (not isinstance(features, list)) and (features is not None):
            features = [features]
        else:
            features = self.dataframe.columns.values

        for pp in features:
            subframe = self.dataframe[['x', 'y', pp]]  # Extract x, y, values
            ps_name = jp(self.res_dir, pp)  # Path of binary file
            write_point_set(ps_name, subframe)  # Write binary file
            if pp not in self.model.object_file_names:  # Adding features name to load them within sgems
                self.model.object_file_names.append(pp)
