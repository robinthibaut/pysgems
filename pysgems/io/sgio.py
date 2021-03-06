#  Copyright (c) 2020. Robin Thibaut, Ghent University
import os
import struct
from os.path import join as jp

from loguru import logger
import numpy as np
import pandas as pd

from pysgems.base.packbase import Package


def datread(file: str = None, start: int = 0, end: int = None):
    # end must be set to None and NOT -1
    """Reads space separated dat file"""
    try:
        with open(file, "r") as fr:
            lines = np.copy(fr.readlines())[start:end]
            try:
                op = np.array([list(map(float, line.split())) for line in lines])
            except ValueError:
                op = [line.split() for line in lines]
    except FileNotFoundError:
        logger.error(f"File {file} not found.")
    return op


def write_point_set(file_name: str, sub_dataframe: pd.DataFrame, nodata: int = -999):
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

    xyz = sub_dataframe[["x", "y", "z"]].to_numpy()

    pp = sub_dataframe.columns[-1]  # Get name of the property

    grid_name = f"{pp}_grid"

    ext = ".sgems"
    if ext not in file_name:
        file_name += ext

    with open(file_name, "wb") as wb:
        wb.write(struct.pack("i", int(1.561792946e9)))  # Magic number
        wb.write(struct.pack(">i", 10))  # Length of 'Point_set' + 1

    with open(file_name, "a") as wb:
        wb.write("Point_set")  # Type of file

    with open(file_name, "ab") as wb:
        wb.write(struct.pack(">b", 0))  # Signed character 0 after str

    with open(file_name, "ab") as wb:
        wb.write(struct.pack(">i", len(grid_name) + 1))  # Length of 'grid' + 1

    with open(file_name, "a") as wb:
        wb.write(grid_name)  # Name of the grid on which points are saved

    with open(file_name, "ab") as wb:
        wb.write(struct.pack(">b", 0))  # Signed character 0 after str

    with open(file_name, "ab") as wb:
        wb.write(struct.pack(">i", 100))  # version number
        wb.write(struct.pack(">i", len(xyz)))  # n data points
        wb.write(struct.pack(">i", 1))  # n property

    with open(file_name, "ab") as wb:
        wb.write(struct.pack(">i", len(pp) + 1))  # Length of property name + 1

    with open(file_name, "a") as wb:
        wb.write(pp)  # Property name

    with open(file_name, "ab") as wb:
        wb.write(struct.pack(">b", 0))  # Signed character 0 after str

    with open(file_name, "ab") as wb:
        for c in xyz:
            ttt = struct.pack(">fff", c[0], c[1], c[2])  # Coordinates x, y, z
            wb.write(ttt)

    with open(file_name, "ab") as wb:
        for v in sub_dataframe[pp]:
            wb.write(struct.pack(">f", v))  # Values


def export_eas(dataframe: pd.DataFrame, filename: str = "dataset"):
    """Exports a Pandas DataFrame to geo-eas format"""
    columns = list(dataframe.columns.values)
    name = os.path.basename(filename).split(".")[0]
    header = [name, str(len(columns))] + columns
    ri = dataframe.iterrows()
    rows = [" ".join(list(map(str, r[1]))) for r in ri]
    lines = header + rows
    with open(filename + ".eas", "w") as eas:
        eas.writelines("\n".join(lines))


class PointSet(Package):
    def __init__(self, project, pointset_path: str = None, force_2d: bool = False):

        # Initiate from parent
        Package.__init__(self, project)

        self.object_file_names = []
        self.project_name = self.parent.project_name
        self.file_path = pointset_path
        self.res_dir = self.parent.res_dir

        self.dimension = 2
        # Load raw data
        self.raw_data, self.project_name, self.columns = self.loader()
        # Put into panda DataFrame
        self.dataframe = pd.DataFrame(data=self.raw_data, columns=self.columns)

        # Assess the dimensionality
        try:
            # If Z (depth) information exists, sets the dimensionality to 3
            self.xyz = self.dataframe[["x", "y", "z"]].to_numpy()
            self.dimension = 3
            logger.info("3D dataset detected.")
            if force_2d:
                logger.info("3D dataset detected, but 2D will be enforced.")
                self.dataframe["z"] = np.zeros(self.dataframe.shape[0])
        except KeyError:  # Assumes 2D dataset
            logger.info("2D dataset detected.")
            # Replaces Z dimension by 0's
            self.dataframe.insert(2, "z", np.zeros(self.dataframe.shape[0]))
            self.columns = list(self.dataframe.columns.values)
            self.xyz = self.dataframe[["x", "y", "z"]].to_numpy()
            self.dimension = 2

        # Set attribute to parent
        setattr(self.parent, "point_set", self)

    # Load sgems dataset
    def loader(self):
        """Parse dataset in GSLIB format"""
        project_info = datread(self.file_path, end=2)  # Name, n features
        project_name = project_info[0][0].lower()  # Project name (lowered)
        # Number of features len([x, y, f1, f2... fn])
        n_features = int(project_info[1][0])
        # Name of features (x, y, z, f1...)
        head = datread(self.file_path, start=2, end=2 + n_features)
        columns_name = [h[0].lower() for h in head]  # Column names (lowered)
        data = datread(self.file_path, start=2 + n_features)  # Raw data
        logger.info("Data loaded.")
        return data, project_name, columns_name

    def export_01(self, features: list = None):
        """
        Gives a list of point set names to be saved in sgems binary format and saves them to the result directory
        :param features: Names of features to export
        """

        if (not isinstance(features, list)) and (features is not None):
            features = [features]
        if features is None:
            features = self.dataframe.columns.values[3:]

        for pp in features:
            # Extract x, y, z, values
            subframe = self.dataframe[["x", "y", "z", pp]]
            ps_name = jp(self.res_dir, pp)  # Path of binary file
            write_point_set(ps_name, subframe)  # Write binary file
            if (pp not in self.parent.object_file_names
                ):  # Adding features name to load them within sgems
                self.parent.object_file_names.append(pp)
            logger.info(f"Feature {pp} exported to binary file")
