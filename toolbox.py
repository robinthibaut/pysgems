import numpy as np
import time
import os
from os.path import join as jp
import shutil
import uuid
import subprocess
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def clockwork(func):
    def wrapper(*args, **kwargs):
        startTime = time.time()
        func(*args, **kwargs)
        endTime = time.time()
        print("time: ", endTime - startTime, " seconds")

    return wrapper


def datread(file=None, start=0, end=-1):
    """Reads space separated dat file"""
    with open(file, 'r') as fr:
        lines = np.copy(fr.readlines())
        try:
            op = np.array([list(map(float, line.split())) for line in lines[start:end]])
        except ValueError:
            op = [line.split() for line in lines[start:end]]
    return op


def joinlist(j, mylist):
    """
    Function that joins an array of numbers with j as separator. For example, joinlist('^', [1,2]) returns 1^2
    """
    gp = j.join(map(str, mylist))

    return gp


def blocks_from_rc(rows, columns, xo, yo):
    """
    Yields blocks defining grid cells
    :param rows: array of x-widths along a row
    :param columns: array of y-widths along a column
    :param xo: x origin
    :param yo: y origin
    :return: generator of (cell node number, block vertices coordinates, block center)
    """
    nrow = len(rows)
    ncol = len(columns)
    delr = rows
    delc = columns
    r_sum = np.cumsum(delr) + yo
    c_sum = np.cumsum(delc) + xo

    def get_node(i, j):
        """
        Get node index to fixed hard data
        :param i: row number
        :param j: column number
        :return: node number
        """
        return int((i+1) * ncol + j) + 1

    for c in range(nrow):
        for n in range(ncol):
            b = [[c_sum[n] - delc[n], r_sum[c] - delr[c]],
                 [c_sum[n] - delc[n], r_sum[c]],
                 [c_sum[n], r_sum[c]],
                 [c_sum[n], r_sum[c] - delr[c]]]
            yield get_node(c, n), np.array(b), np.mean(b, axis=0)


class Sgems:

    def __init__(self, data_dir, file_name, dx, dy, xo=None, yo=None, x_lim=None, y_lim=None):

        # Directories
        self.cwd = os.getcwd()
        self.algo_dir = jp(self.cwd, 'algorithms')
        self.data_dir = data_dir
        self.file_name = jp(self.data_dir, file_name)
        self.node_file = jp(self.data_dir, 'nodes.npy')
        self.node_value_file = jp(self.data_dir, 'fnodes.txt')
        self.dis_file = jp(self.data_dir, 'dis.info')

        # Data
        self.dataframe, self.project_name, self.columns = self.loader()
        self.xy = np.vstack((self.dataframe[:, 0], self.dataframe[:, 1])).T  # X, Y coordinates
        self.nodata = -999

        # Generate result directory
        self.res_dir = jp(self.cwd, 'results', '_'.join([self.project_name, uuid.uuid1().hex]))
        os.makedirs(self.res_dir)

        # Grid geometry
        self.dx = dx  # Block x-dimension
        self.dy = dy  # Block y-dimension
        self.dz = 0  # Block z-dimension
        self.xo, self.yo, self.x_lim, self.y_lim, self.nrow, self.ncol, self.nlay, self.along_r, self.along_c \
            = self.grid(dx, dy, xo, yo, x_lim, y_lim)
        self.bounding_box = Polygon([[self.xo, self.yo],
                                     [self.x_lim, self.yo],
                                     [self.x_lim, self.y_lim],
                                     [self.xo, self.y_lim]])

        # Algorithm
        self.tree = None
        self.root = None
        self.op_file = jp(self.res_dir, 'output.xml')
        try:
            os.remove(self.op_file)
        except FileNotFoundError:
            pass

    # Load sgems dataset
    def loader(self):
        """Parse sgems dataset"""
        project_info = datread(self.file_name, end=2)  # Name, n features
        project_name = project_info[0][0].lower()  # Project name
        n_features = int(project_info[1][0])  # Number of features len([x, y, f1, f2... fn])
        head = datread(self.file_name, start=2, end=2 + n_features)  # Name of features
        columns_name = [h[0].lower() for h in head]
        data = datread(self.file_name, start=2 + n_features)  # Raw data

        return data, project_name, columns_name

    def load_dataframe(self):
        """Loads sgems data set"""
        # At this time, considers only 2D dataset
        self.dataframe, self.project_name, self.columns = self.loader()
        self.xy = np.vstack((self.dataframe[:, 0], self.dataframe[:, 1])).T  # X, Y coordinates

    def grid(self, dx, dy, xo=None, yo=None, x_lim=None, y_lim=None):
        """
        Constructs the grid geometry. The user can not control directly the number of rows and columns
        but instead inputs the cell size in x and y dimensions.
        xo, yo, x_lim, y_lim, defining the bounding box of the grid, are None by default, and are computed
        based on the data points distribution.
        :param dx:
        :param dy:
        :param xo:
        :param yo:
        :param x_lim:
        :param y_lim:
        :return:
        """

        if x_lim is None and y_lim is None:
            x_lim, y_lim = np.round(np.max(self.xy, axis=0)) + np.array([dx, dy]) * 4  # X max, Y max
        if xo is None and yo is None:
            xo, yo = np.round(np.min(self.xy, axis=0)) - np.array([dx, dy]) * 4  # X min, Y min

        nrow = int((y_lim - yo) // dy)  # Number of rows
        ncol = int((x_lim - xo) // dx)  # Number of columns
        nlay = 1  # Number of layers
        along_r = np.ones(ncol) * dx  # Size of each cell along y-dimension - rows
        along_c = np.ones(nrow) * dy  # Size of each cell along x-dimension - columns

        npar = np.array([dx, dy, xo, yo, x_lim, y_lim, nrow, ncol, nlay])

        if os.path.isfile(self.dis_file):  # Check previous grid info
            pdis = np.loadtxt(self.dis_file)
            # If different, recompute data points node by deleting previous node file
            if not np.array_equal(pdis, npar):
                print('New grid found')
                try:
                    os.remove(self.node_file)
                    os.remove(self.node_value_file)
                except FileNotFoundError:
                    pass
                finally:
                    np.savetxt(self.dis_file, npar)
            else:
                print('Using previous grid')
        else:
            np.savetxt(self.dis_file, npar)

        return xo, yo, x_lim, y_lim, nrow, ncol, nlay, along_r, along_c

    def plot_coordinates(self):
        plt.plot(self.dataframe[:, 0], self.dataframe[:, 1], 'ko')
        plt.xticks(np.cumsum(self.along_r) + self.xo - self.dx, labels=[])
        plt.yticks(np.cumsum(self.along_c) + self.yo - self.dy, labels=[])
        plt.grid('blue')
        plt.show()

    def my_node(self, xy):
        """
        Given a point coordinate xy [x, y], computes its node number by computing the euclidean distance of each cell
        center.
        :param xy:  x, y coordinate of data point
        :return:
        """
        start = time.time()
        rn = np.array(xy)
        # first check if point is within the grid
        p = Point(rn)

        if p.within(self.bounding_box):
            dmin = np.min([self.along_c.min(), self.along_r.min()]) / 2
            blocks = blocks_from_rc(self.along_c, self.along_r, self.xo, self.yo)
            vmin = np.inf
            cell = None
            for b in blocks:
                c = b[2]
                dc = np.linalg.norm(rn - c)  # Euclidean distance
                if dc <= dmin:  # If point is inside cell
                    print('found 1 node in {} s'.format(time.time()-start))
                    return b[0]
                if dc < vmin:
                    vmin = dc
                    cell = b[0]
            print('found 1 node in {} s'.format(time.time() - start))
            return cell
        else:
            return -999

    def compute_nodes(self):
        """
        Determines node location for each data point.
        It is necessary to know the node number to assign the hard data property to the sgems grid.
        :return: nodes number
        """
        nodes = np.array([self.my_node(c) for c in self.xy])

        np.save(self.node_file, nodes)  # Save to nodes to avoid recomputing each time

        return nodes

    def get_nodes(self):
        try:
            d_nodes = np.load(self.node_file)
        except FileNotFoundError:
            d_nodes = self.compute_nodes()

        return d_nodes

    def cleanup(self):
        """
        Removes no-data rows from data frame and compute the mean of data points sharing the same cell.
        :return: Filtered list of each attribute
        """
        data_nodes = self.get_nodes()
        unique_nodes = list(set(data_nodes))

        fn = []
        for h in range(2, len(self.columns)):  # For each feature
            # fixed nodes = [[node i, value i]....]
            fixed_nodes = np.array([[data_nodes[dn], self.dataframe[:, h][dn]] for dn in range(len(data_nodes))])
            # Deletes points where val == nodata
            hard_data = np.delete(fixed_nodes, np.where(fixed_nodes == self.nodata)[0], axis=0)
            # If data points share the same cell, compute their mean and assign the value to the cell
            for n in unique_nodes:
                where = np.where(hard_data[:, 0] == n)[0]
                if len(where) > 1:  # If more than 1 point per cell
                    mean = np.mean(hard_data[where, 1])
                    hard_data[where, 1] = mean

            fn.append(hard_data.tolist())

        return fn

    # Save node list to load it into sgems later
    def export_node_idx(self):
        """
        Export the list of shape (n features, m nodes, 2) containing the node of each point data with the corresponding
        value, for each feature
        """
        if not os.path.isfile(self.node_value_file):
            hard = self.cleanup()
            with open(self.node_value_file, 'w') as nd:
                nd.write(repr(hard))
            shutil.copyfile(self.node_value_file, self.node_value_file.replace(self.data_dir, self.res_dir))

    def xml_reader(self, algo_name):
        """
        Reads and parse XML file. It assumes the algorithm XML file is located in the algo_dir folder.
        :param algo_name: Name of the algorithm, without any extension, e.g. 'kriging', 'cokriging'...
        """
        self.tree = ET.parse(jp(self.algo_dir, '{}.xml'.format(algo_name)))
        self.root = self.tree.getroot()

        name = self.root.find('algorithm').attrib['name']

        replace = [['Primary_Harddata_Grid', {'value': self.project_name, 'region': ''}],
                   ['Secondary_Harddata_Grid', {'value': self.project_name, 'region': ''}],
                   ['Grid_Name', {'value': 'computation_grid', 'region': ''}],
                   ['Property_Name', {'value': name}],
                   ['Hard_Data', {'grid': self.project_name, 'property': "hard"}]]

        for r in replace:
            try:
                self.xml_update(r[0], r[1])
            except AttributeError:
                pass

    def show_tree(self):
        """
        Displays the structure of the XML file, in order to get the path of updatable variables
        """
        try:
            for element in self.root:
                print(element.tag)
                print(element.attrib)
                elems = list(element)
                c_list = [element.tag]
                while len(elems) > 0:
                    elems = list(element)
                    for e in elems:
                        c_list.append(e.tag)
                        print('//'.join(c_list))
                        print(e.attrib)
                        element = list(e)
                        if len(element) == 0:
                            c_list.pop(-1)
        except TypeError:
            print('No loaded XML file')

    def xml_update(self, path, new_attribute):
        """
        Given a path in the algorithm XML file, changes the corresponding attribute to the new attribute
        :param path:
        :param new_attribute:
        """
        self.root.find(path).attrib = new_attribute
        self.tree.write(self.op_file)

    def write_command(self):
        """
        Write python script that sgems will run
        """

        run_algo_flag = ''
        try:
            name = self.root.find('algorithm').attrib['name']
            with open(self.op_file) as alx:
                algo_xml = alx.read().strip('\n')

        except AttributeError or FileNotFoundError:
            name = 'None'
            algo_xml = 'None'
            run_algo_flag = '#'  # If no algorithm loaded, then just loads the data

        sgrid = [self.ncol, self.nrow, self.nlay,
                 self.dx, self.dy, self.dz,
                 self.xo, self.yo, 0]  # Grid information
        grid = joinlist('::', sgrid)

        params = [[run_algo_flag, '#'],
                  [self.res_dir.replace('\\', '//'), 'RES_DIR'],
                  [grid, 'GRID'],
                  [self.project_name, 'PROJECT_NAME'],
                  [str(self.columns[2:]), 'FEATURES_LIST'],
                  ['results', 'FEATURE_OUTPUT'],
                  [name, 'ALGORITHM_NAME'],
                  [name, 'PROPERTY_NAME'],
                  [algo_xml, 'ALGORITHM_XML'],
                  [self.node_value_file.replace('\\', '//'), 'NODES_VALUES_FILE']]

        with open('simusgems_template.py') as sst:
            template = sst.read()

        for i in range(len(params)):  # Replaces the parameters
            template = template.replace(params[i][1], params[i][0])

        with open(jp(self.res_dir, 'simusgems.py'), 'w') as sstw:
            sstw.write(template)

    def script_file(self):
        # Create script file
        run_script = jp(self.res_dir, 'sgems.script')
        rscpt = open(run_script, 'w')
        rscpt.write(' '.join(['RunScript', jp(self.res_dir, 'simusgems.py')]))
        rscpt.close()

    def bat_file(self):

        if not os.path.isfile(jp(self.res_dir, 'sgems.script')):
            self.script_file()

        batch = jp(self.res_dir, 'RunSgems.bat')
        bat = open(batch, 'w')
        bat.write(' '.join(['cd', self.res_dir, '\n']))
        bat.write(' '.join(['sgems', 'sgems.script']))
        bat.close()

    def run(self):

        batch = jp(self.res_dir, 'RunSgems.bat')
        if not os.path.isfile(batch):
            self.bat_file()
        start = time.time()
        subprocess.call([batch])  # Opens the BAT file
        print('ran algorithm in {} s'.format(time.time()-start))

