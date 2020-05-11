#  Copyright (c) 2020. Robin Thibaut, Ghent University
import os
import uuid
import xml.etree.ElementTree as ET
from os.path import join as jp

from develop.utils import joinlist


class XML:
    def __init__(self,
                 cwd=None,
                 res_dir=None,
                 project_name=None,
                 auto_update=None,
                 columns=None,
                 algo_dir=None):

        self.cwd = cwd
        self.res_dir = res_dir
        self.project_name = project_name
        self.auto_update = auto_update

        dir_path = os.path.abspath(__file__ + "/../")
        self.template_file = jp(dir_path, 'script_templates/script_template.py')  # Python template file path

        self.columns = columns

        self.algo_dir = algo_dir  # ensure proper algorithm directory
        self.op_file = jp(self.algo_dir, 'temp_output.xml')  # Temporary saves a modified XML

        self.tree = None
        self.root = None
        self.object_file_names = []  # Empty object file names if reading new algorithm

    def xml_reader(self, algo_name):
        """
        Reads and parse XML file. It assumes the algorithm XML file is located in the algo_dir folder.
        :param algo_name: Name of the algorithm, without any extension, e.g. 'kriging', 'cokriging'...
        """
        self.algo_dir = jp(self.cwd, 'algorithms')  # ensure proper algorithm directory

        self.op_file = jp(self.algo_dir, 'temp_output.xml')  # Temporary saves a modified XML
        try:
            os.remove(self.op_file)
        except FileNotFoundError:
            pass

        self.tree = ET.parse(jp(self.algo_dir, '{}.xml'.format(algo_name)))
        self.root = self.tree.getroot()
        self.object_file_names = []  # Empty object file names if reading new algorithm

        name = self.root.find('algorithm').attrib['name']  # Algorithm name

        # result directory generated according to project and algorithm name
        if self.res_dir is None:
            # Generate result directory if none is given
            self.res_dir = jp(self.cwd, 'results', '_'.join([self.project_name, name, uuid.uuid1().hex]))
            os.makedirs(self.res_dir)

        # By default, replace the grid name by 'computation_grid', and the name by the algorithm name.
        replace = [['Grid_Name', {'value': 'computation_grid', 'region': ''}],
                   ['Property_Name', {'value': name}]]

        for r in replace:
            try:
                self.xml_update(path=r[0], new_attribute_dict=r[1])
            except AttributeError:
                pass

    def show_tree(self):
        """
        Displays the structure of the XML file, in order to get the path of updatable variables.
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

    def auto_fill(self):
        """
        Ensures binary file of point set are properly generated.
        In case of kriging, cokriging... ensures proper xml attribute names for feature and feature grid.
        This is still quite specific and lots of nested loops, ideally parse all Sgems default XML
        and build proper 'auto_fill' method.
        """

        try:
            elist = []
            for element in self.root:

                elems = list(element)
                c_list = [element.tag]

                elist.append(element)
                trv = list(element.attrib.values())
                trk = list(element.attrib.keys())

                for i in range(len(trv)):
                    if (trv[i] in self.columns) \
                            and ('Variable' or 'Hard_Data' in element.tag):
                        if trv[i] not in self.object_file_names:
                            self.object_file_names.append(trv[i])
                            try:
                                if trk[i - 1] == 'grid':  # ensure default grid name
                                    print(element.attrib)
                                    element.attrib['grid'] = '{}_grid'.format(trv[i])
                                    self.xml_update('//'.join(c_list), 'grid', '{}_grid'.format(trv[i]))
                                    print('>>>')
                                    print(element.attrib)
                                if trk[i - 1] == 'value' and trk[i] == 'property':  # ensure default grid name
                                    print(element.attrib)
                                    element.attrib['value'] = '{}_grid'.format(trv[i])
                                    self.xml_update('//'.join(c_list), 'value', '{}_grid'.format(trv[i]))
                                    print('>>>')
                                    print(element.attrib)
                            except IndexError:
                                pass
                            try:
                                if 'Grid' in elist[-2].tag:
                                    tp = list(elist[-2].attrib.keys())
                                    if 'grid' in tp:
                                        print('//'.join(c_list[:-2]))
                                        print(elist[-2].attrib)
                                        elist[-2].attrib['grid'] = '{}_grid'.format(trv[i])
                                        self.xml_update(elist[-2].tag, 'grid', '{}_grid'.format(trv[i]))
                                        print('>>>')
                                        print(elist[-2].attrib)
                                    if 'value' in tp:
                                        print('//'.join(c_list[:-2]))
                                        print(elist[-2].attrib)
                                        elist[-2].attrib['value'] = '{}_grid'.format(trv[i])
                                        self.xml_update(elist[-2].tag, 'value', '{}_grid'.format(trv[i]))
                                        print('>>>')
                                        print(elist[-2].attrib)
                            except IndexError:
                                pass

                while len(elems) > 0:
                    elems = list(element)
                    for e in elems:
                        c_list.append(e.tag)

                        trv = list(e.attrib.values())
                        trk = list(e.attrib.keys())

                        for i in range(len(trv)):
                            if trv[i] in self.columns:
                                if trv[i] not in self.object_file_names:
                                    self.object_file_names.append(trv[i])
                                    if trk[i] == 'grid':  # ensure default grid name
                                        print('//'.join(c_list))
                                        print(e.attrib)
                                        e.attrib['grid'] = '{}_grid'.format(trv[i])
                                        self.xml_update('//'.join(c_list), 'grid', '{}_grid'.format(trv[i]))
                                        print('>>>')
                                        print(e.attrib)
                                    if trk[i] == 'value':  # ensure default grid name
                                        print('//'.join(c_list))
                                        print(e.attrib)
                                        e.attrib['value'] = '{}_grid'.format(trv[i])
                                        self.xml_update('//'.join(c_list), 'value', '{}_grid'.format(trv[i]))
                                        print('>>>')
                                        print(e.attrib)

                        element = list(e)
                        if len(element) == 0:
                            c_list.pop(-1)
        except TypeError:
            print('No loaded XML file')

    def xml_update(self, path,
                   attribute_name=None,
                   value=None,
                   new_attribute_dict=None,
                   show=1):
        """
        Given a path in the algorithm XML file, changes the corresponding attribute to the new attribute
        :param path: object path
        :param attribute_name: name of the attribute to modify
        :param value: new value for attribute
        :param new_attribute_dict: dictionary defining new attribute
        :param show: whether to display updated xml or not
        """

        if new_attribute_dict is not None:
            if (self.auto_update is True) and ('property' in new_attribute_dict):
                # If one property point set needs to be used
                pp = new_attribute_dict['property']  # Name property
                if pp in self.columns:
                    ps_name = jp(self.res_dir, pp)  # Path of binary file
                    feature = os.path.basename(ps_name)  # If object not already in list
                    if feature not in self.object_file_names:
                        self.object_file_names.append(feature)
                    if 'grid' in new_attribute_dict:  # ensure default grid name
                        new_attribute_dict['grid'] = '{}_grid'.format(pp)
                    if 'value' in new_attribute_dict:  # ensure default grid name
                        new_attribute_dict['value'] = '{}_grid'.format(pp)

            self.root.find(path).attrib = new_attribute_dict
            self.tree.write(self.op_file)

        else:

            self.root.find(path).attrib[attribute_name] = value
            self.tree.write(self.op_file)

        if show:
            print('Updated')
            print(self.root.find(path).tag)
            print(self.root.find(path).attrib)

    def write_command(self):
        """
        Write python script that sgems will run.
        """
        if self.auto_update:
            # First creates necessary binary files
            self.auto_fill()
            self.make_data(self.object_file_names)

        run_algo_flag = ''  # This empty str will replace the # in front of the commands meant to execute sgems
        # within its python environment
        try:
            name = self.root.find('algorithm').attrib['name']  # Algorithm name
            with open(self.op_file) as alx:  # Remove unwanted \n
                algo_xml = alx.read().strip('\n')

        except AttributeError or FileNotFoundError:
            name = 'None'
            algo_xml = 'None'
            run_algo_flag = '#'  # If no algorithm loaded, then just loads the data

        sgrid = [self.ncol, self.nrow, self.nlay,
                 self.dx, self.dy, self.dz,
                 self.xo, self.yo, self.zo]  # Grid information
        grid = joinlist('::', sgrid)  # Grid in sgems format

        sgems_files = [sf + '.sgems' for sf in self.object_file_names]

        # The list below is the list of flags that will be replaced in the sgems python script
        params = [[run_algo_flag, '#'],
                  [self.res_dir.replace('\\', '//'), 'RES_DIR'],  # for sgems convention...
                  [grid, 'GRID'],
                  [self.project_name, 'PROJECT_NAME'],
                  [str(self.hard_data_objects), 'FEATURES_LIST'],
                  ['results', 'FEATURE_OUTPUT'],  # results.grid = output file
                  [name, 'ALGORITHM_NAME'],
                  [name, 'PROPERTY_NAME'],
                  [algo_xml, 'ALGORITHM_XML'],
                  [str(sgems_files), 'OBJECT_FILES'],
                  [self.node_value_file.replace('\\', '//'), 'NODES_VALUES_FILE']]

        with open(self.template_file) as sst:
            template = sst.read()
        for i in range(len(params)):  # Replaces the parameters
            template = template.replace(params[i][1], params[i][0])

        with open(jp(self.res_dir, 'simusgems.py'), 'w') as sstw:  # Write sgems python file
            sstw.write(template)
