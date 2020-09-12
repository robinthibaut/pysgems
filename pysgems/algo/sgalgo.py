#  Copyright (c) 2020. Robin Thibaut, Ghent University
import os
import uuid
import xml.etree.ElementTree as ET
from os.path import join as jp

from pysgems.base.packbase import Package

auto_update = False


class XML(Package):
    def __init__(self,
                 project,
                 algo_dir=None):

        Package.__init__(self, project)

        self.project_name = self.parent.project_name

        self.cwd = self.parent.project_wd
        self.res_dir = self.parent.res_dir
        self.algo_dir = algo_dir
        if self.algo_dir is None:
            self.algo_dir = jp(self.cwd, 'algorithms')

        self.op_file = jp(self.algo_dir, f'{uuid.uuid4().hex}.xml')  # Temporary saves a modified XML
        self.tree = None
        self.root = None

        setattr(self.parent, 'algo', self)

    def xml_reader(self, algo_name):
        """
        Reads and parse XML file. It assumes the algorithm XML file is located in the algo_dir folder.
        :param algo_name: Name of the algorithm, without any extension, e.g. 'kriging', 'cokriging'...
        """
        try:
            os.remove(self.op_file)
        except FileNotFoundError:
            pass

        self.tree = ET.parse(jp(self.algo_dir, '{}.xml'.format(algo_name)))
        self.root = self.tree.getroot()

        name = self.root.find('algorithm').attrib['name']  # Algorithm name

        # By default, replace the grid name by 'computation_grid', and the name by the algorithm name.
        replace = [['Grid_Name', {'value': 'computation_grid', 'region': ''}],
                   ['Property_Name', {'value': name}]]

        for r in replace:
            try:
                self.xml_update(path=r[0], new_attribute_dict=r[1])
            except AttributeError:
                pass

        setattr(self.parent.algo, 'tree', self.tree)
        setattr(self.parent.algo, 'root', self.root)

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
            if (auto_update is True) and ('property' in new_attribute_dict):
                # If one property point set needs to be used
                pp = new_attribute_dict['property']  # Name property
                if pp in self.parent.point_set.columns:
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

        setattr(self.parent.algo, 'tree', self.tree)
        setattr(self.parent.algo, 'root', self.root)

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
                    if (trv[i] in self.parent.point_set.columns) \
                            and ('Variable' or 'Hard_Data' in element.tag):
                        if trv[i] not in self.parent.object_file_names:
                            self.parent.object_file_names.append(trv[i])
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
                            if trv[i] in self.parent.point_set.columns:
                                if trv[i] not in self.parent.object_file_names:
                                    self.parent.object_file_names.append(trv[i])
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



