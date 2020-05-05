import os
from os.path import join as jp
import xml.etree.ElementTree as ET


# Directories
cwd = os.getcwd()
data_dir = jp(cwd, 'dataset')
res_dir = jp(cwd, 'results')

tree = ET.parse(jp(res_dir, 'cokriging.xml'))
root = tree.getroot()

name = root.find('algorithm').attrib['name']

for element in root:
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


root.find('Variogram_C11//angles').attrib = {'x': '0', 'y': '0', 'z': '0'}

tree.write(jp(res_dir, 'output.xml'))
