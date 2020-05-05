import os
from os.path import join as jp
import xml.etree.ElementTree as ET


# Directories
cwd = os.getcwd()
data_dir = jp(cwd, 'dataset')
res_dir = jp(cwd, 'results')

tree = ET.parse(jp(res_dir, 'algorithm.xml'))
root = tree.getroot()

name = root.find('algorithm').attrib['name']

for element in root:
    print(element.tag)
    print(element.attrib)
    elems = list(element)
    while len(elems) > 0:
        c_list = [element.tag]
        for e in elems:
            c_list.append(e.tag)
            print('//'.join(c_list))
            print(e.attrib)
            elems = list(e)
            c_list.pop(-1)


tree.write(jp(res_dir, 'output.xml'))
