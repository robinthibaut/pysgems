import os
from os.path import join as jp
import xml.etree.ElementTree as ET


# Directories
cwd = os.getcwd()
data_dir = jp(cwd, 'dataset')
res_dir = jp(cwd, 'results')

tree = ET.parse(jp(res_dir, 'algorithm.xml'))
root = tree.getroot()

for child in root.iter():
    print(child.tag)
    print(child.attrib)
    # for t in child:
    #     print(t.tag)
    #     print(t.attrib)

features = root.findall('algorithm')
attribs = [['name', '']]
for f in features:
    for a in attribs:
        f.attrib[a[0]] = a[1]
    print('Updated')
    print(f.attrib)

tree.write(jp(res_dir, 'output.xml'))
