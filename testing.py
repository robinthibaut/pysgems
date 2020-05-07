import os
from os.path import join as join_path
import struct
import toolbox

# Define working directory
cwd = os.getcwd()
# Define datasets directory
data_dir = join_path(cwd, 'datasets', 'test')
# Define file name
f_name = 'sgems_dataset.dat'
# Initialize problem
sgems = toolbox.Sgems(data_dir=data_dir, file_name=f_name, dx=5, dy=5)
# Compute data points node and export file
sgems.export_node_idx()
# Display point coordinates and grid
sgems.plot_coordinates()
# Load your algorithm xml file in the 'algorithms' folder
algo_name = sgems.xml_reader('kriging')
# Show xml structure tree
sgems.show_tree()
# Modify xml below:
sgems.xml_update('Hard_Data', {'grid': 'sgems', 'property': 'au'})
sgems.xml_update('Grid_Name', {'value': 'sgems', 'region': ''})
# Write python script
# sgems.write_command()
# Run sgems
# sgems.run()


# /** The Simulacre_input_filter class is a filter that can read the default file
# * format of GsTLAppli. The format is a binary format, with big endian byte
# * order. Following are a description of the file formats for the pointset and
# * the cartesian grid objects. All file formats begin with magic number
# * 0xB211175D, a string indicating the type of object stored in the file, the
# * name of the object, and a version number (Q_INT32). The rest is specific
# * to the object stored:
# *
# *  - point-set:
# * a Q_UINT32 indicating the number of points in the object.
# * a Q_UINT32 indicating the number of properties in the object
# * strings containing the names of the properties
# * the x,y,z coordinates of each point, as floats
# * all the property values, one property at a time, in the order specified
# * by the strings of names, as floats. For each property there are as many
# * values as points in the point-set.
# *
# *  - cartesian grid:
# * 3 Q_UINT32 indicating the number of cells in the x,y,z directions
# * 3 floats for the dimensions of a single cell
# * 3 floats for the origin of the grid
# * a Q_UINT32 indicating the number of properties
# * all the property values, one property at a time, in the order specified
# * by the strings of names, as floats. For each property, there are nx*ny*nz
# * values (nx,ny,nz are the number of cells in the x,y,z directions).
# */

import numpy as np
xy = sgems.xy
xyz = np.c_[xy, np.zeros(len(xy))]

sgems.res_dir = join_path(sgems.cwd, 'results', 'sgems_820c0fc88f7811eabc6adc5360b010ea')
with open(join_path(sgems.res_dir, 'try.sgems'), 'rb') as rb:
    rr = rb.read()


def to_byte(n):
    return bytes([n]).decode()


pp = 'ag'


def write_char(file, char):
    clen = int(len(char) + 1)
    file.write(to_byte(clen))  # len next str + 1
    file.write(char)  # type
    file.write(to_byte(0))  # 0


with open('head.sgems', 'w') as wb:
    wb.write(str(int(1.561792946e+9)))
    [wb.write(to_byte(0)) for _ in range(3)]


with open('intro.sgems', 'w') as wb:
    write_char(wb, 'Point_set')
    [wb.write(to_byte(0)) for _ in range(3)]

    write_char(wb, sgems.project_name)
    [wb.write(to_byte(0)) for _ in range(3)]

    wb.write(to_byte(int(100)))  # version
    [wb.write(to_byte(0)) for _ in range(3)]

    wb.write((to_byte(int(len(xyz)))))  # n data points
    [wb.write(to_byte(0)) for _ in range(3)]

    wb.write(to_byte(int(1)))  # n property
    [wb.write(to_byte(0)) for _ in range(3)]

    write_char(wb, pp)  # property name
    [wb.write(to_byte(0)) for _ in range(3)]


with open('coord.sgems', 'wb') as wb:
    for c in xyz:
        ttt = struct.pack('>fff', c[0], c[1], c[2])
        wb.write(ttt)
        # [wb.write(d.tobytes()) for d in c]
        # [wb.write(bytes([0])) for _ in range(3)]


with open('values.sgems', 'wb') as wb:
    for v in sgems.dataframe[:, 2]:
        wb.write(v)


pf = join_path(data_dir, 'demo.prj')
with open(join_path(pf, 'coordinates.sgems'), 'rb') as cb:
    lol = cb.read()

struct.unpack('fff000i', ttt)

with open(join_path(pf, 'coordinates.sgems'), 'rb') as cb:
    lol = cb.read()

with open(join_path(cwd, 'coord.sgems'), 'rb') as cb:
    lol2 = cb.read()

for i in range(12, len(lol)//12 - 1, 12):
    print((i-12)/12)
    struct.unpack('>fff000i', lol[(i-12):i])

step = 12
for i in range(len(lol)//12):
    print(i)
    print(struct.unpack('>fff', lol[(i*step):((i+1)*12)]))