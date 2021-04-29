import numpy as np
import pandas as pd
import shutil
import os
from loguru import logger

def main(inputfile, dx_input, dy_input, x0_input, y0_input, x_lim_input, y_lim_input, outputfolder):
    logger.info(f'Adding the coordinates to the kriging results.')
    inputfile = inputfile
    inputfile = inputfile.replace("\\", "/")
    inputfile = inputfile.replace('"', '')
    inputfilename = inputfile.split('resultsdata2d')
    a = inputfilename[1]
    b = inputfilename[2]
    b = b.strip('/')
    inputfilename = a + 'resultsdata2d' + b
    outputfolder = outputfolder.replace("\\", "/")
    outputfolder = outputfolder.replace('"', '')

    nx = (x_lim_input-x0_input)/dx_input
    ny = (y_lim_input-y0_input)/dy_input
    nz = 1
    dx = dx_input
    dy = dy_input
    dz = 0
    x0 = x0_input #+ (dx_input/2)
    y0 = y0_input #+ (dy_input/2)
    z0 = 0

    values = []

    with open(inputfile, "r") as f:
        rows = f.readlines()[1:]


    new_file = open(inputfile + "withouthead.txt", "w")
    for line in rows:
        for i in line:
            if i != ' ':
                None
            else:
                new_file.write(line)
                break

    new_file.close()
    inputfile = new_file.name

    file = open(inputfile, 'r')
    lines = file.readlines()
    for line in lines:
        values.extend([line])

    length = len(lines)
    ind = list(range(1, length+1))

    coordinates = []

    for i in ind:
        iz = 1 + int((i-1)/(nx*ny))
        iy = 1 + int( (i-(iz-1)*nx*ny)/nx)
        ix = i - (iz-1)*nx*ny - (iy-1)*nx
        if ix == 0:
            x = x0 + (dx/2) + (nx-1)*dx
            y = y
            z = z
            coordinate = [x,y,z]
            coordinates.extend([coordinate])
        else:
            x = x0 + (dx/2) + (ix-1)*dx
            y = y0 + (dy/2) + (iy-1)*dy
            z = z0 + (dz/2) + (iz-1)*dz
            coordinate = [x,y,z]
            coordinates.extend([coordinate])

    file.close()

    ind2 = list(range(0,length))
    results = []
    for i in ind2:
        a = [float(values[i])]
        list1 = coordinates[i] + a
        results.extend([list1])

    np.savetxt(outputfolder + inputfilename + '-xyz_value.txt', results, delimiter="\t", fmt="%s")
    logger.info(f'Saved the total results of kriging in the QGiS folder.')

    results_filtered = []
    for i in results:
        if i[3] == -9966699.0:
            None
        else:
            results_filtered.extend([i])

    if len(results) == len(results_filtered):
        None
    else:
        np.savetxt(outputfolder + inputfilename + '-xyz_value_filtered.txt', results_filtered, delimiter="\t", fmt="%s")
        logger.info(f'Saved the filtered results of kriging in the QGiS folder.')
