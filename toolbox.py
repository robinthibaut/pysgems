import numpy as np
import time
from shapely.geometry import Point, Polygon


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

    :param rows: array of x-widths along a row
    :param columns: array of y-widths along a column
    :param xo:
    :param yo:
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
        return int(i*ncol + j)

    for c in range(nrow):
        for n in range(ncol):
            b = [[c_sum[n] - delc[n], r_sum[c] - delr[c]],
                 [c_sum[n] - delc[n], r_sum[c]],
                 [c_sum[n], r_sum[c]],
                 [c_sum[n], r_sum[c] - delr[c]]]
            yield get_node(c, n), np.array(b), np.mean(b, axis=0)


def my_node(xy, rows, columns, xo, yo):
    """
    Given a point coordinate xy [x, y], computes its node number by computing the euclidean distance of each cell
    center.
    :param xy:
    :param rows: array of x-widths along a row
    :param columns: array of y-widths along a column
    :param xo: x origin
    :param yo: y origin
    :return:
    """

    rn = np.array(xy)
    blocks = blocks_from_rc(rows, columns, xo, yo)
    # vmin = np.inf
    # cell = None
    # for b in blocks:
    #     c = b[2]
    #     dc = np.linalg.norm(rn - c)  # Euclidean distance
    #     if vmin > dc:
    #         vmin = dc
    #         cell = b[0]
    for b in blocks:
        poly = Polygon(b[1])
        p = Point(rn)
        if p.within(poly):
            cell = b[0]
            return cell

    # return cell

