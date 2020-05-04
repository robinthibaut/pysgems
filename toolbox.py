import numpy as np
from scipy.spatial import distance_matrix


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


def centroids_from_rc(rows, columns, xo, yo):

    nrow = len(rows)
    ncol = len(columns)
    delr = rows
    delc = columns
    r_sum = np.cumsum(delr) + yo
    c_sum = np.cumsum(delc) + xo

    # centers = []
    # for c in range(nrow):
    #     for n in range(ncol):
    #         b = [[c_sum[n] - delc[n], r_sum[c] - delr[c]],
    #              [c_sum[n] - delc[n], r_sum[c]],
    #              [c_sum[n], r_sum[c]],
    #              [c_sum[n], r_sum[c] - delr[c]]]
    #         centers.append(np.mean(b, axis=0))
    # return np.array(centers)

    for c in range(nrow):
        for n in range(ncol):
            b = [[c_sum[n] - delc[n], r_sum[c] - delr[c]],
                 [c_sum[n] - delc[n], r_sum[c]],
                 [c_sum[n], r_sum[c]],
                 [c_sum[n], r_sum[c] - delr[c]]]
            yield np.mean(b, axis=0)


def my_node(xy, centers):
    """
    :param centers:
    :param xy: [x, y]
    :return:
    """
    rn = np.array([xy])
    dm = distance_matrix(rn, centers).flatten()

    cell = np.where(dm == np.amin(dm))
    return cell[0][0]

