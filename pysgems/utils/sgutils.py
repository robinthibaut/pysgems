#  Copyright (c) 2020. Robin Thibaut, Ghent University


def joinlist(j, mylist):
    """
    Function that joins an array of numbers with j as separator. For example, joinlist('^', [1,2]) returns 1^2

    :param j: separator
    :param mylist: list of numbers
    """
    gp = j.join(map(str, mylist))

    return gp
