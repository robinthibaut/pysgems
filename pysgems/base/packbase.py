#  Copyright (c) 2020. Robin Thibaut, Ghent University

import abc


class PackageInterface(object):
    @property
    @abc.abstractmethod
    def parent(self):
        raise NotImplementedError(
            'must define get_model_dim_arrays in child '
            'class to use this base class')

    @parent.setter
    @abc.abstractmethod
    def parent(self, name):
        raise NotImplementedError(
            'must define get_model_dim_arrays in child '
            'class to use this base class')


class Package(PackageInterface):
    """
    Base package class from which most other packages are derived.

    """

    def __init__(self, parent):
        """
        Package init

        """
        # To be able to access the parent project object's attributes
        self.parent = parent

        return

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent
