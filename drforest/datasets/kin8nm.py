import csv
from os.path import dirname

from drforest.datasets.base import load_data


__all__ = ['load_kin8nm']


def load_kin8nm():
    """Load and return the kin8nm data set (regression).

    This data set is concerned with the forward kinematics of an 8 link
    robot arm. This is the 8nm variant, which is known to be highly
    non-linear and medium in noise.

    ===============     ===============
    Samples total                  8192
    Dimensionality                    8
    Features             real, positive
    ===============     ===============

    Returns
    -------
    (data, target) : tuple
        The X feature matrix and the y target vector.
    """
    module_path = dirname(__file__)
    data, target = load_data(module_path, 'kin8nm.csv',
                             is_classification=False)
    return data, target
