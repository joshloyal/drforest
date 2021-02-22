import csv
from os.path import dirname

from drforest.datasets.base import load_data


__all__ = ['load_cpu_small']


def load_cpu_small():
    """Load and return the cpu small data set (regression).

    The Computer Activity databases are a collection of computer systems
    activity measures. The data was collected from a Sun Sparcstation
    20/712 with 128 Mbytes of memory running in a multi-user university
    department. Users would typically be doing a large variety of tasks
    ranging from accessing the internet, editing files or running very
    cpu-bound programs. The data was collected continuously on two
    separate occasions. On both occassions, system activity was gathered
    every 5 seconds. The final dataset is taken from both occasions with
    equal numbers of observations coming from each collection epoch.

    ===============     ===============
    Samples total                  8192
    Dimensionality                    13
    Features             real, positive
    ===============     ===============

    Returns
    -------
    (data, target) : tuple
        The X feature matrix and the y target vector.
    """
    module_path = dirname(__file__)
    data, target = load_data(module_path, 'cpu_small.csv',
                             is_classification=False)
    return data, target
