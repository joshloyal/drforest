import csv
from os.path import dirname

from drforest.datasets.base import load_data


__all__ = ['load_fishcatch']


def load_fishcatch():
    """Missing values were imputed with -1!"""
    module_path = dirname(__file__)
    data, target = load_data(module_path, 'fishcatch.csv',
                             is_classification=False)
    return data, target
