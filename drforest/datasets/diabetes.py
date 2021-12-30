import csv
import pandas as pd
import os
import numpy as np

from os.path import dirname

from drforest.datasets.base import load_data


__all__ = ['load_diabetes']


def load_diabetes():
    module_path = dirname(__file__)

    data = pd.read_csv(
        os.path.join(module_path, 'data', 'diabetes.csv'))
    y = data.pop('y').values
    W = data.pop('1').values
    X = data.values

    return np.c_[W, X], y
