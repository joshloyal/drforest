import csv
import pandas as pd
import os

from os.path import dirname

from drforest.datasets.base import load_data


__all__ = ['load_athletes']


def load_athletes():
    """Load and return the australian athletes data set (regression).

    This dataset was collected in a study of how data on various
    characteristics of the blood varied with sport body size and sex of the
    athlete.

    Here the goal is to predict lean body mass (lbm) based on the logarithm
    of various attributes of the athelete.

    ===============     ===============
    Samples total                   202
    Dimensionality                    9
    Features             real and categorical, positive
    ===============     ===============

    Returns
    -------
    (data, target) : tuple
        The X feature matrix and the y target vector.

    References
    ----------

    [1] Telford, R.D. and Cunningham, R.B. 1991.
        "Sex, sport and body-size dependency of hematology in highly
        trained athletes", Medicine and Science in Sports and
        Exercise 23: 788-794.
    """
    module_path = dirname(__file__)

    data = pd.read_csv(
        os.path.join(module_path, 'data', 'athletes_categorical.csv'))
    y = data.pop('LBM').values
    X = data.values

    return X, y
