import csv
from os.path import dirname

from drforest.datasets.base import load_data


__all__ = ['load_abalone']


def load_abalone(include_categoricals=True):
    """Load and return the abalone data set (regression).

    The task of the abalone dataset is to predict the age of abalone based on
    several measurments. In the dataset, eight candidate measurments including
    sex, dimensions, and various weights are reported along with the number of
    rings of the abalone as predictor variables. Note that sex is the only
    categorical variable.

    Here the goal is to predict the number of rings of the abalone based on
    the other measurements.

    ===============     ===============
    Samples total                  4117
    Dimensionality                    8
    Features             real, positive
    ===============     ===============

    Returns
    -------
    (data, target) : tuple
        The X feature matrix and the y target vector.

    References
    ----------

    [1] Warwick J Nash, Tracy L Sellers, Simon R Talbot, Andrew J Cawthorn and
        Wes B Ford (1994)  "The Population Biology of Abalone
        (_Haliotis_ species) in Tasmania. I. Blacklip Abalone (_H. rubra_)
        from the North Coast and Islands of Bass Strait",
        Sea Fisheries Division, Technical Report No. 48 (ISSN 1034-3288)
    """
    module_path = dirname(__file__)
    data, target = load_data(module_path, 'abalone.csv',
                             is_classification=False)

    if not include_categoricals:
        # drop the first column which is the sex variable
        data = data[:, 1:]

    return data, target
