from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import pytest

from drforest import datasets
from drforest.ensemble import DimensionReductionForestRegressor


def test_cubic():
    X, y = datasets.make_cubic(random_state=123)

    forest = DimensionReductionForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1).fit(X, y)
