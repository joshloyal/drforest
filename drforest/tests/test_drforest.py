from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import pytest

from drforest import datasets
from drforest.ensemble import DimensionReductionForestRegressor


def test_drforest_smoke():
    X, y = datasets.make_simulation1()

    forest = DimensionReductionForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1).fit(X, y)

    y_pred = forest.predict(X)
    assert y_pred.shape == (1000,)

    imp = forest.local_subspace_importance(np.array([[-1.5, 1.5],
                                                     [0.5, -0.5]]))
    assert imp.shape == (2, 2)
