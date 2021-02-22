from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import pytest

from drforest import datasets
from drforest import sliced_inverse_regression
from drforest import sliced_average_variance_estimation


SDR_METHODS = [
    sliced_inverse_regression,
    sliced_average_variance_estimation
]


SDR_IDS = ['sir', 'save']


@pytest.mark.parametrize('sdr_method', SDR_METHODS, ids=SDR_IDS)
def test_sir_cubic(sdr_method):
    X, y = datasets.make_cubic(random_state=123)

    directions = sdr_method(X, y)

    true_beta = (1 / np.sqrt(2)) * np.hstack((np.ones(2), np.zeros(8)))
    angle = np.dot(true_beta, directions[0, :])
    np.testing.assert_allclose(np.abs(angle), 1, rtol=1e-1)
