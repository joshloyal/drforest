[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/joshloyal/drforest/blob/master/LICENSE)
<!--[![Travis](https://travis-ci.com/joshloyal/dynetlsm.svg?token=gTKqq3zSsip89mhYVQPZ&branch=master)](https://travis-ci.com/joshloyal/dynetlsm)
[![AppVeyor](https://ci.appveyor.com/api/projects/status/github/joshloyal/dynetlsm)](https://ci.appveyor.com/project/joshloyal/dynetlsm/history)
[![PyPI Latest Release](https://img.shields.io/pypi/v/dynetlsm)](https://pypi.org/project/dynetlsm/)-->

# Dimension Reduction Forests

*Author: [Joshua D. Loyal](https://joshloyal.github.io/)*

This package provides the statistical estimation methods for dimension reduction forests and local subspace variable importances described in "Dimension Reduction Forests: Local Variable Importance using Structured Random Forests".

BibTeX reference to cite, if you use this package:
<!--```bibtex
@article{loyal2020hdplpcm,
}
```
--->

Dependencies
------------
``drforest`` requires:

- Python (>= 3.7)

and the requirements highlighted in [requirements.txt](requirements.txt).

Installation
------------
You need a working installation of numpy and scipy to install ``drforest``. If you have a working installation of numpy and scipy, the easiest way to install ``drforest`` is using ``pip``:

```
pip install -U drforest
```

If you prefer, you can clone the repository and run the setup.py file. Note that the package uses OpenMP, so you will need a C/C++ compiler with OpenMP support. In addition, we use ``pthreads``, which means the package currently does not support Windows.

Use the following commands to get the copy from GitHub and install all the dependencies:

```
git clone https://github.com/joshloyal/drforest.git
cd drforest
pip install .
```

Or install using pip and GitHub:

```
pip install -U git+https://github.com/joshloyal/drforest.git
```

Background
----------

Write about some background here.


Examples
--------
As an example, we consider the following regression function
<p align="center">
<img src="/images/simulation1.png" alt="Simulation 1" width="600">
</p>
where we generate five covariates uniformly on [-3, 3]. In this scenario there are two informative covariates and three uninformative covariates. Projected onto the first two coordinates, the regression surface looks like


To fit a dimension reduction forest, we simulate the data,  initialize the estimator, and call fit:
```python
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from drforest.datasets import make_simulation1
from drforest.ensemble import DimensionReductionForestRegressor

X, y = make_simulation1(
    n_samples=n_samples, noise=1, n_features=5, random_state=123)

X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=42)

drforest = DimensionReductionForestRegressor(
    n_estimators=500, min_samples_leaf=1, n_jobs=-1).fit(X, y)

y_pred = drforest.predict(X_test)
print('test MSE: ', mean_squared_error(y_test, y_pred))
>>> 3.79
print('OOB MSE: ', drforest.oob_mse_)
>>> 4.13
```

```python
from drforest.plots import plot_local_importance

importances = drforest.local_subspace_importance(X, n_jobs=-1)

plot_local_importance(importances)
```
<p align="center">
<img src="/images/lsvi_example.png" alt="local subspace variable importances" width="600">
</p>


Simulation Studies and Real-Data Applications
---------------------------------------------
This package includes the simulation studies and real-data applications found in Loyal et al. (2021)<sup>[[6]](#References)</sup>:

* A simple example of local variable importance on a synthetic dataset: ([here](/examples/random_forest_importances.py)).
* A comparison with other methods on synthetic data: ([here](/examples/synthetic_data.py)).
* A evaluation of the performance of local subspace variable importance estimates on synthetic data: ([here](/examples/local_subspace_importances.py)).
* A comparison with other methods on real regression problems: ([here](/examples/real_data.py)).
* An appliction using LSVIs to infer meteorological factors contributing to  pollution in Beijing, China ([here](/examples/beijing_air_quality.py)).

We also provide a few [jupyter notebooks](/notebooks) that demonstrate the use of this package.

References
----------

[1]:
