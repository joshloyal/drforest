[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/joshloyal/drforest/blob/master/LICENSE)
<!--[![Travis](https://travis-ci.com/joshloyal/drforest.svg?token=gTKqq3zSsip89mhYVQPZ&branch=master)](https://travis-ci.com/joshloyal/drforest)
[![PyPI Latest Release](https://img.shields.io/pypi/v/drforest)](https://pypi.org/project/drforest/)-->

# Dimension Reduction Forests

*Author: [Joshua D. Loyal](https://joshloyal.github.io/)*

This package provides the statistical estimation methods for dimension reduction forests and local principal directions described in "Dimension Reduction Forests: Local Variable Importance using Structured Random Forests".

BibTeX reference to cite, if you use this package:

```bibtex
@article{loyal2021drforest,
    title = {Dimension Reduction Forests: Local Variable Importance using Structured Random Forests},
    author = {Joshua Daniel Loyal, Ruoqing Zhu, Yifan Cui, Xin Zhang},
    year = {2021},
    journal = {arXiv preprint arXiv:2103.13233}
}
```


Background
----------

Dimension reduction forests (DRFs) are a method for nonparametric regression
that also quantifies local variable importance by using methods from
sufficient dimension reduction (SDR). In particular, we take the perspective
of random forests as adaptive kernel methods. We pair random forests with
sufficient dimension reduction to estimate a nonparametric kernel that adapts
to the regression functions local contours. We then leverage this adaptivity
to estimate a type of local variable importance we call the
*local principal direction*. The result is a powerful model-free
predictive method that is more accurate than naively combining random forests
with global SDR methods.

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

Example Usage
-------------
As an example, we consider the following regression function
<p align="center">
<img src="/images/simulation1.png" alt="Simulation 1" width="600">
</p>
where we generate five covariates uniformly on [-3, 3]. In this scenario there are two informative covariates and three uninformative covariates. Projected onto the first two coordinates, the regression surface looks like

<p align="center">
<img src="/images/contours.png" alt="Simulation 1" width="400">
</p>

To fit a dimension reduction forest, we simulate the data,  initialize the estimator, and call fit:
```python
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from drforest.datasets import make_simulation1
from drforest.ensemble import DimensionReductionForestRegressor

# simulate regression problem
X, y = make_simulation1(
    n_samples=2000, noise=1, n_features=5, random_state=123)

# 80% - 20% train-test split
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=42)

# fit the dimension reduction forest
drforest = DimensionReductionForestRegressor(
    n_estimators=500, min_samples_leaf=3, n_jobs=-1).fit(X_train, y_train)
```

To make predictions on a new data set, we simply call `predict`. Note that the
dimension reduction forest also calculates the out-of-bag (OOB)
mean squared error (MSE), which is useful when an external test set is
not available:
```python
# predict on the test set
y_pred = drforest.predict(X_test)

# MSE on the test set
print(f'Test MSE: {mean_squared_error(y_test, y_pred):.2f}')
>>> Test MSE: 3.79

# compare with the out-of-bag (OOB) estimate
print(f'OOB MSE: {drforest.oob_mse_:.2f}')
>>> OOB MSE: 4.13
```

An important capability of dimension reduction forests is there ability to
produce a meaningful local variable importance measure known as the
*local principal direction*.  This is a feature importance assigned
to each prediction point. We can visualize the distribution of these importances
on the test set as follows:

```python
from drforest.plots import plot_local_direction

# extract the local principal direction at each point in the test set
directions = drforest.local_principal_direction(X_test, n_jobs=-1)

# plot the marginal distributions of each LPD's loading
plot_local_direction(directions)
```

<p align="center">
<img src="/images/lsvi_example.png" alt="local principal dircections" width="600">
</p>

From this plot, we clearly see that Feature 1 and Feature 2 are significant, while
Features 3, Feature 4, and Feature 5 do not play a role in predicting the outcome.
Finally, we can see how the feature importance varies throughout the input space.

```python
fig, ax = plt.subplots(figsize=(16, 18), ncols=2, sharey=True)

# calculate local principal direction at x = (-1.5, 1.5, 0, 0, 0)
lpd_x = drforest.local_principal_direction(np.array([-1.5, 1.5, 0, 0, 0]))
lpd_x *= np.sign(lpd_x[0])  # force loading of first component positive

# plot local principal direction
plot_single_direction(lpd_x, ax=ax[0], rotation=30)
ax[0].set_title(r'$x = (-1.5, 1.5, 0, 0, 0)$', fontsize=16)

# calculate local principal direction at x = (0.5, -0.5, 0, 0, 0)
lpd_x = drforest.local_principal_direction(np.array([0.5, -0.5, 0, 0, 0]))
lpd_x *= np.sign(lpd_x[0])  # force loading of first component postitive

# plot importance
plot_single_direction(imp_x, ax=ax[1], rotation=30)
ax[1].set_title(r'$x = (0.5, -0.5, 0, 0, 0)$', fontsize=16)
```

<p align="center">
<img src="/images/lsvi_local.png" alt="local principal directions" width="600">
</p>

From these plots, we conclude that at x = (-1.5, 1.5, 0, 0, 0), simultaneously
increasing or decreasing both Feature 1 and Feature 2 influences the regression
function. On the other hand, at x = (0.5, -0.5, 0, 0, 0), simultaneously
increasing Feature 1 while decreasing Feature 2, or vice versa, is influential.

Simulation Studies and Real-Data Applications
---------------------------------------------
This package includes the simulation studies and real-data applications found in
the main article:

* A simple example of local principal directions on a synthetic dataset: ([here](/examples/random_forest_importances.py)).
* A comparison with other methods on synthetic data: ([here](/examples/synthetic_data.py)).
* A evaluation of the performance of local principal direction estimates on synthetic data: ([here](/examples/local_subspace_importances.py)).
* A comparison with other methods on real regression problems: ([here](/examples/real_data.py)).
* An appliction using LPDs to infer meteorological factors contributing to  pollution in Beijing, China ([here](/examples/beijing_air_quality.py)).

We also provide a few [jupyter notebooks](/notebooks) that demonstrate the use of this package.
