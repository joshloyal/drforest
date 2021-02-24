[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/joshloyal/drforest/blob/master/LICENSE)
<!--[![Travis](https://travis-ci.com/joshloyal/dynetlsm.svg?token=gTKqq3zSsip89mhYVQPZ&branch=master)](https://travis-ci.com/joshloyal/dynetlsm)
[![AppVeyor](https://ci.appveyor.com/api/projects/status/github/joshloyal/dynetlsm)](https://ci.appveyor.com/project/joshloyal/dynetlsm/history)
[![PyPI Latest Release](https://img.shields.io/pypi/v/dynetlsm)](https://pypi.org/project/dynetlsm/)-->

# Dimension Reduction Forests

*Author: [Joshua D. Loyal](https://joshloyal.github.io/)*

This package provides an interface for dimension reduction forests described in
"Dimension Reduction Forests: Local Variable Importance using Structured Random Forests".

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

If you prefer, you can clone the repository and run the setup.py file. Use the following commands to get the copy from GitHub and install all the dependencies:

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

Simulation Studies and Real-Data Applications
---------------------------------------------
This package includes the simulation studies and real-data applications found in Loyal et al. (2021)<sup>[[6]](#References)</sup>:

* A simple example of local variable importance on a synthetic dataset: ([here](/examples/random_forest_importances.py)).
* A comparison with other methods on synthetic data: ([here](/examples/synthetic_data.py)).
* A evaluation of the performance of local subspace variable importance estimates on synthetic data: ([here](/examples/local_subspace_importances.py)).
* A comparison with other methods on real regression problems: ([here](/examples/real_data.py)).
* An application to understanding meteorological factors on pollution in Beijing, China ([here](/examples/beijing_air_quality.py)).

We also provide a few [jupyter notebooks](/notebooks) that demonstrate the use of this package.

References
----------

[1]:
