[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/joshloyal/drforest/blob/master/LICENSE)
<!--[![Travis](https://travis-ci.com/joshloyal/dynetlsm.svg?token=gTKqq3zSsip89mhYVQPZ&branch=master)](https://travis-ci.com/joshloyal/dynetlsm)
[![AppVeyor](https://ci.appveyor.com/api/projects/status/github/joshloyal/dynetlsm)](https://ci.appveyor.com/project/joshloyal/dynetlsm/history)
[![PyPI Latest Release](https://img.shields.io/pypi/v/dynetlsm)](https://pypi.org/project/dynetlsm/)-->

# Dimension Reduction Forests: Local Variable Importances using Structured Random Forests

*Author: [Joshua D. Loyal](https://joshloyal.github.io/)*

This package provides an interface for learning and inference in latent
space models for dynamic networks. Inference is performed using
blocked Metropolis-Hastings within Gibbs sampling.

The primary method implemented in this package is the hierarchical Dirichlet
process latent position cluster model (HDP-LPCM) described in
"A Bayesian nonparametric latent space approach to modeling evolving communities in
dynamic networks" [[arXiv:2003.07404](https://arxiv.org/abs/2003.07404)].

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
git clone https://github.com/joshloyal/dynetlsm.git
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
This package includes the simulation studies and real-data applications found in Loyal and Chen (2020)<sup>[[6]](#References)</sup>:

* A synthetic dynamic network with a time-homogeneous community structure: ([here](/examples/homogeneous_simulation.py)).
* A synthetic dynamic network with a time-inhomogeneous community structure: ([here](/examples/inhomogeneous_simulation.py)).
* Sampson's monastery network: ([here](/examples/sampson_monks.py)).
* A dynamic network constructed from international military alliances during the first three decades of the Cold War (1950 - 1979): ([here](/examples/military_alliances.py)).
* A dynamic network constructed from character interactions in the first four seasons of the Game of Thrones television series: ([here](/examples/GoT.py)).

We also provide a few [jupyter notebooks](/notebooks) that demonstrate the use of this package.

References
----------

[1]:
