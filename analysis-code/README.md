# README

This folder contains the analysis code for "Dimension Reduction Forests:
Local Variable Importance using Structured Random Forests". The `drforest/`
directory contains the Python package that implements the DRF algorithm and all
data sets analyzed in the paper, and `analysis-code/` contains the scripts
used to run the analyses in the article.


## Installation

The `drforest` package requires

* Python version >= 3.10
* A C/C++ compiler with C++17 and OpenMP support.

To install the package, run the following commands in the top-level directory:

```bash
pip install -r requirements.txt
cd .. drforest
python setup.py develop
```

The remainder of this document outlines how to reproduce the results in the
article. We break-up the results by the sections in the main paper and
supplementary materials. We assume that your are running the code in the
`analysis-code` directory. In addition, the `analysis-code` directory contains
a Makefile that runs the various analyses in the article.


## Motivation

To reproduce Figure 1 and Figure 2 in Section 2, run the following

```bash
python Figure1.py
python Figure2.py
```

These scripts will produce `Figure 1.png` and `Figure 2.png`, respectively.


## Section 5.1; Predictive Performance

**NOTE:** This analysis was originally run in parallel on a computer cluster.

This section outlines how to reproduce the results in Table 1 of the main text.
To run this analysis, execute the following

```bash
make synthetic_simulations
```

This will produce csv files in the directory `analysis-code/synthetic_data_results/`
of the form `<simulation_name>.csv`, which contains the results for each simulation.


## Section 5.2: Local Principal Directions

**NOTE:** This analysis was originally run in parallel on a computer cluster.

This section outlines how reproduce the results Figure 3 and Figure S.4. To
reproduce the results, execute the following

```bash
make lpd_simulations
```

This will produce `Figure 3.png` and `Figure S.4.png`.


## Section 6.1: Predictive Performance on Real Data Sets

**NOTE:** This analysis was originally run in parallel on a computer cluster.

This section outlines how to reproduce Table 2. To reproduce the results,
execute the following

```bash
make real_data
```

This will produce csv files in the directory `analysis-code/real_data_results/`
of the form `<dataset_name>.csv`, which contains the results for each data set.

## Section 6.2: PM2.5 Concentration in Beijing, China

This section outlines how to reproduce Figure 4, Figure S.5, and Figure S.6.
Run the following command in the `analysis-code` directory

```bash
python beijing_air_quality.py
```

This will produce png files in the directory `analysis-code/beijing_results/`.
The files are `Figure 4.png`, `Figure S.5.png`, and `Figure S.6.png`.


## Section S.4.2: Empirical Run Times

This section outlines how to reproduce Figure S.1. Run the following command
in the `analysis-code` directory

```bash
python compute_benchmark.py
```

This will produce `Figure S.1.png`.

## Section S.8: Sensitivity to the Number of Trees

**NOTE:** This analysis was originally run in parallel on a computer cluster.

This section outlines how to reproduce Figure S.2 and Figure S.3.
To run this analysis, execute the following

```bash
make synthetic_simulations
```

This will produce `Figure S.2.png` and `Figure S.3.png`, respectively.


## Section S.9: Sensitivity to the Number of Slices

**NOTE:** This analysis was originally run in parallel on a computer cluster.

This section outlines how to reproduce the results in Table S.3.
To run this analysis, execute the following

```bash
make sensitivity
```

This will produce csv files in the directory `analysis-code/sensitivity_results/`
of the form `<simulation_name>.csv`, which contains the results for each simulation.
