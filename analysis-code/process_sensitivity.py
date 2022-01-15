import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sns
import os


def get_n_slices(col_name):
    return int(col_name.split()[1].split(',')[-1][2:].split(')')[0])


DIR_NAME = 'sensitivity_output'
OUT_DIR = 'sensitivity_results'


if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)


EST_NAMES = {
    'drrf_s5': 'DRF (n_slices=5)',
    'drrf_s10': 'DRF (n_slices=10)',
    'drrf_s15': 'DRF (n_slices=15)',
    'drrf_s20': 'DRF (n_slices=20)',
}


for i, file_name in enumerate(glob.glob(os.path.join(DIR_NAME, '*csv'))):
    mse = pd.read_csv(file_name)
    mse.pop('resample_id')
    est_names = {name.split(' ')[0] for name in mse.columns}
    results = {}
    for est_name in est_names:
        est = list(mse.columns[mse.columns.str.startswith(est_name)])
        est_mse = mse[est]
        if est_name == 'drrf':
            for s in [5, 10, 15, 20]:
                est = list(est_mse.columns.str.endswith("s={0})".format(s)))
                results[est_name + "_s{0}".format(s)] = (
                    est_mse.iloc[:, est].apply(lambda x: x[x > 0].min(), axis=1))
        else:
            results[est_name] = est_mse.apply(lambda x: x[x > 0].min(), axis=1)
    mse = pd.DataFrame(results)
    results = pd.DataFrame({
        'mse': mse.mean(axis=0),
        'mse_std': mse.std(axis=0)},
        index=mse.mean(axis=0).index)

    results = results.filter(items=EST_NAMES.keys(), axis=0)
    results.index = results.index.map(EST_NAMES)
    results.index.name = 'Estimator'
    results.columns = ['MSE', 'MSE (STD)']

    dataset_name = file_name.split('/')[1].split('_')[0]
    results.to_csv(os.path.join(os.path.join(OUT_DIR, dataset_name + '.csv')))
