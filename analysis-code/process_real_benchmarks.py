import glob
import os

import pandas as pd


DIR_NAME = 'real_data_output'
OUT_DIR = 'real_data_results'


if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)


EST_NAMES = {
    'drrf': 'DRF',
    'sir_rf': 'SIR + RF',
    'save_rf': 'SAVE + RF',
    'kernel_reg_sir': 'SIR + NW Kernel',
    'kernel_reg_save': 'SAVE + NW Kernel',
    'kernel_reg': 'NW Kernel'
}


def get_datasets(dir_name):
    return list({f.split('/')[1].split('_')[0] for f in glob.glob(dir_name + '/*')})


def process_results(dataset_name='wisconsin'):
    files = glob.glob(os.path.join(DIR_NAME, dataset_name + '*'))

    # calculate averages over all replications
    data = None
    for file_name in files:
        rmse = pd.read_csv(file_name).mean(axis=0)
        est_names = {name.split(' ')[0] for name in rmse.index}
        datum = dict()
        for est_name in est_names:
            est_rmse = rmse.iloc[rmse.index.str.startswith(est_name)]
            # filter out zeros
            datum[est_name] = [est_rmse[est_rmse > 0].min()]
        if data is None:
            data = pd.DataFrame(datum)
        else:
            data = pd.concat((data, pd.DataFrame(datum)))

    # remove irrelavent columns
    data.pop('fold')

    ranks = data.values.argsort(axis=1).argsort(axis=1) + 1
    rmse = data.mean(axis=0)
    rmse_std = data.std(axis=0)
    rel_mse = 1. - data.div(data['rf'], axis=0)
    rel_mean = rel_mse.mean(axis=0) * 100
    rel_std = rel_mse.std(axis=0) * 100
    results = pd.DataFrame(
        {'rel_mse': rel_mean.values, 'rel_mse_std': rel_std.values},
        index=rmse.index)

    # rename values
    results = results.filter(items=EST_NAMES.keys(), axis=0)
    results.index = results.index.map(EST_NAMES)
    results.index.name = 'Estimator'
    results.columns = ['Relative MSE', 'Relative MSE (STD)']

    return results

for dataset in get_datasets(DIR_NAME):
    data = process_results(dataset)
    data.to_csv(os.path.join(OUT_DIR, dataset + '.csv'))
