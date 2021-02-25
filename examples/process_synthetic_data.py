import glob
import os

import pandas as pd

DIR_NAME = 'synthetic_data_results'
OUT_DIR = 'synthetic_data_processed'


if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)


def get_datasets(dir_name):
    return list({f.split('/')[1].split('_')[0] for f in glob.glob(dir_name + '/*')})


def process_results(dir_name='results', dataset_name='wisconsin'):
    file_name = glob.glob(os.path.join(dir_name, dataset_name + '*'))[0]
    mse = pd.read_csv(file_name)
    mse.pop('resample_id')
    est_names = {name.split(' ')[0] for name in mse.columns}
    results = {}
    for est_name in est_names:
        est_mse = mse[list(mse.columns[mse.columns.str.startswith(est_name)])]
        results[est_name] = est_mse.apply(lambda x: x[x > 0].min(), axis=1)

    data = pd.DataFrame(results)
    ranks = data.values.argsort(axis=1).argsort(axis=1) + 1
    rmse = data.mean(axis=0)
    rmse_std = data.std(axis=0)
    rel_mse = 1 - data.div(data['rf'], axis=0)
    rel_mean = rel_mse.mean(axis=0) * 100
    rel_std = rel_mse.std(axis=0) * 100
    results = pd.DataFrame(
        {'rank' : ranks.mean(axis=0), 'rank_std': ranks.std(axis=0),
         'mse': rmse.values, 'mse_std': rmse_std.values,
         'rel_mse': rel_mean.values, 'rel_mse_std': rel_std.values},
        index=rmse.index)

    return results

for dataset in get_datasets(DIR_NAME):
    data = process_results(DIR_NAME, dataset)
    data.to_csv(os.path.join(OUT_DIR, dataset + '.csv'))
