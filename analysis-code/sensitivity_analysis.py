import os

import numpy as np
import pandas as pd
import plac

from functools import partial

from sklearn.datasets import (
    make_friedman1,
    make_friedman2,
    make_friedman3
)

from sklearn.model_selection import KFold, ShuffleSplit, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from drforest.ensemble import DimensionReductionForestRegressor
from drforest.dimension_reduction import SlicedInverseRegression
from drforest.dimension_reduction import SlicedAverageVarianceEstimation
from drforest.kernel_regression import fit_kernel_smoother_silverman

from drforest.datasets import (
    make_simulation1, make_simulation2, make_simulation3, make_simulation4,
    make_simulation5, make_simulation6, make_simulation7)


DATASETS = {
    'friedman1': partial(make_friedman1, noise=1),
    'friedman2': partial(make_friedman2, noise=1),
    'friedman3': partial(make_friedman3, noise=1),
    'simulation1': partial(
        make_simulation1, n_features=5, correlate_features=False, noise=1),
    'simulation2': partial(make_simulation2, n_features=4, noise=1),
    'simulation3': make_simulation3,
    'simulation4': make_simulation4,
    'simulation5': make_simulation5,
    'simulation6': make_simulation6,
    'simulation7': make_simulation7
}


OUT_DIR = 'sensitivity_output'


@plac.pos('dataset', 'Data set name', choices = DATASETS.keys())
@plac.opt('n_resamples', 'Number of repeated rounds of cross-validation', type=int)
def sensitivity_synthetic(dataset, n_resamples=50, n_samples_train=2000,
                          n_samples_test=1000):

    n_samples = n_samples_train + n_samples_test

    n_estimators = 500
    min_samples_leaf_params = [1, 5]
    max_feature_params = [2, 4, 6, 1/3., 'sqrt', None]
    slices_params = [5, 10, 15, 20]

    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    results = {
        'mean': np.zeros(n_resamples),
    }

    for min_samples_leaf in min_samples_leaf_params:
        for max_features in max_feature_params:
            results['rf (l={},p={})'.format(min_samples_leaf, max_features)] = (
                np.zeros(n_resamples))
            for n_slices in slices_params:
                results['drrf (l={},p={},s={})'.format(min_samples_leaf, max_features, n_slices)] = (
                    np.zeros(n_resamples))

    for resample_id in range(n_resamples):
        X, y = DATASETS[dataset](n_samples=n_samples,
                                 random_state=resample_id * 42)

        if dataset in ['simulation6', 'simulation7']:
            categorical_cols = [0]
        else:
            categorical_cols = None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=n_samples_test, random_state=resample_id * 123)

        print('Train: {}'.format(X_train.shape))
        print('Test: {}'.format(X_test.shape))

        print("Mean Only")
        err = np.mean((y_test - np.mean(y_train))**2)
        results['mean'][resample_id] = err
        print(err)

        for min_samples_leaf in min_samples_leaf_params:
            for max_features in max_feature_params:
                if isinstance(max_features, int) and X.shape[1] < max_features:
                    continue
                print("RandomForest (l={},p={})".format(min_samples_leaf, max_features))
                forest = RandomForestRegressor(n_estimators=n_estimators,
                                               min_samples_leaf=min_samples_leaf,
                                               max_features=max_features,
                                               random_state=123,
                                               n_jobs=-1).fit(X_train, y_train)
                y_pred = forest.predict(X_test)
                err = np.mean((y_pred - y_test)**2)
                results['rf (l={},p={})'.format(min_samples_leaf, max_features)][resample_id] = err
                print(err)

        for n_slices in slices_params:
            for min_samples_leaf in min_samples_leaf_params:
                for max_features in max_feature_params:
                    if isinstance(max_features, int) and X.shape[1] < max_features:
                        continue

                    print("DR RandomForest (l={},p={},s={})".format(min_samples_leaf, max_features, n_slices))
                    forest = DimensionReductionForestRegressor(
                        n_estimators=n_estimators,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        n_slices=n_slices,
                        categorical_cols=categorical_cols,
                        random_state=123,
                        n_jobs=-1).fit(X_train, y_train)
                    y_pred = forest.predict(X_test)
                    err = np.mean((y_pred - y_test)**2)
                    results['drrf (l={},p={},s={})'.format(min_samples_leaf, max_features, n_slices)][resample_id] = err
                    print(err)


    results_df = pd.DataFrame(results)
    results_df['resample_id'] = np.arange(n_resamples)

    output_name = os.path.join(OUT_DIR, "{0}_{1}n_{2}t_{3}r.csv".format(
        dataset, n_samples_train, n_samples_test, n_resamples))
    results_df.to_csv(output_name, index=False)


if __name__ == '__main__':
    plac.call(sensitivity_synthetic)
