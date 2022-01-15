import os

import numpy as np
import pandas as pd
import plac

from functools import partial

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from drforest.ensemble import DimensionReductionForestRegressor
from drforest.dimension_reduction import SlicedInverseRegression
from drforest.dimension_reduction import SlicedAverageVarianceEstimation
from drforest.kernel_regression import fit_kernel_smoother_silverman

from drforest.datasets import (
    load_abalone,
    load_athletes,
    load_bodyfat,
    load_cpu_small,
    load_fishcatch,
    load_kin8nm,
    load_athletes,
    load_diabetes,
    load_openml,
)

DATASETS = {
    'abalone': load_abalone,
    'athletes': load_athletes,
    'bodyfat': load_bodyfat,
    'cpu' : load_cpu_small,
    'diabetes': load_diabetes,
    'fishcatch': load_fishcatch,
    'kin8nm' : load_kin8nm,
    'autoprice': partial(load_openml, name='autoPrice'),
    'liver': partial(load_openml, name='liver-disorders'),
    'mu284': partial(load_openml, name='mu284'),
    'puma32H': partial(load_openml, name='puma32H'),
    'puma8NH': partial(load_openml, name='puma8NH'),
    'wisconsin': partial(load_openml, name='wisconsin'),
    'bank8FM': partial(load_openml, name='bank8FM'),
}


OUT_DIR = 'real_data_output'


@plac.pos('dataset_name', 'Data set name', choices = DATASETS.keys())
@plac.opt('n_resamples', 'Number of repeated rounds of cross-validation', type=int)
@plac.opt('n_splits', 'Number of folds', type=int, abbrev='k')
def benchmark(dataset_name, n_resamples=15, n_splits=10):
    X, y = DATASETS[dataset_name]()
    if dataset_name in ['athletes', 'diabetes']:
        categorical_cols = [0]
    else:
        categorical_cols = None

    n_samples, n_features = X.shape

    n_estimators = 500
    min_samples_leaf_params = [1, 5]
    max_feature_params = [2, 4, 6, 1/3., 'sqrt', None]

    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    print('Unique y: {}'.format(np.unique(y).shape[0]))
    for resample_id in range(n_resamples):
        cv = KFold(n_splits=n_splits, shuffle=True,
                   random_state=resample_id * 42)
        results = {
            'mean': np.zeros(n_splits),
            'kernel_reg': np.zeros(n_splits),
            'kernel_reg_sir': np.zeros(n_splits),
            'kernel_reg_save': np.zeros(n_splits)
        }
        for min_samples_leaf in min_samples_leaf_params:
            for max_features in max_feature_params:
                results['rf (l={},p={})'.format(min_samples_leaf, max_features)] = (
                    np.zeros(n_splits))
                results['drrf (l={},p={})'.format(min_samples_leaf, max_features)] = (
                    np.zeros(n_splits))
                results['sir_rf (l={},p={})'.format(min_samples_leaf, max_features)] = (
                    np.zeros(n_splits))
                results['save_rf (l={},p={})'.format(min_samples_leaf, max_features)] = (
                    np.zeros(n_splits))


        for k, (train, test) in enumerate(cv.split(X)):
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

            # share a common tuning cv-split
            val_cv = KFold(n_splits=5, shuffle=False)

            print('Train: {}'.format(X_train.shape))
            print('Test: {}'.format(X_test.shape))

            print("Mean Only")
            err = np.mean((y_test - np.mean(y_train))**2)
            results['mean'][k] = err
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
                    results['rf (l={},p={})'.format(min_samples_leaf, max_features)][k] = err
                    print(err)

            for min_samples_leaf in min_samples_leaf_params:
                for max_features in max_feature_params:
                    if isinstance(max_features, int) and X.shape[1] < max_features:
                        continue

                    print("DR RandomForest (l={},p={})".format(min_samples_leaf, max_features))
                    forest = DimensionReductionForestRegressor(
                        n_estimators=n_estimators,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        categorical_cols=categorical_cols,
                        random_state=123,
                        n_jobs=-1).fit(X_train, y_train)
                    y_pred = forest.predict(X_test)
                    err = np.mean((y_pred - y_test)**2)
                    results['drrf (l={},p={})'.format(min_samples_leaf, max_features)][k] = err
                    print(err)

            for min_samples_leaf in min_samples_leaf_params:
                for max_features in max_feature_params:
                    if isinstance(max_features, int) and X.shape[1] < max_features:
                        continue

                    print("SIR + RF (l={},p={})".format(min_samples_leaf, max_features))
                    forest = Pipeline([
                        ('sir', SlicedInverseRegression(n_directions=None)),
                        ('rf', RandomForestRegressor(n_estimators=n_estimators,
                                                     min_samples_leaf=min_samples_leaf,
                                                     max_features=max_features,
                                                     random_state=123,
                                                     n_jobs=-1))
                        ]).fit(X_train, y_train)
                    y_pred = forest.predict(X_test)
                    err = np.mean((y_pred - y_test)**2)
                    results['sir_rf (l={},p={})'.format(min_samples_leaf, max_features)][k] = err
                    print(err)

            for min_samples_leaf in min_samples_leaf_params:
                for max_features in max_feature_params:
                    if isinstance(max_features, int) and X.shape[1] < max_features:
                        continue

                    print("SAVE + RF (l={},p={})".format(min_samples_leaf, max_features))
                    forest = Pipeline([
                        ('save', SlicedAverageVarianceEstimation(n_directions=None)),
                        ('rf', RandomForestRegressor(n_estimators=n_estimators,
                                                     min_samples_leaf=min_samples_leaf,
                                                     max_features=max_features,
                                                     random_state=123,
                                                     n_jobs=-1))
                        ]).fit(X_train, y_train)
                    y_pred = forest.predict(X_test)
                    err = np.mean((y_pred - y_test)**2)
                    results['save_rf (l={},p={})'.format(min_samples_leaf, max_features)][k] = err
                    print(err)


            print("Kernel Regression")
            ksmooth = fit_kernel_smoother_silverman(
                X_train, y_train, feature_type='raw')
            y_pred = ksmooth.predict(X_test)
            err = np.mean((y_pred - y_test)**2)
            results['kernel_reg'][k] = err
            print(err)

            print("SIR Kernel Regression")
            ksmooth = fit_kernel_smoother_silverman(
                X_train, y_train, feature_type='sir')
            y_pred = ksmooth.predict(X_test)
            err = np.mean((y_pred - y_test)**2)
            results['kernel_reg_sir'][k] = err
            print(err)

            print("SAVE Kernel Regression")
            ksmooth = fit_kernel_smoother_silverman(
                X_train, y_train, feature_type='save')
            y_pred = ksmooth.predict(X_test)
            err = np.mean((y_pred - y_test)**2)
            results['kernel_reg_save'][k] = err
            print(err)

        results = pd.DataFrame(results)
        results['fold'] = np.arange(n_splits)

        output_name = os.path.join(OUT_DIR, "{}_{}.csv".format(
            dataset_name, resample_id))
        results.to_csv(output_name, index=False)


if __name__ == '__main__':
    plac.call(benchmark)
