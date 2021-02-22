import warnings

import numpy as np

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.pipeline import Pipeline

from drforest.dimension_reduction import SlicedInverseRegression
from drforest.dimension_reduction import SlicedAverageVarianceEstimation
from drforest.ensemble import DimensionReductionForestRegressor
from drforest.kernel_regression import KernelRegression
from drforest.kernel_regression import DimensionReductionForestKernel


__all__ = ['tune_random_forest', 'tune_dimension_reduction_forest',
           'tune_screened_dimension_reduction_forest',
           'tune_sir_forest', 'tune_save_forest', 'tune_kernel_smoother',
           'tune_kernel_smoother_silverman',
           'tune_kernel_ridge', 'tune_forest_kernel_ridge']


def tune_random_forest(X, y,
                       n_estimators=500, tune_n_estimators=250,
                       cv=5, random_state=123):
    forest = RandomForestRegressor(n_estimators=tune_n_estimators,
                                   random_state=random_state,
                                   n_jobs=-1)
    parameters = {
        'min_samples_leaf': [1, 5, 10],
        #'min_samples_leaf': [1, int((X.shape[0] ** (1./3.)) / 2.)],
        'max_features': [1./3., 'log2', 'sqrt', None],
    }

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        forest_cv = GridSearchCV(forest, parameters,
                                 cv=cv, n_jobs=-1,
                                 iid=False, refit=False,
                                 scoring='neg_mean_squared_error')
        forest_cv.fit(X, y)

    forest.set_params(**forest_cv.best_params_)
    forest.set_params(n_estimators=n_estimators)
    forest.fit(X, y)

    return forest

def tune_dimension_reduction_forest(X, y,
                                    n_estimators=500, tune_n_estimators=250,
                                    cv=5, n_draws=25,
                                    power_transform=False,
                                    max_features=None,
                                    random_state=123):

    if power_transform:
        forest = Pipeline([
            ('standard', StandardScaler()),
            ('power', PowerTransformer()),
            ('rf', DimensionReductionForestRegressor(n_estimators=tune_n_estimators,
                                                     random_state=random_state,
                                                     store_X_y=True,
                                                     compute_directions=True,
                                                     n_jobs=-1))
        ])

        parameters = {
            'rf__min_samples_leaf': [1, 5, 10],
        }

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            forest_cv = GridSearchCV(forest,
                                     parameters,
                                     cv=cv, n_jobs=-1,
                                     iid=False, refit=False,
                                     scoring='neg_mean_squared_error')
            forest_cv.fit(X, y)

        forest.set_params(**forest_cv.best_params_)
        forest.set_params(rf__n_estimators=n_estimators)
        forest.fit(X, y)

        return forest

    parameters = {
        'min_samples_leaf': [1, 5, 10],
    }

    forest = DimensionReductionForestRegressor(n_estimators=tune_n_estimators,
                                               random_state=random_state,
                                               max_features=max_features,
                                               store_X_y=True,
                                               n_jobs=-1)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        forest_cv = GridSearchCV(forest,
                                 parameters,
                                 cv=cv, n_jobs=-1,
                                 iid=False, refit=False,
                                 scoring='neg_mean_squared_error')
        forest_cv.fit(X, y)

    forest.set_params(**forest_cv.best_params_)
    forest.set_params(n_estimators=n_estimators)
    forest.fit(X, y)

    return forest

def tune_screened_dimension_reduction_forest(X, y,
                                             n_estimators=500,
                                             max_features=None,
                                             tune_n_estimators=250,
                                             cv=5, n_draws=25,
                                             power_transform=False,
                                             random_state=123):

    if max_features is None:
        n_features = X.shape[1]
        max_features = [3, 1/3., 'sqrt']
        if power_transform:
            if n_features > 5:
                max_features = max_features + [5]
            parameters = {
                'rf__max_features': max_features,
                'rf__min_samples_leaf': [1, 5, 10],
            }
            forest = Pipeline([
                ('standard', StandardScaler()),
                ('power', PowerTransformer()),
                ('rf', DimensionReductionForestRegressor(n_estimators=tune_n_estimators,
                                                         random_state=random_state,
                                                         store_X_y=True,
                                                         compute_directions=True,
                                                         n_jobs=-1))
            ])
        else:
            if n_features > 5:
                max_features = max_features + [5]
            parameters = {
                'max_features': max_features,
                'min_samples_leaf': [1, 5, 10],
            }
            forest = DimensionReductionForestRegressor(n_estimators=tune_n_estimators,
                                                       random_state=random_state,
                                                       store_X_y=True,
                                                       compute_directions=True,
                                                       n_jobs=-1)
    else:
        if power_transform:
            parameters = {
                'rf__min_samples_leaf': [1, 5, 10],
            }
            forest = Pipeline([
                ('standard', StandardScaler()),
                ('power', PowerTransformer()),
                ('rf', DimensionReductionForestRegressor(n_estimators=tune_n_estimators,
                                                         random_state=random_state,
                                                         max_features=max_features,
                                                         store_X_y=True,
                                                         compute_directions=True,
                                                         n_jobs=-1))
            ])
        else:
            parameters = {
                'min_samples_leaf': [1, 5, 10],
            }
            forest = DimensionReductionForestRegressor(n_estimators=tune_n_estimators,
                                                       random_state=random_state,
                                                       max_features=max_features,
                                                       store_X_y=True,
                                                       compute_directions=True,
                                                       n_jobs=-1)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        forest_cv = GridSearchCV(forest,
                                 parameters,
                                 cv=cv, n_jobs=-1,
                                 iid=False, refit=False,
                                 scoring='neg_mean_squared_error')
        forest_cv.fit(X, y)

    forest.set_params(**forest_cv.best_params_)
    if power_transform:
        forest.set_params(**{'rf__n_estimators' : n_estimators})
    else:
        forest.set_params(n_estimators=n_estimators)
    forest.fit(X, y)

    return forest


def tune_sir_forest(X, y,
                    n_estimators=500, tune_n_estimators=250,
                    n_directions=None,
                    cv=5, random_state=123):
    sir_forest = Pipeline([
        ('sir', SlicedInverseRegression(n_directions=n_directions)),
        ('rf', RandomForestRegressor(n_estimators=tune_n_estimators,
                                     random_state=random_state,
                                     n_jobs=-1))
    ])

    parameters = {
        'rf__min_samples_leaf': [1, 5, 10],
        'rf__max_features': [1./3., 'log2', 'sqrt', None],
    }

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        forest_cv = GridSearchCV(sir_forest,
                                 parameters,
                                 cv=cv, n_jobs=-1,
                                 iid=False, refit=False,
                                 scoring='neg_mean_squared_error')
        forest_cv.fit(X, y)

    sir_forest.set_params(**forest_cv.best_params_)
    sir_forest.set_params(**{'rf__n_estimators' : n_estimators})
    sir_forest.fit(X, y)

    return sir_forest


def tune_save_forest(X, y,
                    n_estimators=500, tune_n_estimators=250,
                    n_directions=None,
                    cv=5, random_state=123):
    save_forest = Pipeline([
        ('save', SlicedAverageVarianceEstimation(n_directions=n_directions)),
        ('rf', RandomForestRegressor(n_estimators=tune_n_estimators,
                                     random_state=random_state,
                                     n_jobs=-1))
    ])

    parameters = {
        'rf__min_samples_leaf': [1, 5, 10],
        #'rf__min_samples_leaf': [1, int((X.shape[0] ** (1./3.)) / 2.)],
        'rf__max_features': [1./3., 'log2', 'sqrt', None],
    }

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        forest_cv = GridSearchCV(save_forest,
                                 parameters,
                                 cv=cv, n_jobs=-1,
                                 iid=False, refit=False,
                                 scoring='neg_mean_squared_error')
        forest_cv.fit(X, y)

    save_forest.set_params(**forest_cv.best_params_)
    save_forest.set_params(**{'rf__n_estimators' : n_estimators})
    save_forest.fit(X, y)

    return save_forest


def tune_kernel_smoother_silverman(X, y, feature_type='raw',
                                   n_gammas='silverman', kernel='rbf',
                                   random_state=123):

        if feature_type == 'sir':
            kernel_smoother = Pipeline([
                ('scaler', StandardScaler()),
                ('sir', SlicedInverseRegression(n_directions=None)),
                ('ksmooth', KernelRegression(kernel=kernel, gamma='silverman'))
            ])
        elif feature_type == 'save':
            kernel_smoother = Pipeline([
                ('scaler', StandardScaler()),
                ('save', SlicedAverageVarianceEstimation(n_directions=None)),
                ('ksmooth', KernelRegression(kernel=kernel, gamma='silverman'))
            ])
        else:
            kernel_smoother = Pipeline([
                ('scaler', StandardScaler()),
                ('ksmooth', KernelRegression(kernel=kernel, gamma='silverman'))
            ])


        kernel_smoother.fit(X, y)
        return kernel_smoother

def tune_kernel_smoother(X, y, save_features=False, save_params=None,
                         n_gammas='silverman', kernel='rbf', boundary_val=3,
                         random_state=123):
    # applies an internal LOOV-CV to select gamma
    if boundary_val <= 0:
        raise ValueError('`boundary_val` must be positive')

    n_gammas = (int(2 * boundary_val + 1) if n_gammas == 'auto' else
                n_gammas)

    if save_features:
        save_params = save_params if save_params is not None else {}
        kernel_smoother = Pipeline([
            ('scaler', StandardScaler()),
            ('save', SlicedAverageVarianceEstimation(**save_params)),
            ('ksmooth', KernelRegression(kernel=kernel,
                                         gamma=np.logspace(-boundary_val,
                                                            boundary_val, n_gammas)))
        ])
    elif n_gammas == 'silverman':
        kernel_smoother = Pipeline([
            ('scaler', StandardScaler()),
            ('ksmooth', KernelRegression(kernel=kernel, gamma='silverman'))
        ])
    else:
        kernel_smoother = Pipeline([
            ('scaler', StandardScaler()),
            ('ksmooth', KernelRegression(kernel=kernel,
                                         gamma=np.logspace(-boundary_val,
                                                            boundary_val, n_gammas)))
        ])

    kernel_smoother.fit(X, y)

    return kernel_smoother


def tune_kernel_ridge(X, y, cv=5, random_state=123):
    kernel_ridge = Pipeline([
        ('scaler', StandardScaler()),
        ('kr', KernelRidge(kernel='rbf'))
    ])

    param_grid = {
        'kr__alpha': [1e0, 0.1, 1e-2, 1e-3],
        'kr__gamma': np.logspace(-2, 2, 5)
    }

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        kernel_ridge_cv = GridSearchCV(kernel_ridge,
                                       param_grid,
                                       cv=cv,
                                       scoring='neg_mean_squared_error',
                                       iid=False,
                                       n_jobs=-1)
        kernel_ridge_cv.fit(X, y)

    return kernel_ridge_cv.best_estimator_


def tune_forest_kernel_ridge(X, y, forest, cv=5, n_jobs=-1, random_state=123):
    kernel = DimensionReductionForestKernel()
    kernel.set_params(**forest.get_params())
    kernel_ridge = Pipeline([
        ('forest', kernel),
        ('kr', KernelRidge(kernel='precomputed'))
    ])

    param_grid = {
        'kr__alpha': [1e0, 0.1, 1e-2, 1e-3],
    }

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        kernel_ridge_cv = GridSearchCV(kernel_ridge,
                                       param_grid,
                                       cv=cv,
                                       scoring='neg_mean_squared_error',
                                       iid=False,
                                       n_jobs=n_jobs)
        kernel_ridge_cv.fit(X, y)

    return kernel_ridge_cv.best_estimator_
