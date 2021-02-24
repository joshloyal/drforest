import numpy as np

from sklearn.ensemble._forest import (
    _get_n_samples_bootstrap, _generate_unsampled_indices)
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state


__all__ = ['permutation_importance']


def get_oob_indices(tree, n_samples):
    n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, n_samples)
    return _generate_unsampled_indices(
        tree.random_state, n_samples, n_samples_bootstrap)


def oob_mean_squared_error(forest, X_train, y_train):
    """Difference in OOB MSE estimates averaged over the entire forest."""
    n_samples = X_train.shape[0]
    oob_mse = np.zeros(forest.n_estimators)
    for t, tree in enumerate(forest.estimators_):
        if hasattr(tree, 'generate_oob_indices'):
            oob = tree.generate_oob_indices()
        else:
            oob = get_oob_indices(tree, n_samples)

        oob_mse[t] = mean_squared_error(
            y_train[oob], tree.predict(X_train[oob, :]))

    return oob_mse


def calculate_permutation_mse(forest, X, y, col_idx, random_state=None):
    random_state = check_random_state(random_state)

    X_permuted = X.copy()
    shuffling_idx = np.arange(X.shape[0])
    random_state.shuffle(shuffling_idx)
    X_permuted[:, col_idx] = X_permuted[shuffling_idx, col_idx]

    return oob_mean_squared_error(forest, X_permuted, y)


def permutation_importance(forest, X, y, scale=False, random_state=None):
    random_state = check_random_state(random_state)

    baseline_mse = oob_mean_squared_error(forest, X, y)

    importances = np.zeros(X.shape[1])
    importances_std = np.zeros(X.shape[1])
    for col_idx in range(X.shape[1]):
        perm_mse = calculate_permutation_mse(forest, X, y, col_idx, random_state)
        tree_importances = np.mean(perm_mse - baseline_mse)
        importances[col_idx] = np.mean(tree_importances)
        importances_std[col_idx] = np.std(tree_importances)

    if scale:
        importances_std[importances_std == 0.] = 1.
        importances /= importances_std

    return importances

