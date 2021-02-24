import os

import numpy as np
import autograd.numpy as anp
import pandas as pd
import plac

from autograd import grad
from autograd import elementwise_grad as egrad
from drforest.ensemble import DimensionReductionForestRegressor
from drforest.dimension_reduction import (
    SlicedAverageVarianceEstimation, SlicedInverseRegression)
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances


n_samples = 2000
n_features = 5
n_points = 100
signal_to_noise = 3
n_iter = 50
OUT_DIR = 'lsvi_results'

if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)


def run_lsvi_sim(dataset_name):
    if dataset_name == 'sim1':
        def func(X):
            return anp.abs(X[:, 0]) + anp.abs(X[:, 1])
    elif dataset_name == 'sim2':
        def func(X):
            return X[:, 0] + X[:, 1] ** 2
    elif dataset_name == 'sim3':
        def func(X):
            scale = 0.25
            return (5 * anp.maximum(
                    anp.exp(-scale * X[:, 0] ** 2),
                    anp.exp(-scale * X[:, 1] ** 2)))
    elif dataset_name == 'sim4':
        def func(X):
            r1 = X[:, 0] - X[:, 1]
            r2 = X[:, 0] + X[:, 1]
            return (20 * anp.maximum(
                anp.maximum(anp.exp(-2 * r1 ** 2), anp.exp(-r2 ** 2)),
                2 * anp.exp(-0.5 * (X[:, 0] ** 2 + X[:, 1] ** 2))))
    else:
        raise ValueError('Unrecognized dataset')

    grad_func = egrad(func)

    def true_directions(X0):
        true_dir = grad_func(X0)
        return true_dir / np.linalg.norm(true_dir, axis=1, keepdims=True)


    drf_metrics = np.zeros((n_iter, 2))
    drf_max_metrics = np.zeros((n_iter, 2))
    save_metrics = np.zeros((n_iter, 2))
    sir_metrics = np.zeros((n_iter, 2))
    local_sir_metrics = np.zeros((n_iter, 2))
    for idx in range(n_iter):
        rng = np.random.RandomState(123 * idx)

        if dataset_name == 'linear':
            cov = np.zeros((n_features, n_features))
            cov[:4, :4] = 0.9
            cov[np.diag_indices_from(cov)] = 1.
            X = rng.multivariate_normal(
                    mean=np.zeros(n_features),
                    cov=cov,
                    size=n_samples)
        else:
            X = rng.uniform(
                -3, 3, n_samples * n_features).reshape(n_samples, n_features)
        dist = euclidean_distances(X, X)

        y = func(X)
        if dataset_name == 'linear':
            sigma = 0.5
        else:
            sigma = np.var(y) / signal_to_noise
        y += np.sqrt(sigma) * rng.randn(n_samples)

        forests = []
        for min_samples_leaf in [3, 10, 25, 50, 100]:
            forests.append(DimensionReductionForestRegressor(
                store_X_y=True, n_jobs=-1,
                min_samples_leaf=min_samples_leaf,
                random_state=42).fit(X, y))

        forest_max = []
        if n_features > 5:
            for min_samples_leaf in [3, 10, 25, 50, 100]:
                forest_max.append(DimensionReductionForestRegressor(
                    store_X_y=True, n_jobs=-1,
                    min_samples_leaf=min_samples_leaf, max_features=5,
                    random_state=42).fit(X, y))

        neighbors = NearestNeighbors(metric='euclidean').fit(X)

        save = SlicedAverageVarianceEstimation().fit(X, y)
        save_dir = save.directions_[0].reshape(-1, 1)

        sir = SlicedInverseRegression().fit(X, y)
        sir_dir = sir.directions_[0].reshape(-1, 1)

        # sample n_point indices and check directions
        indices = rng.choice(np.arange(n_samples), replace=False, size=n_points)
        pred_dirs = []
        for forest in forests:
            pred_dirs.append(forest.local_subspace_importance(
                X[indices], n_jobs=-1))

        pred_dirs_max = []
        if n_features > 5:
            for forest in forest_max:
                pred_dirs_max.append(
                    forest.local_subspace_importance(
                        X[indices], n_jobs=-1))
        else:
            pred_dirs_max = pred_dirs

        true_dir = true_directions(X[indices])

        frob_norm = np.zeros(n_points)
        frob_norm[:] = np.inf
        trcor = np.zeros(n_points)
        trcor[:] = -np.inf

        frob_norm_max = np.zeros(n_points)
        frob_norm_max[:] = np.inf
        trcor_max = np.zeros(n_points)
        trcor_max[:] = -np.inf

        frob_norm_save = np.zeros(n_points)
        trcor_save = np.zeros(n_points)

        frob_norm_sir = np.zeros(n_points)
        trcor_sir = np.zeros(n_points)

        frob_norm_local_sir = np.zeros(n_points)
        frob_norm_local_sir[:] = np.inf
        trcor_local_sir = np.zeros(n_points)
        trcor_local_sir[:] = -np.inf

        for i in range(n_points):
            true_direc = true_dir[i].reshape(-1, 1)
            B_true = np.dot(true_direc, true_direc.T)

            # DRForest LSE
            for pred_dir in pred_dirs:
                pred_direc = pred_dir[i].reshape(-1, 1)
                B_hat = np.dot(pred_direc, pred_direc.T)

                frobk = np.sqrt(np.sum((B_true - B_hat) ** 2))
                if frobk < frob_norm[i]:
                    frob_norm[i] = frobk

                trcork = np.trace(np.dot(B_true, B_hat))
                if trcork > trcor[i]:
                    trcor[i] = trcork

            # DRForest Max LSE
            for pred_dir_max in pred_dirs_max:
                pred_direc = pred_dir_max[i].reshape(-1, 1)
                B_hat = np.dot(pred_direc, pred_direc.T)

                frobk = np.sqrt(np.sum((B_true - B_hat) ** 2))
                if frobk < frob_norm_max[i]:
                    frob_norm_max[i] = frobk

                trcork = np.trace(np.dot(B_true, B_hat))
                if trcork > trcor_max[i]:
                    trcor_max[i] = trcork

            # global save
            B_hat = np.dot(save_dir, save_dir.T)
            frob_norm_save[i] = np.sqrt(np.sum((B_true - B_hat) ** 2))
            trcor_save[i] = np.trace(np.dot(B_true, B_hat))

            # global SIR
            B_hat = np.dot(sir_dir, sir_dir.T)
            frob_norm_sir[i] = np.sqrt(np.sum((B_true - B_hat) ** 2))
            trcor_sir[i] = np.trace(np.dot(B_true, B_hat))

            # local sir (k-nearest neighbors)
            for n_neighbors in [max(10, n_features), 25, 50, 100]:
                try:
                    index = neighbors.kneighbors(
                        X[indices[i]].reshape(1, -1), n_neighbors=n_neighbors,
                        return_distance=False).ravel()
                    local_sir_dir = SlicedInverseRegression(n_directions=1).fit(
                        X[index], y[index]).directions_[0]
                    local_sir_dir = local_sir_dir.reshape(-1, 1)

                    B_hat = np.dot(local_sir_dir, local_sir_dir.T)

                    frobk = np.sqrt(np.sum((B_true - B_hat) ** 2))
                    if frobk < frob_norm_local_sir[i]:
                        frob_norm_local_sir[i] = frobk

                    trcork = np.trace(np.dot(B_true, B_hat))
                    if trcork > trcor_local_sir[i]:
                        trcor_local_sir[i] = trcork
                except np.linalg.LinAlgError:
                    pass

        drf_metrics[idx, 0] = np.mean(frob_norm)
        drf_metrics[idx, 1] = np.mean(trcor)
        drf_max_metrics[idx, 0] = np.mean(frob_norm_max)
        drf_max_metrics[idx, 1] = np.mean(trcor_max)
        save_metrics[idx, 0] = np.mean(frob_norm_save)
        save_metrics[idx, 1] = np.mean(trcor_save)
        sir_metrics[idx, 0] = np.mean(frob_norm_sir)
        sir_metrics[idx, 1] = np.mean(trcor_sir)
        local_sir_metrics[idx, 0] = np.mean(frob_norm_local_sir)
        local_sir_metrics[idx, 1] = np.mean(trcor_local_sir)

    # write to file
    print('Frobenius Norm')
    print("DRForest {:.3f} +/- {:.3f}".format(
        np.mean(drf_metrics[:, 0]), np.std(drf_metrics[:, 0])))
    print("DRForest Max {:.3f} +/- {:.3f}".format(
        np.mean(drf_max_metrics[:, 0]), np.std(drf_max_metrics[:, 0])))
    print("Global SAVE {:.3f} +/- {:.3f}".format(
        np.mean(save_metrics[:, 0]), np.std(save_metrics[:, 0])))
    print("Global SIR {:.3f} +/- {:.3f}".format(
        np.mean(sir_metrics[:, 0]), np.std(sir_metrics[:, 0])))
    print("Local SIR {:.3f} +/- {:.3f}".format(
        np.mean(local_sir_metrics[:, 0]), np.std(local_sir_metrics[:, 0])))

    print('Trace Correlation')
    print("DRForest {:.3f} +/- {:.3f}".format(
        np.mean(drf_metrics[:, 1]), np.std(drf_metrics[:, 1])))
    print("DRForest Max {:.3f} +/- {:.3f}".format(
        np.mean(drf_max_metrics[:, 1]), np.std(drf_max_metrics[:, 1])))
    print("Global SAVE {:.3f} +/- {:.3f}".format(
        np.mean(save_metrics[:, 1]), np.std(save_metrics[:, 1])))
    print("Global SIR {:.3f} +/- {:.3f}".format(
        np.mean(sir_metrics[:, 1]), np.std(sir_metrics[:, 1])))
    print("Local SIR {:.3f} +/- {:.3f}".format(
        np.mean(local_sir_metrics[:, 1]), np.std(local_sir_metrics[:, 1])))

    data = pd.DataFrame({
            'DRF' : drf_metrics[:, 0],
            'DRF (max_features=5)' : drf_max_metrics[:, 0],
            'Global SAVE': save_metrics[:, 0],
            'Global SIR': sir_metrics[:, 0],
            'Local SIR': local_sir_metrics[:, 0]})
    data.to_csv(os.path.join(OUT_DIR, '{}_p{}_frob.csv'.format(
        dataset_name, n_features), index=False))

    data = pd.DataFrame({
            'DRF' : drf_metrics[:, 1],
            'DRF (max_features=5)' : drf_max_metrics[:, 1],
            'Global SAVE': save_metrics[:, 1],
            'Global SIR': sir_metrics[:, 1],
            'Local SIR': local_sir_metrics[:, 1]})
    data.to_csv(os.path.join(OUT_DIR, '{}_p{}_trcor.csv'.format(
        dataset_name, n_features), index=False))


if __name__ == '__main__':
    plac.call(run_lsvi_sim)
