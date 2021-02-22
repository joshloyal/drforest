import numpy as np
import matplotlib.pyplot as plt

__all__ = ['plot_depth', 'plot_node']


def plot_node(X, y, tree, node_id, axes=None):
    path = tree.decision_path(X)
    samples_node = np.where(path[:, node_id].ravel() == 1)[0]
    X = X[samples_node, :]
    feature = tree.tree_.feature[node_id]
    if feature >= 0:
        direction = tree.tree_.directions[node_id, :]
        X_sdr = np.dot(X, direction)
        if axes:
            axes.scatter(X_sdr, y[samples_node], edgecolor='k')
            axes.axvline(tree.tree_.threshold[node_id], linestyle='--', color='k')
            return axes
        return plt.scatter(X_sdr, y[samples_node], edgecolor='k', alpha=0.75)


def plot_depth(X, y, tree, depth, show_root=False):
    path = tree.decision_path(X)
    node_ids = np.where(tree.tree_.node_depth == depth)[0]

    n_nodes = len(node_ids)
    if n_nodes == 1:
        return plot_node(X, y, tree, node_ids[0])

    if show_root:
        n_plots = n_nodes + 1
        n_cols = 3
        n_rows = int(np.ceil(n_plots / n_cols))
        f, axes = plt.subplots(n_rows, n_cols, sharey=True, figsize=(20, 10))

        if n_rows > 1:
            axes = [item for sublist in axes for item in sublist]
        plot_node(X, y, tree, 0, axes=axes[0])
        for i, node_id in enumerate(node_ids):
            samples_node = np.where(path[:, node_id].ravel() == 1)[0]
            X_node = X[samples_node, :]
            feature = tree.tree_.feature[node_id]
            if feature >= 0:
                direction = tree.tree_.directions[feature, :]
                X_sdr = np.dot(X_node, direction)
                axes[i+1].scatter(X_sdr, y[samples_node], edgecolor='k')
                axes[i+1].axvline(tree.tree_.threshold[node_id], linestyle='--', color='k')
    else:
        f, axes = plt.subplots(1, n_nodes, sharey=True, figsize=(20, 10))
        for i, node_id in enumerate(node_ids):
            samples_node = np.where(path[:, node_id].ravel() == 1)[0]
            X_node = X[samples_node, :]
            feature = tree.tree_.feature[node_id]
            if feature >= 0:
                direction = tree.tree_.directions[node_id, :]
                X_sdr = np.dot(X_node, direction)
                axes[i].scatter(X_sdr, y[samples_node], edgecolor='k')

    return f

