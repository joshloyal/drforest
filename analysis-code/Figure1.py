import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import pairwise_distances

from drforest.datasets import make_simulation1
from drforest.ensemble import DimensionReductionForestRegressor

plt.rc('font', family='serif')
fontsize = 17

def func(x):
    r1 = x[0] - x[1]
    r2 = x[0] + x[1]
    return (20 * np.maximum(
        np.maximum(np.exp(-2 * r1 ** 2), np.exp(-r2 ** 2)),
        2 * np.exp(-0.5 * (x[0] ** 2 + x[1] ** 2))))


n_contours = 5
n_samples = 2000

X, y = make_simulation1(
    n_samples=n_samples, noise=1, n_features=2, random_state=123)

drforest = DimensionReductionForestRegressor(
    n_estimators=500, min_samples_leaf=1, n_jobs=-1).fit(X, y)

forest = RandomForestRegressor(
    n_estimators=500, min_samples_leaf=1, n_jobs=-1).fit(X, y)

x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1, x2)
R1 = X1 - X2
R2 = X1 + X2
Z = 20 * np.maximum.reduce([np.exp(-2 * R1 ** 2),
                       np.exp(-1 * R2 ** 2),
                       2 * np.exp(-0.5 * (X1 ** 2 + X2 ** 2))])

fig, axes = plt.subplots(ncols=4, figsize=(18, 6))

for ax in axes.flat:
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

rf_kernel = 1 - pairwise_distances(
    forest.apply([[-1.5, 1.5]]), forest.apply(X), metric='hamming')
rf_kernel = rf_kernel.ravel() / rf_kernel.ravel().sum()

axes[0].imshow(
    Z, extent=[-3, 3, -3, 3], origin='lower', cmap='YlGnBu_r', alpha=0.5)
axes[0].contour(X1, X2, Z, levels=n_contours,
            linewidths=0.5, colors='k', linestyles='--')
axes[0].scatter(X[:, 0], X[:, 1],
            edgecolor='k',
            color='white',
            sizes=50 * np.sqrt(rf_kernel))
axes[0].scatter(-1.5, 1.5, color='tomato', edgecolor='black', marker='P', s=50)
axes[0].set_title("Random Forest", fontsize=fontsize)

rf_kernel = 1 - pairwise_distances(
    forest.apply([[0.5, -0.5]]), forest.apply(X), metric='hamming')
rf_kernel = rf_kernel.ravel() / rf_kernel.ravel().sum()
axes[2].imshow(
    Z, extent=[-3, 3, -3, 3], origin='lower', cmap='YlGnBu_r', alpha=0.5)
axes[2].contour(X1, X2, Z, levels=n_contours,
            linewidths=0.5, colors='k', linestyles='--')
axes[2].scatter(X[:, 0], X[:, 1],
            edgecolor='k',
            color='white',
            sizes=50 * np.sqrt(rf_kernel))
axes[2].scatter(
    0.5, -0.5, color='tomato', edgecolor='black', marker='P', s=50)
axes[2].set_title("Random Forest", fontsize=fontsize)


dr_kernel = drforest([-1.5, 1.5]).ravel()
dr_kernel /= dr_kernel.sum()
axes[1].imshow(
    Z, extent=[-3, 3, -3, 3], origin='lower', cmap='YlGnBu_r', alpha=0.5)
axes[1].contour(X1, X2, Z, levels=n_contours,
            linewidths=0.5, colors='k', linestyles='--')
axes[1].scatter(X[:, 0], X[:, 1],
            edgecolor='k',
            color='gray',
            sizes=50 * np.sqrt(dr_kernel))
axes[1].scatter(
    -1.5, 1.5, color='tomato', edgecolor='black', marker='P', s=50)
axes[1].set_title("Dimension Reduction Forest", fontsize=fontsize)

dr_kernel = drforest([0.5, -0.5]).ravel()
dr_kernel /= dr_kernel.sum()
axes[3].imshow(
    Z, extent=[-3, 3, -3, 3], origin='lower', cmap='YlGnBu_r', alpha=0.5)
axes[3].contour(X1, X2, Z, levels=n_contours,
            linewidths=0.5, colors='k', linestyles='--')
axes[3].scatter(X[:, 0], X[:, 1],
            edgecolor='k',
            color='gray',
            sizes=50 * np.sqrt(dr_kernel))
axes[3].scatter(0.5, -0.5, color='tomato', edgecolor='black', marker='P', s=50)
axes[3].set_title("Dimension Reduction Forest", fontsize=fontsize)

for ax in axes:
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.95, wspace=0.2)
fig.savefig('Figure 1.png', dpi=300, bbox_inches='tight')
