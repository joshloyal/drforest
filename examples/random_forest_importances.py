import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from drforest.datasets import make_simulation1
from drforest.ensemble import DimensionReductionForestRegressor
from drforest.ensemble import permutation_importance
from drforest.plots import plot_local_importance


plt.rc('font', family='serif')
fontsize = 14

n_samples = 2000
n_features = 5

X, y = make_simulation1(
    n_samples=n_samples, noise=1, n_features=n_features, random_state=1234)

forest = DimensionReductionForestRegressor(
    n_estimators=500, store_X_y=True, n_jobs=-1,
    min_samples_leaf=3, max_features=None,
    random_state=42).fit(X, y)

x0 = np.zeros(n_features)
x0[:2] = np.array([-1.5, 1.5])
local_direc_x0 = forest.local_subspace_importance(x0)
local_direc_x0 *= np.sign(local_direc_x0[0])

x1 = np.zeros(n_features)
x1[:2] = [0.5, -0.5]
local_direc_x1 = forest.local_subspace_importance(x1)
local_direc_x1 *= np.sign(local_direc_x1[0])

forest = RandomForestRegressor(n_estimators=500,
                               min_samples_leaf=3,
                               n_jobs=-1, max_features=None,
                               oob_score=True,
                               random_state=42).fit(X, y)

forest_imp = permutation_importance(
    forest, X, y, random_state=forest.random_state)
forest_imp /= np.sum(forest_imp)

fig, ax = plt.subplots(figsize=(18, 6), ncols=4)

def f(x, y):
    r1 = x - y
    r2 = x + y
    return (20 * np.maximum(
        np.maximum(np.exp(-2 * r1 ** 2), np.exp(-r2 ** 2)),
        2 * np.exp(-0.5 * (x ** 2 + y ** 2))))
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

ax[0].contour(X, Y, Z, 3, colors='black', linestyles='--', levels=5, linewidths=1.5)
ax[0].imshow(Z, extent=[-3, 3, -3, 3], origin='lower', cmap='YlGnBu_r', alpha=0.5)
ax[0].scatter([-1.5, 0.5], [1.5, -0.5], color=None, edgecolor='black')
ax[0].annotate(r'(-1.5, 1.5)', (-1.5, 1.5), xytext=(-1.4, 1.6), fontname='Sans', weight='bold')
ax[0].annotate(r'(0.5, -0.5)', (0.5, -0.5), xytext=(0.6, -0.4), fontname='Sans', weight='bold')
ax[0].set_aspect('equal')

ax[1].bar(np.arange(1, n_features + 1), forest_imp, color='gray')
ax[1].set_ylabel('Importance', fontsize=12)
ax[1].set_title('Random Forest', fontsize=fontsize)
ax[1].set_xlabel(None)
ax[1].axhline(0, color='black', linestyle='-')
ax[1].set_ylim(-1, 1)
ax[1].set_xlabel('Variable')
ax[1].text(3.5, 0.8, 'Global', fontsize=16)

color = ['tomato' if x > 0 else 'cornflowerblue' for x in local_direc_x0]
ax[2].bar(np.arange(1, n_features + 1), local_direc_x0, color=color)
ax[2].set_title('Dimension Reduction Forest', fontsize=fontsize)
ax[2].axhline(0, color='black', linestyle='-', lw=1)
ax[2].set_ylim(-1, 1)
ax[2].set_xlabel('Variable')
ax[2].text(2.5, 0.8, '$\mathbf{x}_0 = (-1.5, 1.5, 0, 0, 0)$', fontsize=10)

color = ['tomato' if x > 0 else 'cornflowerblue' for x in local_direc_x1]
ax[3].bar(np.arange(1, n_features + 1), local_direc_x1, color=color)
ax[3].set_title('Dimension Reduction Forest', fontsize=fontsize)
ax[3].set_xlabel('Variable')
ax[3].invert_yaxis()
ax[3].axhline(0, color='black', linestyle='-', lw=1)
ax[3].text(2.5, 0.8, '$\mathbf{x}_0 = (0.5, -0.5, 0, 0, 0)$', fontsize=10)
ax[3].set_ylim(-1, 1)

plt.subplots_adjust(wspace=0.3, left=0.03, right=0.985)
plt.show()
#fig.savefig('local_svi.pdf', dpi=300, bbox_inches='tight')
