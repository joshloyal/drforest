import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from drforest.datasets import make_simulation1
from drforest.ensemble import DimensionReductionForestRegressor
from drforest.plots import plot_local_importance, plot_single_importance


plt.rc('font', family='serif')
fontsize = 14

n_samples = 2000
n_features = 5

X, y = make_simulation1(
    n_samples=n_samples, noise=1, n_features=n_features, random_state=1234)

# plot regression function
n_contours = 5
x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1, x2)
R1 = X1 - X2
R2 = X1 + X2
Z = 20 * np.maximum.reduce([np.exp(-2 * R1 ** 2),
                       np.exp(-1 * R2 ** 2),
                       2 * np.exp(-0.5 * (X1 ** 2 + X2 ** 2))])

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(
    Z, extent=[-3, 3, -3, 3], origin='lower', cmap='YlGnBu_r', alpha=0.5)
ax.contour(X1, X2, Z, levels=n_contours,
            linewidths=0.5, colors='k', linestyles='--')
ax.set_xlabel(r'$X_1$', fontsize=16)
ax.set_ylabel(r'$X_2$', fontsize=16)
fig.savefig('contours.png', dpi=100, bbox_inches='tight')

# 80% - 20% train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# fit the dimension reduction forest
drforest = DimensionReductionForestRegressor(
    n_estimators=500, min_samples_leaf=3, n_jobs=-1, random_state=42)
drforest.fit(X_train, y_train)

y_pred = drforest.predict(X_test)

print(f'Test MSE: {mean_squared_error(y_test, y_pred):.2f}')
print(f'OOB MSE: {drforest.oob_mse_:.2f}')


importances = drforest.local_subspace_importance(X_test, n_jobs=-1)

fig, ax = plot_local_importance(importances)

fig.savefig('lsvi_example.png', dpi=100, bbox_inches='tight')

# local importance
fig, ax = plt.subplots(figsize=(16, 8), ncols=2, sharey=True)

imp_x = drforest.local_subspace_importance(np.array([-1.5, 1.5, 0, 0, 0]))
imp_x *= np.sign(imp_x[0])

plot_single_importance(imp_x, ax=ax[0], rotation=30)
ax[0].set_title(r'$x = (-1.5, 1.5, 0, 0, 0)$', fontsize=16)

imp_x = drforest.local_subspace_importance(np.array([0.5, -0.5, 0, 0, 0]))
imp_x *= np.sign(imp_x[0])

plot_single_importance(imp_x, ax=ax[1], rotation=30)
ax[1].set_title(r'$x = (0.5, -0.5, 0, 0, 0)$', fontsize=16)

fig.savefig('lsvi_local.png', dpi=100, bbox_inches='tight')
