import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from drforest.datasets import load_beijing
from drforest.ensemble import DimensionReductionForestRegressor
from drforest.ensemble import permutation_importance
from drforest.plots import plot_local_importance


OUT_DIR = 'beijing_results'
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

# load data
data = load_beijing()

# plot pairwise scatter plots
df = data.copy()
cols = [
    'Temperature ($^{\circ}$C)',
    'Humidity (%)',
    'Pressure (hPa)',
    'Wind Speed (m/s)',
    'Month',
    'PM25_Concentration'
]
df.columns = cols
cols = cols[:-1]
g = sns.PairGrid(df, vars=cols, diag_sharey=False, corner=True, height=2)
g = sns.PairGrid(df, vars=cols, diag_sharey=False, corner=True, height=2)
g.map_diag(sns.histplot, color='0.3', bins='sturges', edgecolor='w')
g.map_lower(sns.scatterplot,
            hue=df['PM25_Concentration'], palette='YlGnBu', alpha=0.05)
g.add_legend(title=r'PM2.5 Concentration ($\mu g / m^3$)', loc='upper right',
             bbox_to_anchor=(0.7, 0.8))
g.savefig(os.path.join(OUT_DIR, 'beijing_pairplot.png'),
          dpi=300, bbox_inches='tight')


# standardize data and perform a 80%-20% train/test split
cols = ['Temperature', 'Humidity', 'Pressure', 'Wind Speed', 'Month']
X = data[cols].values
y = data['PM25_Concentration'].values

scaler =  StandardScaler().fit(X)
X_std = scaler.transform(X)

train, test = train_test_split(np.arange(X.shape[0]), test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = X_std[train], X_std[test], y[train], y[test]

# tune and train a dimension reduction forest (DRF)
errors = np.zeros(3)
drforests = []
min_sample_leaves = [3, 5, 10]
for i, min_sample_leaf in enumerate(min_sample_leaves):
    drforest = DimensionReductionForestRegressor(
        n_estimators=500,
        min_samples_leaf=min_sample_leaf,
        random_state=42,
        store_X_y=True,
        n_jobs=-1)
    drforest.fit(X_train, y_train)
    errors[i] = np.mean((y_test -  drforest.predict(X_test)) ** 2)
    drforests.append(drforest)

min_sample_leaf = min_sample_leaves[np.argmin(errors)]
print('min_sample_leaf = ', min_sample_leaf)

# re-train on full dataset
drforest = DimensionReductionForestRegressor(
    n_estimators=500,
    min_samples_leaf=min_sample_leaf,
    random_state=42,
    store_X_y=True,
    oob_mse=True,
    n_jobs=-1)
drforest.fit(X_std, y)
r_sq = 1 - np.mean((drforest.predict(X_std) - y) ** 2) / np.var(y)
print('R2 = ', r_sq)

# global permutation importance
fig, ax = plt.subplots(figsize=(12, 6))

forest_imp = drforest.feature_importances_
order = np.argsort(forest_imp)
ax.barh(y=np.arange(5), width=forest_imp[order], color='gray',
        tick_label=np.asarray(cols)[order], height=0.5)
ax.set_xlabel('Variable Importance')

fig.savefig(os.path.join(OUT_DIR, 'drf_airquality_imp.png'),
           dpi=300, bbox_inches='tight')

# extract local subspace variable importances
importances = drforest.local_subspace_importances(X_std, n_jobs=-1)
importances = np.sign(importances[:, 0]).reshape(-1, 1) * importances

# visualize marginal LSVI histograms
fig = plot_local_importance(
    importances, feature_names=np.asarray(cols), color='0.3')
fig.savefig(os.path.join(OUT_DIR, 'beijing_loadings.png'), dpi=300)

# visualize LSVI's as a function of month
fig, ax = plt.subplots(figsize=(10, 6))
month_col = 4
loading_medians = []
loading_up = []
loading_low = []
q_alpha = 0.25
for t in range(1, 13):
    val = (t - scaler.mean_[month_col]) /  scaler.scale_[month_col]
    mask = X_std[:, month_col] == val

    local_dir_month = importances[mask][:, :month_col]
    res = np.quantile(local_dir_month, q=[q_alpha, 0.5, 1 - q_alpha], axis=0)

    loading_low.append(res[0, :])
    loading_medians.append(res[1, :])
    loading_up.append(res[2, :])

loading_medians = np.asarray(loading_medians)
loading_up = np.asarray(loading_up)
loading_low = np.asarray(loading_low)

for p in range(4):
    ax.plot(loading_medians[:, p], 'o-', lw=2, label=cols[p], markersize=8,
            linestyle='--')
    ax.fill_between(np.arange(12),
                    loading_low[:, p], loading_up[:, p], alpha=0.2)

tick_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks([i for i in range(0, 12, 2)])
ax.set_xticklabels(tick_labels[::2])
ax.grid(axis='x')

ax.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=4, facecolor='w')
ax.set_xlabel('Month')
ax.set_ylabel('LSE Loadings')

fig.savefig(os.path.join(OUT_DIR, 'lsvi_month.png'), dpi=300,
            bbox_inches='tight')


# Also include the variable importance of a random forest
errors = np.zeros(3)
forests = []
min_sample_leaves = [3, 5, 10]
for i, min_samples_leaf in enumerate(min_sample_leaves):
    forest = RandomForestRegressor(n_estimators=500,
                                   min_samples_leaf=min_samples_leaf,
                                   max_features=None,
                                   random_state=42,
                                   n_jobs=-1)
    forest.fit(X_train, y_train)
    errors[i] = np.mean((y_test -  forest.predict(X_test)) ** 2)
    forests.append(forest)

min_sample_leaf = min_sample_leaves[np.argmin(errors)]
forest = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=min_sample_leaf,
    random_state=42,
    max_features=None,
    n_jobs=-1)
forest.fit(X_std, y)

forest_imp = permutation_importance(forest, X_std, y, random_state=42)
forest_imp /= np.sum(forest_imp)

fig, ax = plt.subplots(figsize=(12, 6))

order = np.argsort(forest_imp)
ax.barh(y=np.arange(5), width=forest_imp[order], color='gray',
        tick_label=np.asarray(cols)[order], height=0.5)
ax.set_xlabel('Variable Importance')

fig.savefig(os.path.join(OUT_DIR, 'rf_airquality_imp.png'),
           dpi=300, bbox_inches='tight')
