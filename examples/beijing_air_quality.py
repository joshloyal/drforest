import os
import itertools

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import get_cmap
from matplotlib.colors import rgb2hex

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from drforest.datasets import load_beijing
from drforest.ensemble import DimensionReductionForestRegressor
from drforest.ensemble import permutation_importance


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

# special palette for feature importances
pal = {cols[i] : rgb2hex(get_cmap('tab10')(i)) for i in range(4)}
pal[cols[-1]] = rgb2hex(get_cmap('tab10')(6))

X = data[cols].values
y = data['PM25_Concentration'].values
month_col = 4

scaler =  StandardScaler().fit(X)
X_std = scaler.transform(X)

train, test = train_test_split(np.arange(X.shape[0]), test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = X_std[train], X_std[test], y[train], y[test]

# tune and train a dimension reduction forest (DRF)
#drforests = []
#min_sample_leaves = [3, 5, 10]
#n_slices = [10, 100, 500]
#params = list(itertools.product(min_sample_leaves, n_slices))
#errors = np.zeros(len(params))
#for i, (min_sample_leaf, n_slice) in enumerate(params):
#    drforest = DimensionReductionForestRegressor(
#        n_estimators=500,
#        min_samples_leaf=min_sample_leaf,
#        n_slices=n_slice,
#        random_state=42,
#        store_X_y=True,
#        n_jobs=-1)
#    drforest.fit(X_train, y_train)
#    errors[i] = np.mean((y_test -  drforest.predict(X_test)) ** 2)
#    drforests.append(drforest)
#
#min_sample_leaf, n_slice = params[np.argmin(errors)]
#print('min_sample_leaf = ', min_sample_leaf)
#print('n_slices = ', n_slice)
min_sample_leaf = 3
n_slice = np.sqrt(X.shape[0])
print(n_slice)
#n_slice = 200
# re-train on full dataset
drforest = DimensionReductionForestRegressor(
    n_estimators=500,
    min_samples_leaf=min_sample_leaf,
    n_slices=n_slice,
    random_state=42,
    store_X_y=True,
    oob_mse=True,
    n_jobs=-1)
drforest.fit(X_std, y)
r_sq = 1 - np.mean((drforest.predict(X_std) - y) ** 2) / np.var(y)
print('DRF R2 = ', r_sq)

# extract local principal directions
importances = drforest.local_principal_direction(X_std, exclude_cols=[month_col], k=3, n_jobs=-1)
importances = np.sign(importances[:, 0]).reshape(-1, 1) * importances

# visualize LPD's as a function of month and their marginal distributions
fig, ax = plt.subplots(figsize=(20, 6), ncols=2)
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
    ax[1].plot(loading_medians[:, p], 'o-', lw=2, label=cols[p], markersize=8,
               linestyle='--')
    ax[1].fill_between(np.arange(12),
                       loading_low[:, p], loading_up[:, p], edgecolor='k', alpha=0.2)

tick_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax[1].set_xticks([i for i in range(0, 12, 2)])
ax[1].set_xticklabels(tick_labels[::2], fontsize=16)
ax[1].tick_params(axis='y', labelsize=16)
ax[1].grid(axis='x')

ax[1].legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=4, facecolor='w')
ax[1].set_xlabel('Month', fontsize=20)
ax[1].set_ylabel('LPD Loadings', fontsize=16)
ax[1].yaxis.set_tick_params(labelleft=True)

# re-calculate importances with month included for marginal plots
importances = drforest.local_principal_direction(X_std, n_jobs=-1)
importances = np.sign(importances[:, 0]).reshape(-1, 1) * importances
imp = pd.melt(pd.DataFrame(importances, columns=cols))

order = np.argsort(np.var(importances, axis=0))[::-1]
sns.violinplot(x='variable', y='value', data=imp, order=np.asarray(cols)[order],
               inner='quartile', palette=pal, ax=ax[0], scale='count')

ax[0].set_ylabel('LPD Loadings', fontsize=18)
ax[0].set_xlabel('')
ax[0].tick_params(axis='y', labelsize=16)
ax[0].tick_params(axis='x', labelsize=16)

fig.savefig(os.path.join(OUT_DIR, 'lpd_month.png'), dpi=300,
            bbox_inches='tight')


# Also include the variable importance of a random forest
#errors = np.zeros(3)
#forests = []
#min_sample_leaves = [3, 5, 10]
#for i, min_samples_leaf in enumerate(min_sample_leaves):
#    forest = RandomForestRegressor(n_estimators=500,
#                                   min_samples_leaf=min_samples_leaf,
#                                   max_features=None,
#                                   random_state=42,
#                                   n_jobs=-1)
#    forest.fit(X_train, y_train)
#    errors[i] = np.mean((y_test -  forest.predict(X_test)) ** 2)
#    forests.append(forest)
#
#min_sample_leaf = min_sample_leaves[np.argmin(errors)]
#forest = RandomForestRegressor(
#    n_estimators=500,
#    min_samples_leaf=min_sample_leaf,
#    random_state=42,
#    max_features=None,
#    n_jobs=-1)
#forest.fit(X_std, y)
#r_sq = 1 - np.mean((forest.predict(X_std) - y) ** 2) / np.var(y)
#print('RF R2 = ', r_sq)
#
#
## global permutation-based importance
#fig, ax = plt.subplots(figsize=(18, 6), ncols=2, sharex=True)
#
#forest_imp = permutation_importance(forest, X_std, y, random_state=42)
#forest_imp /= np.sum(forest_imp)
#order = np.argsort(forest_imp)
#ax[0].barh(y=np.arange(5), width=forest_imp[order], color='gray',
#        tick_label=np.asarray(cols)[order], height=0.5)
#ax[0].set_xlabel('RF Variable Importance')
#
#forest_imp = drforest.feature_importances_
#order = np.argsort(forest_imp)
#ax[1].barh(y=np.arange(5), width=forest_imp[order], color='gray',
#        tick_label=np.asarray(cols)[order], height=0.5)
#ax[1].set_xlabel('DRF Variable Importance')
#
#fig.savefig(os.path.join(OUT_DIR, 'airquality_imp.png'),
#           dpi=300, bbox_inches='tight')
