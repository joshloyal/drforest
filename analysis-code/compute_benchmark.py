import timeit
import numpy as np
import matplotlib.pyplot as plt


setup_str = """
import numpy as np

from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor
from drforest.tree import DecisionTreeRegressor, DimensionReductionTreeRegressor
from drforest.datasets import make_simulation1

X, y = make_simulation1(n_samples={0}, n_features=5, noise=1, random_state=1)

# sort
y_order = np.argsort(y)
y = y[y_order]
X = X[y_order, :]
"""

fontsize = 16
n_reps = 30
n_number = 1
n_min = 150
n_max = 10000
sample_sizes = np.linspace(n_min, n_max, 10).astype(int)
out_rf = np.zeros(10)
out_drf = np.zeros(10)
out_drf_screen = np.zeros(10)
out_skrf = np.zeros(10)
stat_func = np.median
for i, n in enumerate(sample_sizes):
    res = np.asarray(timeit.repeat(
        "DecisionTreeRegressor(min_samples_leaf=1, max_depth=None).fit(X, y)",
        setup=setup_str.format(n), number=n_number, repeat=n_reps))
    out_rf[i] = stat_func(res / n_number)

    res = timeit.repeat(
        "DimensionReductionTreeRegressor(min_samples_leaf=1, max_depth=None, presorted=True).fit(X, y)",
        setup=setup_str.format(n), number=1, repeat=n_reps)
    out_drf[i] = np.median(res)


fig, ax = plt.subplots(figsize=(18, 8), nrows=2, ncols=2, sharey='row', sharex='col')
ax[0, 0].plot(sample_sizes, out_drf, 'ko-', lw=3, ms=10, label='Dimension Reduction Tree')
ax[0, 0].plot(sample_sizes, out_rf, 'ks--', lw=3, ms=10, label='Axis-Aligned Decision Tree')
ax[0, 0].set_title(r'$p = 5$', fontsize=fontsize)
ax[0, 0].set_ylabel('Time [sec]', fontsize=fontsize)
plt.setp(ax[0, 0].get_yticklabels(), fontsize=fontsize)
ax[0, 0].legend(loc='upper left', frameon=False, fontsize=fontsize)

ax[1, 0].plot(sample_sizes, out_drf / out_rf, 'k^:', lw=3, ms=10)
ax[1, 0].set_xlabel('Sample Size', fontsize=fontsize)
ax[1, 0].set_xticks([n_min, 2000, 4000, 6000, 8000, n_max])
plt.setp(ax[1, 0].get_xticklabels(), fontsize=fontsize)


setup_str = """
from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor
from drforest.tree import DecisionTreeRegressor, DimensionReductionTreeRegressor
from drforest.datasets import make_simulation1

X, y = make_simulation1(n_samples=2000, n_features={0}, noise=1, random_state=1)
"""

p_min = 5
p_max = 50
feature_sizes = np.linspace(p_min, p_max, 10).astype(int)
out_rf = np.zeros(10)
out_drf = np.zeros(10)
out_drf_screen = np.zeros(10)
out_skrf = np.zeros(10)
for i, n in enumerate(feature_sizes):
    res = np.asarray(timeit.repeat(
        "DecisionTreeRegressor(min_samples_leaf=1, max_depth=None).fit(X, y)",
        setup=setup_str.format(n), number=n_number, repeat=n_reps))
    out_rf[i] = stat_func(res / n_number)

    res = np.asarray(timeit.repeat(
        "DimensionReductionTreeRegressor(min_samples_leaf=1, max_depth=None).fit(X, y)",
        setup=setup_str.format(n), number=n_number, repeat=n_reps))
    out_drf[i] = stat_func(res / n_number)


ax[0, 1].plot(feature_sizes, out_drf, 'ko-', lw=3, ms=10, label='Dimension Reduction Tree')
ax[0, 1].plot(feature_sizes, out_rf, 'ks--', lw=3, ms=10, label='Axis-Aligned Decision Tree')
ax[0, 1].set_title(r'$n = 2000$', fontsize=fontsize)
ax[0, 1].legend(loc='upper left', frameon=False, fontsize=fontsize)

ax[1, 1].plot(feature_sizes, out_drf / out_rf, 'k^:', lw=3, ms=10)
ax[1, 1].set_xlabel('Number of Features', fontsize=fontsize)
ax[1, 1].set_xticks([p_min, 10, 20, 30, 40, p_max])
ax[1, 0].set_ylabel('Ratio of Run Times [DRT / DT]', fontsize=fontsize)
plt.setp(ax[1, 0].get_yticklabels(), fontsize=fontsize)
ax[1, 0].set_ylim(0, ax[1, 0].get_ylim()[-1])

plt.tick_params(axis='both', which='major', labelsize=fontsize)

plt.subplots_adjust(right=0.8)
fig.savefig('Figure S.1.png', dpi=300, bbox_inches='tight')
