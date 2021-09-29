import timeit
import numpy as np
import matplotlib.pyplot as plt


setup_str = """
from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor
from drforest.tree import DecisionTreeRegressor, DimensionReductionTreeRegressor
from drforest.datasets import make_simulation1

X, y = make_simulation1(n_samples={0}, n_features=5, noise=1, random_state=1)
"""

n_reps = 10
sample_sizes = np.linspace(100, 10000, 10).astype(int)
out_rf = np.zeros(10)
out_drf = np.zeros(10)
out_drf_screen = np.zeros(10)
out_skrf = np.zeros(10)
for i, n in enumerate(sample_sizes):
    res = timeit.repeat(
        "DecisionTreeRegressor(min_samples_leaf=1, max_depth=None).fit(X, y)",
        setup=setup_str.format(n), number=1, repeat=n_reps)
    out_rf[i] = np.median(res)

    res = timeit.repeat(
        "DimensionReductionTreeRegressor(min_samples_leaf=1, max_depth=None).fit(X, y)",
        setup=setup_str.format(n), number=1, repeat=n_reps)
    out_drf[i] = np.median(res)

    #res = timeit.repeat(
    #    "DimensionReductionTreeRegressor(min_samples_leaf=1, max_features=2, max_depth=None).fit(X, y)",
    #    setup=setup_str.format(n), number=1, repeat=n_reps)
    #out_drf_screen[i] = np.median(res)

    res = timeit.repeat(
        "SklearnDecisionTreeRegressor(min_samples_leaf=1, max_depth=None).fit(X, y)",
        setup=setup_str.format(n), number=1, repeat=n_reps)
    out_skrf[i] = np.median(res)


fig, ax = plt.subplots(figsize=(18, 4), ncols=2, sharey=True)
ax[0].plot(sample_sizes, out_drf, 'ko-', label='Dimension Reduction Tree')
#ax[0].plot(sample_sizes, out_drf_screen, 'bo-', label='Dimension Reduction Tree [max_features]')
ax[0].plot(sample_sizes, out_rf, 'ks--', label='Axis-Aligned Decision Tree')
ax[0].plot(sample_sizes, out_skrf, 'k^:', label='Axis-Aligned Decision Tree (Scikit-Learn)')
ax[0].set_ylabel('Time [sec]', fontsize=12)
ax[0].set_xlabel('Sample Size', fontsize=12)
#ax[0].set_yticklabels(ax[0].get_yticklabels(), fontsize=10)
#ax[0].set_xticklabels(ax[0].get_xticklabels(), fontsize=10)

#fig.savefig('bench_sim1.pdf', dpi=300, bbox_inches='tight')


setup_str = """
from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor
from drforest.tree import DecisionTreeRegressor, DimensionReductionTreeRegressor
from drforest.datasets import make_simulation1

X, y = make_simulation1(n_samples=2000, n_features={0}, noise=1, random_state=1)
"""

feature_sizes = np.linspace(2, 50, 10).astype(int)
out_rf = np.zeros(10)
out_drf = np.zeros(10)
out_drf_screen = np.zeros(10)
out_skrf = np.zeros(10)
for i, n in enumerate(feature_sizes):
    res = timeit.repeat(
        "DecisionTreeRegressor(min_samples_leaf=1, max_depth=None).fit(X, y)",
        setup=setup_str.format(n), number=1, repeat=n_reps)
    out_rf[i] = np.median(res)

    res = timeit.repeat(
        "DimensionReductionTreeRegressor(min_samples_leaf=1, max_depth=None).fit(X, y)",
        setup=setup_str.format(n), number=1, repeat=n_reps)
    out_drf[i] = np.median(res)

    #res = timeit.repeat(
    #    "DimensionReductionTreeRegressor(min_samples_leaf=1, max_features=0.2, max_depth=None).fit(X, y)",
    #    setup=setup_str.format(n), number=1, repeat=n_reps)
    #out_drf_screen[i] = np.median(res)

    res = timeit.repeat(
        "SklearnDecisionTreeRegressor(min_samples_leaf=1, max_depth=None).fit(X, y)",
        setup=setup_str.format(n), number=1, repeat=n_reps)
    out_skrf[i] = np.median(res)


ax[1].plot(feature_sizes, out_drf, 'ko-', label='Dimension Reduction Tree')
#ax[1].plot(feature_sizes, out_drf_screen, 'bo-', label='Dimension Reduction Tree [max_features]')
ax[1].plot(feature_sizes, out_rf, 'ks--', label='Axis-Aligned Decision Tree')
ax[1].plot(feature_sizes, out_skrf, 'k^:', label='Axis-Aligned Decision Tree (Scikit-Learn)')
ax[1].set_xlabel('Number of Features', fontsize=12)
#ax[1].set_yticklabels(ax[1].get_yticklabels(), fontsize=10)
#ax[1].set_xticklabels(ax[1].get_xticklabels(), fontsize=10)
ax[1].legend(loc='upper left', frameon=False, bbox_to_anchor=(1, 1), fontsize=12)

plt.tick_params(axis='both', which='major', labelsize=12)
plt.subplots_adjust(right=0.8)
fig.savefig('bench_sim1.pdf', dpi=300, bbox_inches='tight')