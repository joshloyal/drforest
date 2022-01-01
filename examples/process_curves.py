import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sns


color = np.asarray(sns.color_palette('tab20'))

rf_vals = np.zeros((len(glob.glob('results/*csv')), 500))
drf_vals = np.zeros((len(glob.glob('results/*csv')), 500))

file_names = list(glob.glob('results/*csv'))
#fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(12, 6), sharey=True)
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(13, 5), sharey=True)

FILE_DICT = {
    'errors_simulation1__2000n_1000t_50r.csv': 'Simulation #1',
    'errors_simulation2__2000n_1000t_50r.csv': 'Simulation #2',
    'errors_simulation3__2000n_1000t_50r.csv': 'Simulation #3',
    'errors_simulation4__2000n_1000t_50r.csv': 'Simulation #4',
    'errors_simulation5__2000n_1000t_50r.csv': 'Simulation #5',
    'errors_simulation6__2000n_1000t_50r.csv': 'Simulation #6',
    'errors_simulation7__2000n_1000t_50r.csv': 'Simulation #7',
    'errors_friedman1__2000n_1000t_50r.csv': 'Friedman #1',
    'errors_friedman2__2000n_1000t_50r.csv': 'Friedman #2',
    'errors_friedman3__2000n_1000t_50r.csv': 'Friedman #3'
}

for i, file_name in enumerate(FILE_DICT.keys()):
    df = pd.read_csv('errors/' + file_name)
    col_names = df.columns

    rf_df = df[[c for c in col_names if c.split()[0] == 'rf']]
    rf_final_errors = rf_df.iloc[-1]
    rf_final_errors[rf_final_errors == 0] = np.nan
    rf_df = rf_df.iloc[:, rf_final_errors.argmin()]

    drf_df = df[[c for c in col_names if c.split()[0] == 'drrf']]
    drf_final_errors = drf_df.iloc[-1]
    drf_final_errors[drf_final_errors == 0] = np.nan
    drf_df = drf_df.iloc[:, drf_final_errors.argmin()]

    #ax.flat[i].plot(rf_df.values, 'k--', lw=3, label = 'Random Forest')
    #ax.flat[i].plot(drf_df.values, 'k-', lw=3, label = 'Dimension Reduction Forest')

    ax.flat[i].plot(drf_df.values / rf_df.values[-1], 'k--', lw=3, label = 'DRF / RF 500 Trees')
    ax.flat[i].plot(drf_df.values / rf_df.values, 'k-', lw=3, label = 'DRF / RF Same Number of Trees')

    ax.flat[i].set_title(FILE_DICT[file_name])

    if i in [0, 5]:
        ax.flat[i].set_ylabel('Test MSE Ratio', fontsize=16)

    if i >= 5:
        ax.flat[i].set_xlabel('Number of Trees', fontsize=12)
    else:
        ax.flat[i].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax.flat[i].set_xscale('log')
    ax.flat[i].set_xticks([1, 10, 100, 500])
    ax.flat[i].set_ylim(0.2, 2.8)
    #ax.flat[i].set_xticks([0, 100, 200, 300, 400, 500])

    ax.flat[i].axhline(1, color='k', linestyle=':')

    #plt.plot(drrf_df.values / rf_df.values[-1], 'k-', label='DRF / RF 500 Trees')
    #plt.plot(drrf_df.values / rf_df.values, 'k--', label='DRF / RF Same Number of Trees')

    #ax.flat[i].plot(drf_df.values / rf_df.values[-1], '-', c=color[i], lw=2, label = file_name)
    #ax.flat[i].plot(drf_df.values / rf_df.values, '--', c=color[i], lw=2)
    #ax[1].annotate('stuff',
    #    xy=(500, drf_df.values[-1] / rf_df.values[-1]),
    #    xytext=(5,0),
    #    textcoords='offset points',
    #    va='center')
    #rf_vals[i] = rf_df.values
    #drf_vals[i] = drf_df.values

#rf_vals = np.mean(rf_vals, axis=0)
#drf_vals = np.mean(drf_vals, axis=0)
#plt.plot(drf_vals / rf_vals[-1], 'k-', lw=2)
#plt.plot(drf_vals / rf_vals, 'k--', lw=2)

#ax[0].axhline(1, color='k', linestyle=':')
#ax[1].axhline(1, color='k', linestyle=':')
ax.flat[2].legend(loc='upper center', frameon=False, bbox_to_anchor=(0.7, 1.4), ncol=2, fontsize=14)
plt.subplots_adjust(hspace=0.3)
plt.show()
fig.savefig('test.png', dpi=300, bbox_inches='tight')
