import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sns


color = np.asarray(sns.color_palette('tab20'))

rf_vals = np.zeros((len(glob.glob('results/*csv')), 500))
drf_vals = np.zeros((len(glob.glob('results/*csv')), 500))

fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(13, 5), sharey=True)

DIR_NAME = 'synthetic_data_errors'
FILE_DICT = {
    'errors_simulation1.csv': 'Simulation #1',
    'errors_simulation2.csv': 'Simulation #2',
    'errors_simulation3.csv': 'Simulation #3',
    'errors_simulation4.csv': 'Simulation #4',
    'errors_simulation5.csv': 'Simulation #5',
    'errors_simulation6.csv': 'Simulation #6',
    'errors_simulation7.csv': 'Simulation #7',
    'errors_friedman1.csv': 'Friedman #1',
    'errors_friedman2.csv': 'Friedman #2',
    'errors_friedman3.csv': 'Friedman #3'
}


for i, file_name in enumerate(FILE_DICT.keys()):
    file_path = os.path.join(DIR_NAME, file_name)

    if not os.path.exists(file_path):
        continue

    df = pd.read_csv(file_path)
    col_names = df.columns

    rf_df = df[[c for c in col_names if c.split()[0] == 'rf']]
    rf_final_errors = rf_df.iloc[-1]
    rf_final_errors[rf_final_errors == 0] = np.nan
    rf_df = rf_df.iloc[:, rf_final_errors.argmin()]

    drf_df = df[[c for c in col_names if c.split()[0] == 'drrf']]
    drf_final_errors = drf_df.iloc[-1]
    drf_final_errors[drf_final_errors == 0] = np.nan
    drf_df = drf_df.iloc[:, drf_final_errors.argmin()]

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

    ax.flat[i].axhline(1, color='k', linestyle=':')


ax.flat[2].legend(loc='upper center', frameon=False, bbox_to_anchor=(0.7, 1.4), ncol=2, fontsize=14)
plt.subplots_adjust(hspace=0.3)
plt.show()
fig.savefig('Figure S.3.png', dpi=300, bbox_inches='tight')
