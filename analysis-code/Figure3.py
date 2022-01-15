import os

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family='serif')

file_names = ['simulation1', 'simulatoin2', 'simulation3', 'simulation4']
metric = 'trcor'
signal_to_noise = [0.75, 1.0, 1.5, 3.0, 5.0]
stn_remap = {0.75 : '3:4', 1.0 : '1:1', 1.5 : '3:2', 3.0 : '3:1', 5.0: '5:1'}

DIR_NAME = "LPD_results"

fig, ax = plt.subplots(figsize=(20, 4), ncols=4, sharey=True)

for s, fn in enumerate(file_names):
    data = None
    for p in signal_to_noise[::-1]:
        df = pd.read_csv(os.path.join(DIR_NAME,
            fn + '_p10_s' + str(p) + '_' + metric + '.csv'))

        df = pd.melt(df)
        df['s'] = stn_remap[p]
        if data is None:
            data = df.copy()
        else:
            data = pd.concat((data, df))

    # rename some variables
    data.loc[data['variable'] == 'Global SAVE', 'variable'] = 'SAVE'
    data.loc[data['variable'] == 'Global SIR', 'variable'] = 'SIR'

    sns.boxplot(x='variable', y='value', hue='s', data=data, ax=ax[s],
                order=[r'DRF', 'SIR', 'SAVE', 'Local SIR'], fliersize=0,
                hue_order=['5:1', '3:1', '3:2', '1:1', '3:4'])
    ax[s].legend([],[], frameon=False)
    ax[s].set_xlabel('')
    ax[s].set_ylabel('')
    ax[s].set_title('Simulation {}'.format(s + 1), fontsize=24)

    if s == 0:
        ax[s].set_ylabel('Trace Correlation', fontsize=18)
    ax[s].set_ylim(0, 1)

    ax[s].set_xticklabels(ax[s].get_xticklabels(), fontsize=12)
    ax[s].tick_params(axis='y', labelsize=16)


legend = ax[-1].legend(title='signal:noise', bbox_to_anchor=(1, 1),
    loc='upper left', fontsize=18, frameon=False)
plt.setp(legend.get_title(), fontsize=18)
plt.subplots_adjust(right=0.9)
fig.savefig('Figure 3.png', dpi=300, bbox_inches='tight')
