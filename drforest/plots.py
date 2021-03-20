import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


__all__ = ['plot_local_importance', 'plot_variable_importance',
           'plot_local_importance_histogram', 'plot_single_importance']


def label(x, color, label):
    ax = plt.gca()
    ax.text(0, 0.2, label, color='black', #fontweight="bold",
            ha="left", va="center", transform=ax.transAxes)



def plot_variable_importance(importances, plot_type='bar', normalize=False,
                             names=None, xlabel=None, title=None,
                             figsize=(8, 6), ax=None):
    """Plot global variable importances."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    n_features = importances.shape[0]

    if normalize:
        importances = importances / np.sum(importances)

    # re-order from largets to smallest
    order = np.argsort(importances)
    if names is None:
        names = ['Feature {}'.format(i + 1) for i in order]
    else:
        names = names[order]

    margin = 0.1 if plot_type == 'lollipop' else 0.0
    ax.set_xlim(0, importances.max() + margin)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if plot_type == 'bar':
        ax.barh(y=np.arange(n_features), width=importances[order],
                color='gray', tick_label=names, height=0.5)
    elif plot_type == 'lollipop':
        for k in range(n_features):
            ax.hlines(y=k, xmin=0, xmax=importances[order[k]],
                      color='k', linewidth=2)
            ax.plot(importances[order[k]], k, 'o', color='k')
        ax.axvline(x=0, ymin=0, ymax=1, color='k', linestyle='--')


        ax.set_yticks(range(n_features))
        ax.set_yticklabels(names)
    else:
        raise ValueError(
            "Unrecognized plot_type. Should be 'bar' or 'lollipop'")

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel('Variable Importance')

    if title:
        ax.set_title(title)

    return fig, ax


def plot_local_importance(importances, sort_features=False, feature_names=None,
                          figsize=(10, 6), palette='Set3', scale='count',
                          inner='quartile'):
    n_features = importances.shape[1]
    if feature_names is None:
        feature_names = ["Feature {}".format(i + 1) for i in range(n_features)]
    feature_names = np.asarray(feature_names)

    if sort_features:
        order = np.argsort(np.var(importances, axis=0))[::-1]
        importances = importances[:, order]
        feature_names = feature_names[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    data = pd.melt(pd.DataFrame(importances, columns=feature_names))
    sns.violinplot(x='variable', y='value', data=data, palette=palette,
                   scale=scale, inner='quartile', ax=ax)


    ax.set_xlabel('')
    ax.set_ylabel('LSVI Loadings', fontsize=18)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)

    return fig, ax

def plot_local_importance_histogram(
        directions, importances=None, feature_names=None,
        figsize=(10, 6), color='0.3', bins=30):
    """Plot marginal distribution of local subspace variable importances."""
    n_samples, n_features = directions.shape

    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0),
                               "figure.figsize": figsize,
                               "font.family": "serif"})

    if feature_names is None:
        feature_names = np.asarray(["Feature {}".format(i + 1) for
                                    i in range(n_features)])

    data = pd.DataFrame({'x': directions.T.ravel(),
                         'g': np.repeat(feature_names, n_samples)})

    if importances is not None:
        sort_ids = np.argsort(importances)[::-1]
    else:
        sort_ids = np.argsort(np.var(directions, axis=0))[::-1]
    pal = sns.cubehelix_palette(n_features, light=0.8, reverse=True)
    g = sns.FacetGrid(data, row='g', hue='g', aspect=15, height=0.5,
                      palette=pal, xlim=(-1.5, 1.1),
                      row_order=feature_names[sort_ids],
                      hue_order=feature_names[sort_ids])

    # add histograms
    g.map(sns.distplot, "x", hist_kws={'color': color, 'alpha': 1},
          bins=bins, kde=False)
    g.map(sns.distplot, "x", hist_kws={'color': 'w', 'lw': 1.5}, bins=bins,
          kde=False)

    g.fig.subplots_adjust(hspace=-0.25)
    g.map(label, "x")
    g.set_titles("")
    g.set_xlabels("Loadings", fontsize=14)
    g.set(yticks=[])
    g.set(xticks=[-1.0, -0.5, 0, 0.5, 1.0])
    g.set_xticklabels([-1.0, -0.5, 0, 0.5, 1.0], size=12)
    g.despine(bottom=True, left=True)

    return g


def plot_single_importance(imp_x, figsize=(10, 12), rotation=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    color = ['tomato' if x > 0 else 'cornflowerblue' for x in imp_x]

    n_features = imp_x.shape[0]
    ax.bar(np.arange(n_features), imp_x, color=color)

    ax.axhline(0, color='black', linestyle='-', lw=1)

    ax.set_ylabel('Importance', fontsize=18)
    ax.set_ylim(-1, 1)

    ax.set_xlabel('')
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.set_xticks(np.arange(n_features))
    ax.set_xticklabels(
        ['Feature {}'.format(i + 1) for i in range(n_features)],
        rotation=rotation)

    return ax
