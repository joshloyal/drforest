import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


__all__ = ['plot_local_importance', 'plot_variable_importance']


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


def plot_local_importance(directions, importances=None, feature_names=None,
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
