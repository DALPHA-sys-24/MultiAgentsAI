import tensorflow as tf
import numpy as np
from collections import defaultdict,Counter
from typing import Any, List, Dict,Tuple
from matplotlib import pyplot as plt

def count_values(tensor):
    """
    Compte le nombre d'occurrences de chaque valeur dans un tenseur 1D.
    Retourne un dictionnaire {valeur: occurrences}.
    """
    x = tf.reshape(tensor, [-1]).numpy()  # flatten + numpy
    counts = Counter(x.tolist())
    return dict(counts)

def plot_results(df_list, x_min, x_max, y_min, y_max, 
                       grid_shape=(3, 3), alpha=0.1, lw=2,
                       figsize=(10, 8), dpi=250, loc='lower right', markersize=2,
                       linestyle='--', fontsize=12, use_grid=False, xlabel='N',
                       ylabel='Accuracy', name_fig=None, save_fig=True,
                       show_fig=False, percentile=True, titles=None):
    """
    Affiche plusieurs graphiques dans une figure selon une grille (ex: 2x2).
    Chaque élément de df_list est un dictionnaire contenant les DataFrames à tracer.
    
    Parameters
    ----------
    grid_shape : tuple
        (nrows, ncols) pour organiser les sous-graphes (ex: (2, 2)).
    titles : list of str
        Liste optionnelle des titres pour chaque sous-graphe.
    """

    COLORS = ["k", "b", "r", "c", "y", "m", "g"]
    MARKER = ['*', 'o', '^', 'v', 's', 'd']

    nrows, ncols = grid_shape
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    axes = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]
    if  not isinstance(df_list,list):
        raise SystemExit
        
    for idx, df in enumerate(df_list):
        if idx >= len(axes):
            break  # Si plus d'images que de cases
        ax = axes[idx]
        ax.axis([x_min, x_max, y_min, y_max])

        if percentile:
            for i, name in enumerate(df.keys()):
                ax.plot(df[name].p, df[name].med, lw=lw, label=name,
                        linestyle=linestyle, color=COLORS[i % len(COLORS)],
                        marker=MARKER[i % len(MARKER)], markersize=markersize)
                ax.fill_between(df[name].p, df[name].q1, df[name].q2,
                                color=COLORS[i % len(COLORS)], alpha=alpha)
        else:
            for i, name in enumerate(df.keys()):
                ax.plot(df[name].index, df[name].med, lw=lw, label=name,
                        linestyle=linestyle,
                        color=COLORS[i % len(COLORS)])
                ax.fill_between(df[name].index, df[name].q1, df[name].q2,
                                color=COLORS[i % len(COLORS)], alpha=alpha)

        if titles and idx < len(titles):
            ax.set_title(titles[idx], fontsize=fontsize + 1)

        ax.legend(loc=loc, fontsize=fontsize - 2)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize - 2)
        ax.grid(use_grid)

    # Supprimer les axes inutilisés si la grille est plus grande que le nb d'images
    for j in range(len(df_list), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if save_fig and name_fig:
        plt.savefig(f"{name_fig}.pdf", dpi=dpi, bbox_inches="tight")

    if show_fig:
        plt.tight_layout()
        plt.show()

    plt.close(fig)
    
    return fig, axes


# Exemple d'utilisation
if __name__ == "__main__":
    pass