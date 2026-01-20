import tensorflow as tf
import numpy as np
from collections import defaultdict,Counter
from typing import Any, List, Dict,Tuple
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd



def plot_heatmap(df_results, heatmap_type='c_j', fixed_param_values=None, 
                 metric='accuracy_median', figsize=None, cmap='RdYlGn', 
                 annot=True, save=False, filename=None):
    """
    Trace des heatmaps d'accuracy selon le type choisi.
    
    Parameters
    ----------
    metric : str
        Métrique à afficher : 'accuracy_median', 'accuracy_mean', 'accuracy_std', etc.
    """
    
    config = {
        'c_j': {
            'index': 'j_augment',
            'columns': 'c_test_class',
            'fixed': 'hidden_dim',
            'index_label': 'j (augmentation)',
            'columns_label': 'Classe (c)',
            'fixed_label': 'Hidden Dim'
        },
        'c_hidden': {
            'index': 'hidden_dim',
            'columns': 'c_test_class',
            'fixed': 'j_augment',
            'index_label': 'Hidden Dim',
            'columns_label': 'Classe (c)',
            'fixed_label': 'j'
        },
        'j_hidden': {
            'index': 'hidden_dim',
            'columns': 'j_augment',
            'fixed': 'c_test_class',
            'index_label': 'Hidden Dim',
            'columns_label': 'j (augmentation)',
            'fixed_label': 'Classe'
        }
    }
    
    if heatmap_type not in config:
        raise ValueError(f"heatmap_type doit être parmi {list(config.keys())}")
    
    cfg = config[heatmap_type]
    
    if fixed_param_values is None:
        fixed_param_values = sorted(df_results[cfg['fixed']].unique())
    
    n_plots = len(fixed_param_values)
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    for idx, fixed_val in enumerate(fixed_param_values):
        ax = axes[idx]
        
        df_filtered = df_results[df_results[cfg['fixed']] == fixed_val]
        
        df_pivot = df_filtered.pivot_table(
            index=cfg['index'],
            columns=cfg['columns'],
            values=metric,
            aggfunc='mean'
        )
        
        sns.heatmap(
            df_pivot,
            annot=annot,
            fmt='.2f',
            cmap=cmap,
            vmin=0,
            vmax=1,
            ax=ax,
            cbar=True,
            annot_kws={'size': 8},
            linewidths=0.5
        )
        
        ax.set_title(f'{cfg["fixed_label"]} = {fixed_val}', fontsize=12, fontweight='bold')
        ax.set_xlabel(cfg['columns_label'], fontsize=10)
        ax.set_ylabel(cfg['index_label'] if idx % n_cols == 0 else '', fontsize=10)
    
    metric_name = metric.replace('accuracy_', '').capitalize()
    title_map = {
        'c_j': f'Accuracy ({metric_name}): Classe (c) vs Augmentation (j)',
        'c_hidden': f'Accuracy ({metric_name}): Classe (c) vs Hidden Dim',
        'j_hidden': f'Accuracy ({metric_name}): Augmentation (j) vs Hidden Dim'
    }
    fig.suptitle(title_map[heatmap_type], fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save:
        if filename is None:
            filename = f'heatmap_{heatmap_type}_{metric}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardée: {filename}")
    
    plt.show()
    
    return fig, axes
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