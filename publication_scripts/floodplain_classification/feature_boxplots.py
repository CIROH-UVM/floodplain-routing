import os
import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec

include_clusters = ['A', 'B', 'C', 'D', 'E', 'F']
feature_cols = ['el_edap_scaled', 'el_edep_scaled', 'height_scaled', 'w_edep', 'valley_confinement', 'min_rhp', 'vol']
misc_cols = ['slope', 'DASqKm', 'regression_valley_confinement', 'streamorder']
rename_dict = {
    'el_edap_scaled': 'EDZ Access Stage',
    'el_edep_scaled': 'EDZ Exit Stage',
    'height_scaled': 'EDZ Stage Range',
    'w_edep': 'EDZ Width',
    'valley_confinement': 'EDZ Relative Width',
    'vol': 'Diagnostic Size',
    'min_rhp': 'Max Lateral Expansion',
    'slope': 'Slope',
    'DASqKm': 'Drainage Area',
    'regression_valley_confinement': 'Valley Confinement',
    'streamorder': 'Stream Order',
    'celerity_detrended': 'Shape Celerity (m^(2/3))',
    'celerity': 'Celerity (m/s)'
}
cpal = ["#8f00cc", "#cc0000", "#cc7000", "#cdbc00", "#07cc00", "#00cccc", '#2b54d8', '#979797']
significant_groups = {
    'EDZ Access Stage': {'A': 'i', 'B': 'ii', 'C': 'iii', 'D': 'iii', 'E': 'iv', 'F': 'iv'},
    'EDZ Exit Stage': {'A': 'i', 'B': 'ii', 'C': 'iii', 'D': 'iv', 'E': 'v', 'F': 'vi'},
    'EDZ Stage Range': {'A': 'i', 'B': 'ii', 'C': 'iii', 'D': 'ii', 'E': 'iii', 'F': 'i'},
    'EDZ Width': {'A': 'i', 'B': 'i', 'C': 'ii', 'D': 'ii', 'E': 'iii', 'F': 'ii'},
    'Diagnostic Size': {'A': 'i', 'B': 'ii', 'C': 'ii', 'D': 'iii', 'E': 'iv', 'F': 'iv'},
    'EDZ Relative Width': {'A': 'i', 'B': 'ii', 'C': 'iii', 'D': 'ii', 'E': 'iv', 'F': 'i'},
    'Max Lateral Expansion': {'A': 'i', 'B': 'ii', 'C': 'ii', 'D': 'iii', 'E': 'iv', 'F': 'iii'},
    'Drainage Area': {'A': 'i', 'B': 'ii', 'C': 'iii', 'D': 'iv', 'E': 'iii', 'F': 'iv', 'W': 'i', 'X': 'i'},
    'Valley Confinement': {'A': 'i', 'B': 'ii', 'C': 'ii, iii, iv', 'D': 'iii', 'E': 'v', 'F': 'i, iv', 'W': 'iv', 'X': 'vi'},
    'Stream Order': {'A': 'i', 'B': 'ii', 'C': 'iii', 'D': 'iv, v', 'E': 'iii, iv', 'F': 'v', 'W': 'vi', 'X': 'vi'},
    'Slope': {'A': 'i', 'B': 'ii', 'C': 'i', 'D': 'iii', 'E': 'iii', 'F': 'iv', 'W': 'v', 'X': 'i, ii'},
}
significance_locs = {
    'EDZ Access Stage': 'top',
    'EDZ Exit Stage': 'top',
    'EDZ Stage Range': 'top',
    'EDZ Width': 'top',
    'Diagnostic Size': 'top',
    'EDZ Relative Width': 'top',
    'Max Lateral Expansion': 'bottom',
    'Drainage Area': 'top',
    'Valley Confinement': 'top',
    'Stream Order': 'top',
    'Slope': 'top',
}

run_path = sys.argv[1]
with open(run_path, 'r') as f:
    run_dict = json.loads(f.read())
in_path = os.path.join(run_dict['analysis_directory'], 'clustering', 'all_data.csv')
out_dir = os.path.join(run_dict['analysis_directory'], 'clustering')
df = pd.read_csv(in_path)
df = df.rename(columns=rename_dict)
feature_cols = [rename_dict[col] for col in feature_cols]
misc_cols = [rename_dict[col] for col in misc_cols]
df = df[df['cluster'].isin(include_clusters)]
ord = sorted(df['cluster'].unique())

def feature_boxplots():
    size = 6.5
    aspect = (9 / 6.5)
    w = size
    h = size * aspect

    cols = 2
    rows = 4

    fig, ax = plt.subplots(figsize=(w, h), sharey=False, sharex=True)
    ax.remove()
    spec = gridspec.GridSpec(ncols=2*cols, nrows=rows, figure=fig)
    axs = np.empty((rows, cols), dtype=object)
    for i in range(rows-1):
        for j in range(cols):
            tmp_ax = fig.add_subplot(spec[i, 2*j:(2*j)+2])
            axs[i, j] = tmp_ax
    tmp_ax = fig.add_subplot(spec[3, 1:3])
    axs[3, 0] = tmp_ax

    ax_list = [ax for ax in axs.flat if ax is not None]
    for i, ax in enumerate(ax_list):
        c = feature_cols[i]
        sns.boxplot(x='cluster', y=c, data=df, ax=ax, palette=cpal, order=ord, showfliers=False)
        ax.set(xlabel=None, ylabel=None, title=c, facecolor='#f5f5f5')
        if i not in [4, 5, 6]:
            ax.set_xticklabels([])

        # add significance groups
        labels = significant_groups[c]
        if significance_locs[c] == 'top':
            ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0]))
            y = ax.get_ylim()[1] * 0.975
        else:
            ax.set_ylim(ax.get_ylim()[0] - 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0]), ax.get_ylim()[1])
            y = ax.get_ylim()[0] * 0.975
        for ind, clus in enumerate(labels):
            x = ind
            ax.text(x, y, labels[clus], ha='center', va=significance_locs[c], fontsize=10, zorder=100, path_effects=[pe.withStroke(linewidth=4, foreground="white")])


    axs[0, 0].set_ylabel('Meters', fontsize=10)
    axs[1, 0].set_ylabel('Meters', fontsize=10)
    axs[2, 0].set_ylabel('Unitless', fontsize=10)
    axs[3, 0].set_ylabel('Meters', fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'feature_boxplots.pdf'), dpi=300)

def misc_boxplots():
    size = 6.5
    aspect = (4.5 / 6.5)
    w = size
    h = size * aspect

    cols = 2
    rows = 2

    fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(w, h), sharey=False, sharex=True)

    ax_list = [ax for ax in axs.flat if ax.axes is not None]
    yscales = ['log', 'log', 'linear', 'linear']
    for i, ax in enumerate(ax_list):
        c = misc_cols[i]
        sns.boxplot(x='cluster', y=c, data=df, ax=ax, palette=cpal, order=ord, showfliers=False)
        ax.set(xlabel=None, ylabel=None, title=c, facecolor='#f5f5f5', yscale=yscales[i])

        # add significance groups
        labels = significant_groups[c]
        labels = {k: v for k, v in labels.items() if k in include_clusters}

        if yscales[i] == 'log':
            delta = 1 * ax.get_ylim()[1]
            m = 0.85
        else:
            delta = 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            m = 0.975
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + delta)
        y = ax.get_ylim()[1] * m

        for ind, clus in enumerate(labels):
            x = ind
            ax.text(x, y, labels[clus], ha='center', va=significance_locs[c], fontsize=10, zorder=100, path_effects=[pe.withStroke(linewidth=4, foreground="white")])

    for ind, i in enumerate([r'm ${m}^{-1}$', r'${km}^{2}$', 'unitless', 'unitless']):
        ax_list[ind].set_ylabel(i, fontsize=10)


    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'misc_boxplots.pdf'), dpi=300)

# feature_boxplots()
misc_boxplots()