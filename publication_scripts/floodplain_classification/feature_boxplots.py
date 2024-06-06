import os
import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

include_clusters = ['A', 'B', 'C', 'D', 'E', 'F']
feature_cols = ['el_edap_scaled', 'el_edep_scaled', 'height_scaled', 'w_edep', 'vol', 'valley_confinement', 'min_rhp']
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

    cols = int(np.ceil(np.sqrt(len(feature_cols))))
    rows = int(len(feature_cols) / cols) + 1
    cols = 3
    rows = np.ceil(len(feature_cols) / cols).astype(int)

    fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(13, 9), sharey=False, sharex=True)
    axs[1, 2].remove()
    axs[2, 2].remove()

    ax_list = [ax for ax in axs.flat if ax.axes is not None]
    for i, ax in enumerate(ax_list):
        c = feature_cols[i]
        sns.boxplot(x='cluster', y=c, data=df, ax=ax, palette=cpal, order=ord, showfliers=False)
        ax.set(xlabel=None, ylabel=None, title=c, facecolor='#f5f5f5')

        # add significance groups
        labels = significant_groups[c]
        for ind, clus in enumerate(labels):
            x = ind
            if i == len(ax_list) - 1:
                y = ax.get_ylim()[0] + (0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]))
            else:
                y = ax.get_ylim()[1] - (0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]))
            ax.text(x, y, labels[clus], ha='center', va='center', fontsize=10, zorder=100)
            x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            ell = Ellipse((x, y), width=0.05 * x_range, height=0.075 * y_range, angle=0, edgecolor='black', facecolor='white', lw=1, zorder=99)
            ax.add_patch(ell)

    axs[0, 0].set_ylabel('Meters', fontsize=10)
    axs[1, 0].set_ylabel('Meters', fontsize=10)
    axs[2, 0].set_ylabel('Unitless', fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'feature_boxplots.pdf'), dpi=300)

def misc_boxplots():
    cols = 2
    rows = 2

    fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(13, 8), sharey=False, sharex=True)

    ax_list = [ax for ax in axs.flat if ax.axes is not None]
    yscales = ['log', 'log', 'linear', 'log']
    for i, ax in enumerate(ax_list):
        c = misc_cols[i]
        sns.boxplot(x='cluster', y=c, data=df, ax=ax, palette=cpal, order=ord, showfliers=False)
        ax.set(xlabel=None, ylabel=None, title=c, facecolor='#f5f5f5', yscale=yscales[i])

        # add significance groups
        label_ax = ax.twinx()
        label_ax.set_yticks([])
        label_ax.set_yticklabels([])
        label_ax.set_ylim(0, 1)
        labels = significant_groups[c]
        for ind, clus in enumerate(labels):
            if clus not in include_clusters:
                continue
            x = ind
            if i == 1 or i == 3:
                y = 0.05
            else:
                y = 0.95
            l = labels[clus]
            label_ax.text(x, y, l, ha='center', va='center', fontsize=10, zorder=100)
            ell = Ellipse((x, y), width=(0.07 * len(l)) + 0.05, height=0.07, angle=0, edgecolor='black', facecolor='white', lw=1, zorder=99)
            label_ax.add_patch(ell)

    for ind, i in enumerate([r'${km}^{2}$', 'unitless', 'unitless', r'm ${m}^{-1}$']):
        ax_list[ind].set_ylabel(i, fontsize=10)


    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'misc_boxplots.png'), dpi=300)

# feature_boxplots()
misc_boxplots()