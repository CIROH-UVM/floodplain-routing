import json
import pandas as pd
import itertools as it
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import seaborn as sns
from scipy.stats import kstest, anderson, shapiro, normaltest
from sklearn import linear_model
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
import sys
import matplotlib.patheffects as pe

c1 = '#32d4e6'
c2 = '#faa319'
c3 = 'k'
pt_size = 10
pt_alpha = 0.5
background_color = '#f0f0f0'

run_path = sys.argv[1]
with open(run_path, 'r') as f:
    run_dict = json.loads(f.read())
out_dir = os.path.join(run_dict['analysis_directory'], 'clustering')

# Load results data
results = pd.read_csv(run_dict['analysis_path'])
results['UVM_ID'] = results['UVM_ID'].astype(str)
results = results.set_index(run_dict['id_field'])
clusters = pd.read_csv(os.path.join(run_dict['analysis_directory'], 'clustering', 'clustered_features.csv'))
clusters['UVM_ID'] = clusters['UVM_ID'].astype(str)
clusters = clusters.set_index(run_dict['id_field'])
subset = ['A', 'B', 'C', 'D', 'E', 'F']
results = results.join(clusters['cluster'], how='inner')

magnitudes = ['Q2', 'Q10', 'Q50', 'Q100']
durations = ['Short', 'Medium', 'Long']
mag_renames = {'Q2': '0.5AEP', 'Q10': '0.1AEP', 'Q50': '0.02AEP', 'Q100': '0.01AEP'}

def event_trends():
    # results = results[results['Q2_Medium_mass_conserve'] > 0.95]
    # results = results[results['Q100_Medium_mass_conserve'] > 0.95]
    # results = results[results['Q2_Medium_mass_conserve'] < 1.05]
    # results = results[results['Q100_Medium_mass_conserve'] < 1.05]
    pt_size = 7
    pt_alpha = 0.5
    scale_size = 8
    aspect = 9 / 13
    w = scale_size
    h = scale_size * aspect

    lowess = sm.nonparametric.lowess
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(w, h))
    mags = ['Q2', 'Q100']
    for m in mags:
        results[f'{m}_Medium_pct_attenuation_per_km'] = results[f'{m}_Medium_pct_attenuation'] * 100
    y1_max = max([results[f'{m}_Medium_pct_attenuation_per_km'].max() for m in mags])
    y2_max = max([results[f'{m}_Medium_cms_attenuation_per_km'].max() for m in mags])
    y1_min = 0
    y2_min = 0
    # math to make 0 line up between the two axes
    y1_range = y1_max - y1_min
    y2_range = y2_max - y2_min
    r1 = (y1_min / y1_range)
    r2 = (y2_min / y2_range)
    if abs(r1) > abs(r2):
        y2_min = (r1 * y2_max) / (1 + r1)
    else:
        y1_min = (r2 * y1_max) / (1 + r2)
    y1_range = y1_max - y1_min
    y2_range = y2_max - y2_min
    # add buffer
    y1_max += 0.1*y1_range
    y2_max += 0.1*y2_range
    y1_min -= 0.1*y1_range
    y2_min -= 0.1*y2_range

    for ind, m in enumerate(mags):
        # Panel 1:  slope vs attenuation
        x = results['slope']
        y1 = results[f'{m}_Medium_pct_attenuation_per_km']
        y2 = results[f'{m}_Medium_cms_attenuation_per_km']
        pct_label = axs[0, ind].scatter(x, y1, c=c1, s=pt_size, alpha=pt_alpha)
        # z = lowess(y1, x)
        # axs[0, ind].plot(z[:, 0], z[:, 1], c='b')
        ax_00_twin = axs[0, ind].twinx()
        cms_label = ax_00_twin.scatter(x, y2, c=c2, s=pt_size, alpha=pt_alpha)
        # z = lowess(y2, x)
        # ax_00_twin.plot(z[:, 0], z[:, 1], c='r')
        axs[0, ind].axvline(x=3e-3, c='k', ls='--', alpha=0.5)
        axs[0, ind].set(ylim=(y1_min, y1_max), title=mag_renames[m], xlabel='Slope (m/m)', xscale='log', facecolor=background_color)
        ax_00_twin.set(ylim=(y2_min, y2_max), facecolor=background_color)
        if ind == 0:
            axs[0, ind].set(ylabel='Attenuation per km (pct)')
        else:
            ax_00_twin.set(ylabel='Attenuation per km (cms)')

        # Panel 2:  DA vs attenuation
        x = results['DASqKm']
        axs[1, ind].scatter(x, y1, c=c1, s=pt_size, alpha=pt_alpha)
        # z = lowess(y1, x)
        # axs[1, ind].plot(z[:, 0], z[:, 1], c='b')
        ax_10_twin = axs[1, ind].twinx()
        ax_10_twin.scatter(x, y2, c=c2, s=pt_size, alpha=pt_alpha)
        # z = lowess(y2, x)
        # ax_10_twin.plot(z[:, 0], z[:, 1], c='r')
        axs[1, ind].set(ylim=(y1_min, y1_max), xlabel='Drainage Area (sqkm)', xscale='log', facecolor=background_color)
        ax_10_twin.set(ylim=(y2_min, y2_max), facecolor=background_color)
        if ind == 0:
            axs[1, ind].set(ylabel='Attenuation per km (pct)')
        else:
            ax_10_twin.set(ylabel='Attenuation per km (cms)')

    axs[0, 1].legend([pct_label, cms_label], ['pct', 'cms'], loc='upper right')
    fig.tight_layout()
    out_path = os.path.join(out_dir, 'Attenuation_twopanel.png')
    fig.savefig(out_path, dpi=400)

def celerity_trends():
    pt_alpha = 0.5
    pt_size = 1
    lowess_alpha = 0.95
    lowess_lw = 1
    
    results = pd.read_csv(r"G:\floodplainsData\runs\8\analysis\clustering\all_data.csv")
    results = results[results['Celerity (m/s)'] < 10]
    results = results[results['cluster'].isin(subset)]

    fig, axs = plt.subplots(ncols=2, figsize=(6.5, 3))
    axs[0].scatter(results['Slope'], results['Celerity (m/s)'], c=c1, s=pt_size, alpha=pt_alpha, zorder=1)
    twinx0 = axs[0].twinx()
    twinx0.scatter(results['Slope'], results['Shape Celerity (m^(2/3))'], c=c2, s=pt_size, alpha=pt_alpha, zorder=2)

    # # add lowess
    # x = np.log(results['Slope'])
    # y = np.log(results['Celerity (m/s)'])
    # z = lowess(y, x)
    # axs[0].plot(np.exp(z[:, 0]), np.exp(z[:, 1]), c='k', alpha=lowess_alpha, lw=lowess_lw, zorder=3, label='Series Lowess')
    # y = np.log(results['Shape Celerity (m^(2/3))'])
    # z = lowess(y, x)
    # twinx0.plot(np.exp(z[:, 0]), np.exp(z[:, 1]), c='k', alpha=lowess_alpha, lw=lowess_lw, zorder=4)

    # add linreg
    print('Slope regressions')
    x = np.log(results['Slope'].to_numpy())
    y = np.log(results['Celerity (m/s)'].to_numpy())
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    print(f'Celerity: {model.summary()}')
    y_pred = model.predict(x)
    axs[0].plot(np.exp(x[:, 1]), np.exp(y_pred), c='#3495ad', alpha=lowess_alpha, lw=lowess_lw, zorder=3, label='Series Lowess')
    y = np.log(results['Shape Celerity (m^(2/3))'].to_numpy())
    model = sm.OLS(y, x).fit()
    print(f'Shape Celerity: {model.summary()}')
    y_pred = model.predict(x)
    twinx0.plot(np.exp(x[:, 1]), np.exp(y_pred), c='#de8500', alpha=lowess_alpha, lw=lowess_lw, zorder=4)

    # Label
    axs[0].set(xlabel='Slope (m/m)', ylabel='Celerity (m/s)', facecolor=background_color, xscale='log', yscale='log')
    twinx0.set(yscale='log')
    axs[0].tick_params(labelsize=8)
    twinx0.tick_params(labelsize=8)

    y2 = 'Diagnostic Size'
    axs[1].scatter(results[y2], results['Celerity (m/s)'], c=c1, s=pt_size, alpha=pt_alpha, zorder=1)
    twinx1 = axs[1].twinx()
    twinx1.scatter(results[y2], results['Shape Celerity (m^(2/3))'], c=c2, s=pt_size, alpha=pt_alpha, zorder=2)
    axs[1].set(xlabel='Diagnostic Size (m)', facecolor=background_color, xscale='log', yscale='log')
    twinx1.set(yscale='log', ylabel=r'Shape Celerity (${m}^{2/3}$)')
    axs[1].tick_params(labelsize=8)
    twinx1.tick_params(labelsize=8)

    # # add lowess
    # x = np.log(results['Ave_Rh'])
    # y = np.log(results['Celerity (m/s)'])
    # z = lowess(y, x)
    # axs[1].plot(np.exp(z[:, 0]), np.exp(z[:, 1]), c='k', alpha=lowess_alpha, lw=lowess_lw, zorder=3)
    # y = np.log(results['Shape Celerity (m^(2/3))'])
    # z = lowess(y, x)
    # twinx1.plot(np.exp(z[:, 0]), np.exp(z[:, 1]), c='k', alpha=lowess_alpha, lw=lowess_lw, zorder=4)

    # add linreg
    print('Diagnostic Size regressions')
    x = np.log(results[y2].to_numpy())
    y = np.log(results['Celerity (m/s)'].to_numpy())
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    print(f'Celerity: {model.summary()}')
    y_pred = model.predict(x)
    axs[1].plot(np.exp(x[:, 1]), np.exp(y_pred), c='#3495ad', alpha=lowess_alpha, lw=lowess_lw, zorder=3)
    y = np.log(results['Shape Celerity (m^(2/3))'].to_numpy())
    model = sm.OLS(y, x).fit()
    print(f'Shape Celerity: {model.summary()}')
    y_pred = model.predict(x)
    twinx1.plot(np.exp(x[:, 1]), np.exp(y_pred), c='#ba7002', alpha=lowess_alpha, lw=lowess_lw, zorder=4)

    # Add custom legend
    custom_lines = [plt.Line2D([0], [0], color=c1, lw=0, marker='o', markersize=5),
                    plt.Line2D([0], [0], color=c2, lw=0, marker='o', markersize=5)]
    fig.legend(custom_lines, ['Celerity', 'Shape Celerity'], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.99), fancybox=False)

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig('Celerity Plots.png', dpi=400)
    # plt.show()

def cluster_routing():
    results = pd.read_csv(os.path.join(run_dict['analysis_directory'], 'clustering', 'all_data.csv'))

    colors = ["#8f00cc", "#cc0000", "#cc7000", "#cdbc00", "#07cc00", "#00cccc", '#2b54d8', '#979797']
    order = ['A', 'B', 'C', 'D', 'E', 'F', 'W', 'X']
    size = 6.5
    aspect = (9 / 6.5)
    w = size
    h = size * aspect
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(h, w))

    # Panel 1:  cluster vs celerity
    significance_scores = ['I', 'II', 'II', 'III', 'IV', 'IV', 'II', 'V']
    sns.boxplot(x='cluster', y='Celerity (m/s)', data=results, ax=axs[0, 0], palette=colors, order=order, flierprops={"marker": "o", 'markersize': 1})
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_ylim(axs[0, 0].get_ylim()[0], axs[0, 0].get_ylim()[1] * 2)
    y = axs[0, 0].get_ylim()[1] * 0.75
    for ind, label in enumerate(significance_scores):
        axs[0, 0].text(ind, y, label, ha='center', va='top', color='k', fontsize=10, zorder=100, path_effects=[pe.withStroke(linewidth=4, foreground="white")])
    axs[0, 0].set_facecolor(background_color)
    

    # Panel 2:  cluster vs shape celerity
    significance_scores = ['I', 'II', 'III', 'III,IV', 'V', 'II', 'III', 'IV']
    sns.boxplot(x='cluster', y='Shape Celerity (m^(2/3))', data=results, ax=axs[0, 1], palette=colors, order=order, flierprops={"marker": "o", 'markersize': 1})
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_ylim(axs[0, 1].get_ylim()[0], axs[0, 1].get_ylim()[1] * 2)
    y = axs[0, 1].get_ylim()[1] * 0.75
    for ind, label in enumerate(significance_scores):
        axs[0, 1].text(ind, y, label, ha='center', va='top', color='k', fontsize=10, zorder=100, path_effects=[pe.withStroke(linewidth=4, foreground="white")])
    axs[0, 1].set_ylabel(r'Shape Celerity $({m}^{2/3})$')
    axs[0, 1].set_facecolor(background_color)
    

    # Panel 3:  event vs attenuation cms
    for mag in magnitudes:
        results[f'{mag}_Medium_pct_attenuation_per_km'] = results[f'{mag}_Medium_pct_attenuation'] * 100
        results[f'{mag}_Medium_pct_attenuation_per_km'] = results[f'{mag}_Medium_pct_attenuation_per_km'].clip(-1, 100)
        results[f'{mag}_Medium_cms_attenuation_per_km'] = results[f'{mag}_Medium_cms_attenuation_per_km'].clip(-1, 100)
    pct_melted = results.melt(id_vars=['cluster'], value_vars=[f'{m}_Medium_pct_attenuation_per_km' for m in magnitudes], value_name='Attenuation per km (pct)')
    pct_melted['ri'] = pct_melted['variable'].map(lambda x: mag_renames[x.split('_')[0]])
    pct_melted = pct_melted.sort_values(by='cluster')
    sns.boxplot(x='ri', y='Attenuation per km (pct)', data=pct_melted, ax=axs[1, 0], hue='cluster', palette=colors, flierprops={"marker": "o", 'markersize': 0.5})
    axs[1, 0].set_ylim(axs[1, 0].get_ylim()[0], axs[1, 0].get_ylim()[1] * 1.05)
    y = axs[1, 0].get_ylim()[1] * 0.95
    significance_scores = [['I', 'I', 'I', 'II', 'II', 'III', 'I', 'IV'], 
                           ['I', 'I', 'I', 'II', 'II', 'III', 'I', 'IV'],
                           ['I', 'I', 'I', 'II', 'III', 'IV', 'I', 'V'],
                           ['I', 'I', 'I', 'II', 'III', 'IV', 'I', 'V']]
    width = (axs[1, 0].get_xlim()[1] / 4) / 9
    for ind, label in enumerate(significance_scores):
        for i2, l in enumerate(label):
            x = ind + (i2 - 3.5) * width
            print(x)
            axs[1, 0].text(x, y, l, ha='center', va='top', color='k', fontsize=7, zorder=100, path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    axs[1, 0].get_legend().remove()
    axs[1, 0].set_xlabel(None)
    axs[1, 0].set_facecolor(background_color)


    # Panel 4:  event vs attenuation percent
    cms_melted = results.melt(id_vars=['cluster'], value_vars=[f'{m}_Medium_cms_attenuation_per_km' for m in magnitudes], value_name='Attenuation per km (cms)')
    cms_melted['ri'] = cms_melted['variable'].map(lambda x: mag_renames[x.split('_')[0]])
    cms_melted = cms_melted.sort_values(by='cluster')
    sns.boxplot(x='ri', y='Attenuation per km (cms)', data=cms_melted, ax=axs[1, 1], hue='cluster', palette=colors, flierprops={"marker": "o", 'markersize': 0.5})
    axs[1, 1].get_legend().remove()
    axs[1, 1].set_xlabel(None)
    axs[1, 1].set_facecolor(background_color)


    fig.tight_layout()
    out_path = os.path.join(out_dir, 'cluster_routing.png')
    fig.savefig(out_path, dpi=400)


# event_trends()
# celerity_trends()
cluster_routing()