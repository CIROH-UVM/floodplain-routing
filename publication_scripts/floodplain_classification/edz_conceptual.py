import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def edz_conceptual_figure(run_path):
    # Load data
    with open(run_path, 'r') as f:
        run_dict = json.loads(f.read())
    working_dir = run_dict['geometry_directory']
    reach_path = run_dict['analysis_path']
    el_path = os.path.join(working_dir, 'el.csv')
    el_scaled_path = os.path.join(working_dir, 'el_scaled.csv')
    rh_path = os.path.join(working_dir, 'rh.csv')
    rh_prime_path = os.path.join(working_dir, 'rh_prime.csv')
    area_path = os.path.join(working_dir, 'area.csv')

    reach_data = pd.read_csv(reach_path)
    reach_data = reach_data.dropna(axis=0)
    reach_data['ReachCode'] = reach_data['ReachCode'].astype(np.int64).astype(str)
    reach_data = reach_data.set_index('ReachCode')

    el_data = pd.read_csv(el_path)

    el_scaled_data = pd.read_csv(el_scaled_path)

    rh_data = pd.read_csv(rh_path)
    rh_data = rh_data.dropna(axis=1)

    rh_prime_data = pd.read_csv(rh_prime_path)
    # Clean Rh prime
    rh_prime_data.iloc[-1] = rh_prime_data.iloc[-2]
    rh_prime_data[:] = gaussian_filter1d(rh_prime_data.T, 15).T
    rh_prime_data[rh_prime_data < -3] = -3
    rh_prime_data = rh_prime_data.dropna(axis=1)

    area_data = pd.read_csv(area_path)
    area_data = area_data.dropna(axis=1)
    
    reach = '4300103000939'
    tmp_meta = reach_data.loc[reach]
    tmp_el = el_data.to_numpy()[:, 0]
    y = el_scaled_data.to_numpy()[:, 0]
    rh = rh_data.to_numpy()[:, 0]
    x = rh_prime_data.to_numpy()[:, 0]
    tmp_area = area_data.to_numpy()[:, 0] / tmp_meta['length']
    edep = tmp_meta['el_edep_scaled']
    edap = tmp_meta['el_edap_scaled']

    fs = 0
    afs = 28
    fig, ax = plt.subplots(figsize=(8.5, 6.25))
    width = tmp_area / 2
    width = np.append(-width[::-1], width)
    width = width - min(width)
    section_el = np.append(y[::-1], y)
    ax.plot(width, section_el, c='k', lw=3)
    ax.fill_between([min(width), max(width)], [edep, edep], [edap, edap], fc='lightblue', alpha=0.9)
    ax.set(xlim=(min(width), max(width)), ylim=(0, 5))
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=afs)
    xticks = np.arange(0, 500, 100)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=afs)
    ax.grid(alpha=0.5)
    ax.set_facecolor("#ededed")
    fig.tight_layout()
    fig.savefig(r'/netfiles/ciroh/floodplainsData/runs/mock/working_section.png', dpi=300)
    fig.savefig(r'/netfiles/ciroh/floodplainsData/runs/mock/working_section.eps', dpi=300, format='eps')
    fig.savefig(r'/netfiles/ciroh/floodplainsData/runs/mock/working_section.pdf', dpi=300, format='pdf')

    fig, ax = plt.subplots(figsize=(8.5, 6.25))
    ax.plot(rh, y, c='k', lw=3)
    ax.fill_between([min(rh), max(rh)], [edep, edep], [edap, edap], fc='lightblue', alpha=0.9)
    ax.set(xlim=(min(rh), 8), ylim=(0, 5))
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=afs, color='w')
    xticks = np.arange(0, 10, 2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=afs)
    ax.grid(alpha=0.5)
    ax.set_facecolor("#ededed")
    fig.tight_layout()
    fig.savefig(r'/netfiles/ciroh/floodplainsData/runs/mock/working_rh.png', dpi=300)
    fig.savefig(r'/netfiles/ciroh/floodplainsData/runs/mock/working_rh.eps', dpi=300, format='eps')
    fig.savefig(r'/netfiles/ciroh/floodplainsData/runs/mock/working_rh.pdf', dpi=300, format='pdf')

    y_cross = np.argmax(y > edap)
    y_cross_2 = np.argmax(y[y_cross:] > edep) + y_cross

    fig, ax = plt.subplots(figsize=(8.5, 6.25))
    ax.plot(x, y, c='k', lw=3)
    ax.axvline(0.5, ls='dashed', c='k', alpha=0.7)
    ax.fill_between([-1, 1], [edep, edep], [edap, edap], fc='lightblue', alpha=0.9)
    ax.fill_betweenx(y[y_cross:y_cross_2], 0.5, x[y_cross:y_cross_2])
    ax.set(xlim=(-1, 1), ylim=(0, 5))
    yticks = np.arange(0, 6, 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels([])
    ax2 = ax.twinx()
    ax2.set_ylim(0, tmp_el.max())
    yticks = np.arange(2, 16, 2)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(yticks, fontsize=afs)
    xticks = [round(i, 1) for i in np.arange(-1, 1.5, 0.5)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=afs)
    ax.grid(alpha=0.5)
    ax.set_facecolor("#ededed")
    fig.tight_layout()
    fig.savefig(r'/netfiles/ciroh/floodplainsData/runs/mock/working_rhp.png', dpi=300)
    fig.savefig(r'/netfiles/ciroh/floodplainsData/runs/mock/working_rhp.eps', dpi=300, format='eps')
    fig.savefig(r'/netfiles/ciroh/floodplainsData/runs/mock/working_rhp.pdf', dpi=300, format='pdf')



if __name__ == '__main__':
    run_path = r'/netfiles/ciroh/floodplainsData/runs/mock/run_metadata.json'
    edz_conceptual_figure(run_path)