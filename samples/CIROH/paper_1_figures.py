import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb, to_rgb
import seaborn as sns
import itertools as it
import statsmodels.api as sm
from scipy.interpolate import splrep, BSpline
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d


# Set Nimbus Sans or a similar font as the default font
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Nimbus Sans'


c_dict = {'orange': '#fdc161',
          'purple': '#c1bae0',
          'green': '#badeab',
          'yellow': '#fbf493', 
          'blue': '#03bde3', 
          'pink': '#fbabb7'}


def diffusion_vs_attenuation(data_path, norm_length=False):
    # Load data
    with open(data_path, 'r') as f:
        run_dict = json.loads(f.read())
    in_data = pd.read_csv(run_dict['analysis_path'])

    
    ris = ['Q2', 'Q10', 'Q50', 'Q100']
    durs = ['Short', 'Medium', 'Long']
    for ri in ris:
        for dur in durs:
            event = '_'.join([ri, dur])
            diff_column = '_'.join([event, 'diffusion_number'])
            slope_column = 'slope'
            if norm_length:
                att_column = '_'.join([event, 'pct_attenuation_per_km'])
            else:
                att_column = '_'.join([event, 'pct_attenuation'])
            in_data = in_data.sort_values(by=diff_column)
            diff_nums = in_data[diff_column]
            slopes = in_data[slope_column]
            y = in_data[att_column].to_numpy()
            y = np.nan_to_num(y)
            y[y < 0] = 0
            y[y > 1] = 1
            y *= 100
            y = y[~np.isnan(diff_nums.to_numpy())]
            slopes = slopes[~np.isnan(diff_nums.to_numpy())]
            diff_nums = diff_nums.to_numpy()[~np.isnan(diff_nums.to_numpy())]
            if not norm_length:
                tck = splrep(np.log10(diff_nums), y, task=-1, t=[-2, 0.2, 0.9])  # used to be [-2, 0.5, 1]
                interp_x = np.logspace(np.log10(diff_nums.min()), np.log10(diff_nums.max()), 250)
                interp_y = BSpline(*tck)(np.log10(interp_x))

            sorted_y = np.sort(y)
            cum_pct_less = (np.arange(len(sorted_y)) / len(sorted_y)) * 100

            fig, (diff_ax, slope_ax, hist_ax) = plt.subplots(ncols=3, sharey=True, figsize=(6.5, 2))
            color = 'blue'
            diff_ax.scatter(diff_nums, y, c=c_dict[color], alpha=0.45, s=3, lw=0.1)
            if not norm_length:
                diff_ax.plot(interp_x, interp_y, c='k', ls='dashed', alpha=0.5, lw=0.5)

            slope_ax.scatter(slopes, y, c=c_dict[color], alpha=0.45, s=3, lw=0.1)

            hist_ax.plot(cum_pct_less, sorted_y, c=c_dict[color], lw=2)
            if norm_length:
                p90 = int(np.interp(90, cum_pct_less, sorted_y))
                hist_ax.text(60, 20, f'90% of reaches\nhave less than\n{p90}% attenuation', horizontalalignment='right', verticalalignment='center', fontsize=3)
                hist_ax.arrow(62, 20, 26, (p90 - 18), fc='k', ec='k', lw=0.5, head_width=1, head_length=1)
            else:
                p80 = int(np.interp(80, cum_pct_less, sorted_y))
                hist_ax.text(60, 20, f'80% of reaches\nhave less than\n{p80}% attenuation', horizontalalignment='right', verticalalignment='center', fontsize=3)
                hist_ax.arrow(62, 20, 16, (p80 - 18), fc='k', ec='k', lw=0.5, head_width=1, head_length=1)

            diff_ax.set_xscale('log')
            slope_ax.set_xscale('log')
            diff_ax.set_facecolor("#ededed")
            slope_ax.set_facecolor("#ededed")
            hist_ax.set_facecolor("#ededed")

            diff_ax.grid(c='w', lw=0.4)
            slope_ax.grid(c='w', lw=0.4)
            hist_ax.grid(c='w', lw=0.4)

            diff_ax.tick_params(axis='both', labelsize=5)
            slope_ax.tick_params(axis='x', labelsize=5)
            hist_ax.tick_params(axis='x', labelsize=5)

            if norm_length:
                diff_ax.set_ylabel('% Attenuation Per km', fontsize=6)
            else:
                diff_ax.set_ylabel('% Attenuation After One Time-to-Rise', fontsize=6)
            diff_ax.set_xlabel('Diffusion-Frequency #', fontsize=6)
            slope_ax.set_xlabel('Slope (m/m)', fontsize=6)
            hist_ax.set_xlabel('% of Reaches With Less Attenuation', fontsize=6)

            diff_ax.set_axisbelow(True)
            slope_ax.set_axisbelow(True)
            hist_ax.set_axisbelow(True)

            if norm_length:
                diff_ax.set_ylim(-1, 50)
            else:
                diff_ax.set_ylim(-1, 70)
            
            diff_ax.set_xlim(10 ** -6, 400)
            slope_ax.set_xlim(8.5 * (10 ** -6), 1)

            for ax in (diff_ax, slope_ax, hist_ax):
                for axis in ['top','bottom','left','right']:
                    ax.spines[axis].set_linewidth(0.1)

                # increase tick width
                ax.tick_params(width=0.1, which='both')

            # plt.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(run_dict['analysis_directory'], f'att_summary_{ri}_{dur}.png'), dpi=450)
            plt.close()

def attenuation_densities(data_path, norm_length=False):
    def scale_vibrance(color, factor=0.5):
        h, s, v = rgb_to_hsv(to_rgb(color))
        new_v = min([factor * v, 1])
        return hsv_to_rgb((h, s, new_v))
    
    # Load data
    with open(data_path, 'r') as f:
        run_dict = json.loads(f.read())
    in_data = pd.read_csv(run_dict['analysis_path'])
    
    ris = ['Q2', 'Q10', 'Q50', 'Q100']
    durs = ['Short', 'Medium', 'Long']
    if norm_length:
        att_cols = [f'{i[0]}_{i[1]}_pct_attenuation_per_km' for i in it.product(ris, durs)]
    else:
        att_cols = [f'{i[0]}_{i[1]}_pct_attenuation' for i in it.product(ris, durs)]
    df = pd.melt(in_data, value_vars=att_cols)
    df[['ri','dur']] = df['variable'].str.split('_',expand=True)[[0,1]]
    df['value'][df['value'] < 0] = 0
    df['value'][df['value'] > 1] = 1
    df['value'] *= 100

    sns.set_palette(sns.color_palette(c_dict.values()))
    plt.figure(figsize=(6.5,2.5))
    flierprops={'marker': 'o', 'markersize': 1.5, 'alpha':0.8, 'markeredgewidth':0.1}
    ax = sns.boxplot(df, y='ri', hue='dur', x='value', flierprops=flierprops)
        
    patches = [i for i in ax.patches if i.get_label() == '']
    counter = 0
    for i, box in enumerate(ax.patches):
        color = box.get_facecolor()
        color = scale_vibrance(color, 1.1)
        dark_col = scale_vibrance(color, 0.75)
        box.set_facecolor(color)
        box.set_edgecolor(dark_col)
        box.set_linewidth(0.5)
        if box.get_label() == '':
            for j in range(counter*6,counter*6+6):
                line = ax.lines[j]
                line.set_color(dark_col)
                line.set_mfc(color)
                line.set_mec(dark_col)
                line.set_linewidth(0.5)
                if line.get_marker() != '':
                    ys = line.get_ydata()
                    ys += np.random.uniform(-0.05, 0.05, len(ys))
                    line.set_ydata(ys)
            counter += 1
    fs = 5
    if norm_length:
       ax.set_xlabel('Attenuation After One Kilometer', fontsize=fs*1.1)
    #    ax.set_xlim(0, 20)
    else:
        ax.set_xlabel('Attenuation After One Time-To-Rise', fontsize=fs*1.1)
    
    ax.set_ylabel(None)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width, box.height * 0.95])
    ax.legend(title=None, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=fs)
    ax.set_xticklabels([f'{int(i)}%' for i in ax.get_xticks()], fontsize=fs)
    ax.set_yticklabels([i.get_text() for i in ax.get_yticklabels()], fontsize=fs)

    ax.set_facecolor("#ededed")
    ax.grid(c='w', lw=0.4)
    ax.set_axisbelow(True)

    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(os.path.join(run_dict['analysis_directory'], f'boxplot_attenuation.png'), dpi=600)


def cluster_map(data_path):
    # Load data
    with open(data_path, 'r') as f:
        run_dict = json.loads(f.read())
    in_data = pd.read_csv(run_dict['analysis_path'])
    used_cols = ['ave_rhp', 'stdev_rhp', 'cumulative_volume', 'cumulative_height', 'valley_confinement', 'el_bathymetry', 'el_edap', 'el_min', 'el_edep', 'el_bathymetry_scaled', 'el_edap_scaled', 'el_min_scaled', 'el_edep_scaled', 'height', 'height_scaled', 'vol', 'vol_scaled', 'min_rhp', 'slope_start_min', 'slope_min_stop', 'rh_bottom', 'rh_edap', 'rh_min', 'rh_edep', 'w_bottom', 'w_edap', 'w_min', 'w_edep']
    used_cols = ['ave_rhp', 'stdev_rhp', 'cumulative_volume', 'cumulative_height', 'valley_confinement', 'el_bathymetry', 'el_edap', 'el_min', 'el_edep', 'el_bathymetry_scaled', 'el_edap_scaled', 'el_min_scaled', 'el_edep_scaled', 'height', 'height_scaled', 'vol', 'vol_scaled', 'min_rhp', 'slope_start_min', 'slope_min_stop', 'rh_edap', 'rh_min', 'rh_edep', 'w_bottom', 'w_edap', 'w_min', 'w_edep']
    df = in_data.loc[:, used_cols]
    df = df.dropna(axis=0)

    # Draw the full plot
    g = sns.clustermap(df.corr(), center=0, cmap="vlag", dendrogram_ratio=(.1, .2), linewidths=.75, figsize=(12, 13))
    g.ax_row_dendrogram.remove()
    fig = g.figure

    fig.savefig(os.path.join(run_dict['analysis_directory'], 'clustermap.jpg'), dpi=300)

def att_vs_da(data_path, norm_length):
    # Load data
    with open(data_path, 'r') as f:
        run_dict = json.loads(f.read())
    in_data = pd.read_csv(run_dict['analysis_path'])

    in_data['has_edz'] = (in_data['cumulative_volume'] > 0)
    if norm_length:
        y_col = 'Q2_Short_pct_attenuation_per_km'
    else:
        y_col = 'Q2_Short_pct_attenuation'
    in_data[y_col][in_data[y_col] < 0] = in_data[y_col][in_data[y_col] > 0].min()

    fig, ax = plt.subplots()
    sns.scatterplot(in_data, x='DASqKm', y=y_col, ax=ax, hue='has_edz', palette='bright')
    ax.set(xscale='log')
    fig.savefig(os.path.join(run_dict['analysis_directory'], 'da_vs_att.jpg'), dpi=300)

def edz_setting_da(data_path):
    # Load data
    with open(data_path, 'r') as f:
        run_dict = json.loads(f.read())
    in_data = pd.read_csv(run_dict['analysis_path'])

    in_data['has_edz'] = (in_data['cumulative_volume'] > 0)
    bins = np.linspace(0.6, 3.4, 15)
    bins = np.round(bins, 1)
    labels = [f'e{bins[i]}-e{bins[i + 1]}' for i in range(len(bins) - 1)]
    bins = 10 ** bins
    in_data['DASqKm'][in_data['DASqKm'] > bins.max()] = bins.max()  # artificial cutoff
    in_data['DASqKm'] = pd.cut(in_data['DASqKm'], bins=bins, labels=labels)

    fig, (hist, dist) = plt.subplots(nrows=2, figsize=(4, 5.5), sharex=True)
    sns.histplot(data=in_data, x="DASqKm", hue="has_edz", multiple="fill", stat="proportion", discrete=True, shrink=.8, ax=dist)
    sns.histplot(data=in_data, x="DASqKm", discrete=True, shrink=.8, ax=hist, fc='darkgray')
    dist.set_xticks(dist.get_xticks(), labels, rotation=90, ha='right')
    fig.tight_layout()
    fig.savefig(os.path.join(run_dict['analysis_directory'], 'edz_proportions.jpg'), dpi=300)


def boxplots(run_path):
    # Load data
    with open(run_path, 'r') as f:
        run_dict = json.loads(f.read())
    in_path = os.path.join(run_dict['analysis_directory'], 'fit_celerity_rmse.csv')
    in_data = pd.read_csv(in_path)
    labels = ['Fitted\nSigmoidal Channel', 'Fitted\nTwo-stage Channel', 'Regression\nTwo-stage Channel']
    in_data = in_data.rename(columns={'sig_rmse': labels[0], 'nwm_rmse': labels[1], 'nwm_reg_rmse': labels[2]})
    in_data = pd.melt(in_data, id_vars=['ReachCode'], value_vars=labels)

    my_pal = {labels[0]: c_dict['green'], labels[1]: c_dict['purple'], labels[2]:c_dict['orange']}

    fig, ax = plt.subplots()
    sns.boxplot(in_data, x='variable', y='value', palette=my_pal, ax=ax, showfliers=False, linewidth=1)
    for artist in ax.artists:
        artist.set_edgecolor('black')
    ax.set(xlabel=None, ylabel='Celerity RMSE (m/s)')
    ax.set_facecolor("#ededed")
    ax.grid(c='w', lw=0.4)
    ax.set_axisbelow(True)
    plt.legend([],[], frameon=False)
    fig.savefig(r'/netfiles/ciroh/floodplainsData/runs/6/working/sig_fits.png', dpi=300)
    

def param_scaling(run_path):
    # Load data
    with open(run_path, 'r') as f:
        run_dict = json.loads(f.read())
    in_path = os.path.join(run_dict['analysis_directory'], 'fitted_curves.csv')
    in_data = pd.read_csv(in_path)
    reach_data = pd.read_csv(run_dict['reach_meta_path'])
    in_data = in_data.merge(reach_data, how='left', on='ReachCode')

    # power law fitting
    x = np.log(in_data['TotDASqKm'])
    x = np.vstack([x, np.ones(len(x))]).T
    pred_space = np.array([in_data['TotDASqKm'].min(), in_data['TotDASqKm'].max()])

    y = np.log(in_data['sig_ch_w'])
    m_ch_w, b_ch_w = np.linalg.lstsq(x, y, rcond=None)[0]
    b_ch_w = np.exp(b_ch_w)
    ch_w_pred = b_ch_w * (pred_space ** m_ch_w)
    text_ch_w = '$w={' + str(round(b_ch_w, 3)) + 'DA' + '}^{' + str(round(m_ch_w, 3)) + '}$'

    y = in_data['sig_fp_w'].to_numpy()
    y[y == 0] = 1
    y = np.log(y)
    m_fp_w, b_fp_w = np.linalg.lstsq(x, y, rcond=None)[0]
    b_fp_w = np.exp(b_fp_w)
    fp_w_pred = b_fp_w * (pred_space ** m_fp_w)
    text_fp_w = '$w={' + str(round(b_fp_w, 3)) + 'DA' + '}^{' + str(round(m_fp_w, 3)) + '}$'

    y = np.log(in_data['sig_bkf_el'])
    m_bkf, b_bkf = np.linalg.lstsq(x, y, rcond=None)[0]
    b_bkf = np.exp(b_bkf)
    bkf_pred = b_bkf * (pred_space ** m_bkf)
    text_bkf = '$bkf={' + str(round(b_bkf, 3)) + 'DA' + '}^{' + str(round(m_bkf, 3)) + '}$'

    y = np.log(in_data['sig_fp_s'])
    m_fp_s, b_fp_s = np.linalg.lstsq(x, y, rcond=None)[0]
    b_fp_s = np.exp(b_fp_s)
    fp_s_pred = b_fp_s * (pred_space ** m_fp_s)
    text_fp_s = '$s={' + str(round(b_fp_s, 3)) + 'DA' + '}^{' + str(round(m_fp_s, 3)) + '}$'

    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(6.5, 9))
    axs[0, 0].scatter(in_data['TotDASqKm'], in_data['sig_ch_w'], s=1, fc='none', ec=c_dict['blue'], alpha=0.5)
    axs[1, 0].scatter(in_data['TotDASqKm'], in_data['sig_fp_w'], s=1, fc='none', ec=c_dict['blue'], alpha=0.5)
    axs[2, 0].scatter(in_data['TotDASqKm'], in_data['sig_bkf_el'], s=1, fc='none', ec=c_dict['blue'], alpha=0.5)
    axs[3, 0].scatter(in_data['TotDASqKm'], in_data['sig_fp_s'], s=1, fc='none', ec=c_dict['blue'], alpha=0.5)

    axs[0, 0].plot(pred_space, ch_w_pred, c='k', ls='dashed', lw=1, alpha=0.5)
    axs[1, 0].plot(pred_space, fp_w_pred, c='k', ls='dashed', lw=1, alpha=0.5)
    axs[2, 0].plot(pred_space, bkf_pred, c='k', ls='dashed', lw=1, alpha=0.5)
    axs[3, 0].plot(pred_space, fp_s_pred, c='k', ls='dashed', lw=1, alpha=0.5)

    axs[0, 0].text(0.95, 0.05, text_ch_w, transform=axs[0, 0].transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    axs[1, 0].text(0.95, 0.05, text_fp_w, transform=axs[1, 0].transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    axs[2, 0].text(0.95, 0.05, text_bkf, transform=axs[2, 0].transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    axs[3, 0].text(0.05, 0.05, text_fp_s, transform=axs[3, 0].transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='left')

    axs[0, 0].set(xscale='log', yscale='log', ylabel='Channel Width (m)')
    axs[1, 0].set(xscale='log', yscale='log', ylabel='Floodplain Width (m)')
    axs[2, 0].set(xscale='log', yscale='log', ylabel='Bankfull Stage (m)')
    axs[3, 0].set(xscale='log', yscale='log', ylabel='Floodplain Slope (larger=flatter)', xlabel=r'Drainage area (${km}^{2}$)')

    axs[0, 1].scatter(in_data['slope'], in_data['sig_ch_w'], s=1, fc='none', ec=c_dict['blue'], alpha=0.5)
    axs[1, 1].scatter(in_data['slope'], in_data['sig_fp_w'], s=1, fc='none', ec=c_dict['blue'], alpha=0.5)
    axs[2, 1].scatter(in_data['slope'], in_data['sig_bkf_el'], s=1, fc='none', ec=c_dict['blue'], alpha=0.5)
    axs[3, 1].scatter(in_data['slope'], in_data['sig_fp_s'], s=1, fc='none', ec=c_dict['blue'], alpha=0.5)

    axs[0, 1].set(xscale='log', yscale='log', ylabel='Channel Width (m)')
    axs[1, 1].set(xscale='log', yscale='log', ylabel='Floodplain Width (m)')
    axs[2, 1].set(xscale='log', yscale='log', ylabel='Bankfull Stage (m)')
    axs[3, 1].set(xscale='log', yscale='log', ylabel='Floodplain Slope (larger=flatter)', xlabel='slope (m/m)')

    for r in axs:
        for ax in r:
            ax.set_facecolor("#ededed")

    fig.tight_layout()
    fig.savefig(r'/netfiles/ciroh/floodplainsData/runs/6/working/scaling.png', dpi=300)

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
    tmp_el = el_data[reach].to_numpy()
    y = el_scaled_data[reach].to_numpy()
    rh = rh_data[reach].to_numpy()
    x = rh_prime_data[reach].to_numpy()
    tmp_area = area_data[reach].to_numpy() / tmp_meta['length']
    edep = tmp_meta['el_edep_scaled']
    edap = tmp_meta['el_edap_scaled']

    fig, ax = plt.subplots(figsize=(8.5, 6.25))
    width = tmp_area / 2
    width = np.append(-width[::-1], width)
    width = width - min(width)
    section_el = np.append(y[::-1], y)
    ax.plot(width, section_el, c='k', lw=3)
    ax.fill_between([min(width), max(width)], [edep, edep], [edap, edap], fc='lightblue', alpha=0.9)
    ax.set(xlim=(min(width), max(width)), ylim=(0, 5), xlabel='Station (m)', ylabel='Stage / Bankfull Depth')
    ax2 = ax.twinx()
    ax2.set_ylim(0, tmp_el.max())
    ax2.set_ylabel(' ', fontsize=28)
    ax.set_xlabel(' ', fontsize=28)
    ax.set_ylabel(' ', fontsize=28)
    ax.grid(alpha=0.5)
    ax.set_facecolor("#ededed")
    fig.tight_layout()
    fig.savefig(r'/netfiles/ciroh/floodplainsData/runs/6/working/paper/working_section.png', dpi=300)

    fig, ax = plt.subplots(figsize=(8.5, 6.25))
    ax.plot(rh, y, c='k', lw=3)
    ax.fill_between([min(rh), max(rh)], [edep, edep], [edap, edap], fc='lightblue', alpha=0.9)
    ax.set(xlim=(min(rh), max(rh)), ylim=(0, 5), xlabel=' ', ylabel='Stage / Bankfull Depth')
    ax2 = ax.twinx()
    ax2.set_ylim(0, tmp_el.max())
    ax2.set_ylabel(' ', fontsize=28)
    ax.set_xlabel(' ', fontsize=28)
    ax.set_ylabel(' ', fontsize=28)
    ax.grid(alpha=0.5)
    ax.set_facecolor("#ededed")
    fig.tight_layout()
    fig.savefig(r'/netfiles/ciroh/floodplainsData/runs/6/working/paper/working_rh.png', dpi=300)

    y_cross = np.argmax(y > edap)
    y_cross_2 = np.argmax(y[y_cross:] > edep) + y_cross

    fig, ax = plt.subplots(figsize=(8.5, 6.25))
    ax.plot(x, y, c='k', lw=3)
    ax.axvline(0.5, ls='dashed', c='k', alpha=0.7)
    ax.fill_between([-1, 1], [edep, edep], [edap, edap], fc='lightblue', alpha=0.9)
    ax.fill_betweenx(y[y_cross:y_cross_2], 0.5, x[y_cross:y_cross_2])
    ax.set(xlim=(-1, 1), ylim=(0, 5))
    ax2 = ax.twinx()
    ax2.set_ylim(0, tmp_el.max())
    ax2.set_ylabel(' ', fontsize=28)
    ax.set_xlabel(' ', fontsize=28)
    ax.set_ylabel(' ', fontsize=28)
    ax.grid(alpha=0.5)
    ax.set_facecolor("#ededed")
    fig.tight_layout()
    fig.savefig(r'/netfiles/ciroh/floodplainsData/runs/6/working/paper/working_rhp.png', dpi=300)


def diffusion_vs_attenuation_2(data_path, norm_length=False):
    # Load data
    with open(data_path, 'r') as f:
        run_dict = json.loads(f.read())
    in_data = pd.read_csv(run_dict['analysis_path'])
    in_data = in_data.set_index(run_dict['id_field'])
    clusters = pd.read_csv(os.path.join(run_dict['analysis_directory'], 'clustering', 'clustered_features.csv'))
    clusters = clusters.set_index(run_dict['id_field'])
    in_data = in_data.join(clusters, how='inner')

    event = 'Q100_Medium'
    diff_column = '_'.join([event, 'diffusion_number'])
    if norm_length:
        att_column = '_'.join([event, 'pct_attenuation_per_km'])
    else:
        att_column = '_'.join([event, 'pct_attenuation'])
    
    vals = in_data[[diff_column, att_column, 'cluster']].dropna(axis=0)
    vals[att_column] = vals[att_column].clip(0, 1) * 100
    colors = ["#8f00cc", "#cc0000", "#cc7000", "#cdbc00", "#07cc00", "#00cccc", '#2b54d8', '#979797']
    clusters = sorted(vals['cluster'].unique())
    c_dict = {k: v for k, v in zip(clusters, colors)}
    vals['cluster'] = vals['cluster'].map(c_dict)
    
    fig, ax = plt.subplots(figsize=(3.5, 2))
    ax.scatter(vals[diff_column], vals[att_column], c=vals['cluster'], alpha=0.45, s=3, lw=0.1)
    
    ax.set_xscale('log')
    ax.set_facecolor("#ededed")

    ax.grid(c='w', lw=0.4)

    ax.tick_params(axis='both', labelsize=5)

    if norm_length:
        ax.set_ylabel('% Attenuation Per km', fontsize=6)
    else:
        ax.set_ylabel('% Attenuation After One Time-to-Rise', fontsize=6)
    ax.set_xlabel('Diffusion-Frequency #', fontsize=6)

    ax.set_axisbelow(True)

    if norm_length:
        ax.set_ylim(-1, 50)
    else:
        ax.set_ylim(-1, 70)
    
    ax.set_xlim(10 ** -6, 400)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.1)

    # increase tick width
    ax.tick_params(width=0.1, which='both')

    # plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(run_dict['analysis_directory'],'clustering', f'att_cluster_summary_{event}.png'), dpi=450)
    plt.close()


if __name__ == '__main__':
    run_path = r'/netfiles/ciroh/floodplainsData/runs/8/run_metadata.json'
    # diffusion_vs_attenuation(run_path, norm_length=False)
    attenuation_densities(run_path, norm_length=True)
    # att_vs_da(run_path, norm_length=False)
    # edz_setting_da(run_path)
    # cluster_map(run_path)
    # boxplots(run_path)
    # param_scaling(run_path)
    # edz_conceptual_figure(run_path)
    diffusion_vs_attenuation_2(run_path)