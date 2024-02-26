import numpy as np
import pandas as pd
import geopandas as gpd
import os
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import json

# Load regressions
with open('source/regressions.json') as in_file:
    REGRESSIONS = json.load(in_file)

FEATURE_NAMES = ['ReachCode', 'ave_rhp', 'el_bathymetry', 'el_edap', 'el_min', 'el_edep', 'el_bathymetry_scaled', 'el_edap_scaled', 'el_min_scaled', 'el_edep_scaled', 'height', 'height_scaled', 'vol', 'vol_scaled', 'min_rhp', 'slope_start_min', 'slope_min_stop', 'rh_bottom', 'rh_edap', 'rh_min', 'rh_edep', 'w_bottom', 'w_edap', 'w_min', 'w_edep', 'invalid_geometry']
ERROR_ARRAY = [np.nan for i in FEATURE_NAMES]

class ReachPlot:

    def __init__(self, out_dir, reach) -> None:
        self.reach = reach
        self.out_path = os.path.join(out_dir, f'{reach}.png')
        self.fig, (self.section_ax, self.rh_ax, self.rhp_ax) = plt.subplots(ncols=3, figsize=(10, 3), sharey=True)
        self.all_ax = (self.section_ax, self.rh_ax, self.rhp_ax)

        # Add labels
        self.section_ax.set_xlabel('Station (m)', fontsize=10)
        self.section_ax.set_ylabel('Stage / Bankfull Depth', fontsize=10)
        self.rh_ax.set_xlabel(r'$R_{h}$', fontsize=10)
        self.rhp_ax.set_xlabel(r"$R_{h}$'", fontsize=10)

    def no_geometry(self):
        for ax in self.all_ax:
            ax.text(0.5, 0.5, 'NO GEOMETRY', fontsize=10, verticalalignment='center', horizontalalignment='center')
        self.save()
    
    def add_geometry(self, el, width, rh, rhp, ave):
        # convert semi-cross-section to pseudo-cross-section
        width = width / 2
        width = np.append(-width[::-1], width)
        width = width - min(width)
        section_el = np.append(el[::-1], el)

        # Cache geometry
        self.el = el
        self.width = width
        self.rh = rh
        self.rhp = rhp

        # Plot
        self.section_ax.plot(width, section_el, c='k', lw=3)
        self.rh_ax.plot(rh, el, c='k', lw=3)
        self.rhp_ax.plot(rhp, el, c='k', lw=3)
        self.rhp_ax.axvline(ave, ls='dashed', c='k', alpha=0.2)
        self.rhp_ax.axvline(0.5, ls='solid', c='k', alpha=0.7)

    def add_edzs(self, edzs, main_edz):
        for e in edzs:
            edz = edzs[e]
            self.section_ax.fill_between([min(self.width), max(self.width)], [edz['stop_el_scaled'], edz['stop_el_scaled']], [edz['start_el_scaled'], edz['start_el_scaled']], fc='lightblue', alpha=0.9)
            self.rh_ax.fill_between([min(self.rh), max(self.rh)], [edz['stop_el_scaled'], edz['stop_el_scaled']], [edz['start_el_scaled'], edz['start_el_scaled']], fc='lightblue', alpha=0.9)
            self.rhp_ax.fill_between([-1, 1], [edz['stop_el_scaled'], edz['stop_el_scaled']], [edz['start_el_scaled'], edz['start_el_scaled']], fc='lightblue', alpha=0.9)
        
        if main_edz:
            start = main_edz['start_ind']
            stop = main_edz['stop_ind']
            self.rhp_ax.fill_betweenx(self.el[start:stop], 0.5, self.rhp[start:stop])

    def add_aeps(self, q, da):
       for reg in REGRESSIONS['peak_flowrate']:
            params = REGRESSIONS['peak_flowrate'][reg]
            q_ri = (params[0] * ((da / 2.59) ** params[1])) / 35.3147
            norm_stage = np.interp(q_ri, q, self.el)

            self.section_ax.axhline(norm_stage, c='c', alpha=0.3, ls='dashed')
            self.section_ax.text(min(self.width), norm_stage, reg, horizontalalignment='left', verticalalignment='bottom', fontsize='xx-small')
            self.rh_ax.axhline(norm_stage, c='c', alpha=0.3, ls='dashed')
            self.rh_ax.text(min(self.rh), norm_stage, reg, horizontalalignment='left', verticalalignment='bottom', fontsize='xx-small')
            self.rhp_ax.axhline(norm_stage, c='c', alpha=0.3, ls='dashed')
            self.rhp_ax.text(-1, norm_stage, reg, horizontalalignment='left', verticalalignment='bottom', fontsize='xx-small')

    def save(self, dpi=100):
        # update extents
        self.section_ax.set(xlim=(min(self.width), max(self.width)), ylim=(0, 6))
        self.rh_ax.set(xlim=(min(self.rh), max(self.rh)), ylim=(0, 6))
        self.rhp_ax.set(xlim=(-1, 1), ylim=(0, 6))

        # Export
        self.fig.suptitle(self.reach)
        self.fig.tight_layout()
        self.fig.savefig(self.out_path, dpi=dpi)
        plt.close()


def get_edzs(el, el_scaled, rh, rh_prime, widths, thresh=0.5, max_stage=2.5):
    # Initialize outputs
    edzs = dict()

    # Establish additional params
    max_stage_ind = np.argmax(el_scaled > max_stage)
    bathymetry_break = np.argmax(widths > widths[0])  # will only work for rectangular cross sections
    stage_inc = el[1] - el[0]
    stage_inc_scaled = el_scaled[1] - el_scaled[0]

    # Define potential EDZs
    edz_bool = (rh_prime < thresh)
    transitions = edz_bool[:-1] != edz_bool[1:]
    transitions[0] = True
    transitions[-1] = True
    edz_indices = np.argwhere(transitions)[:, 0]

    # Add attributes
    for i in range(len(edz_indices) - 1):
        start, stop = edz_indices[i], edz_indices[i + 1]
        # Limit EDZs.  Above-channel to max-stage
        if start < bathymetry_break:
            if stop < bathymetry_break:
                continue
            else:
                start = bathymetry_break
        elif start > max_stage_ind:
            continue

        tmp_rhp = rh_prime[start:stop]

        vol = (thresh - tmp_rhp).sum() * stage_inc
        vol_scaled = (thresh - tmp_rhp).sum() * stage_inc_scaled
        if vol <= 0:
            continue

        start_el = el[start]
        stop_el = el[stop]
        start_el_scaled = el_scaled[start]
        stop_el_scaled = el_scaled[stop]

        height = stop_el - start_el
        height_scaled = stop_el_scaled - start_el_scaled

        argmin = np.argmin(tmp_rhp) + start
        min_val = rh_prime[argmin]
        el_argmin = el[argmin]
        el_argmin_scaled = el_scaled[argmin]

        slope_start_min = (min_val - thresh) / (el_argmin - start_el)
        slope_min_stop = (thresh - min_val) / (stop_el - el_argmin)

        rh_bottom = rh[max(bathymetry_break - 1, 0)]
        rh_edap = rh[start]
        rh_min = rh[argmin]
        rh_edep = rh[stop]

        w_bottom = widths[max(bathymetry_break - 1, 0)]
        w_edap = widths[start]
        w_min = widths[argmin]
        w_edep = widths[stop]
    
        edzs[i] = {
            'start_ind': start,
            'stop_ind': stop,
            'start_el': start_el,
            'stop_el': stop_el,
            'start_el_scaled': start_el_scaled,
            'stop_el_scaled': stop_el_scaled,
            'volume': vol,
            'vol_scaled': vol_scaled,
            'height': height,
            'height_scaled': height_scaled,
            'min_val': min_val,
            'min_ind': argmin,
            'min_el': el_argmin,
            'min_el_scaled': el_argmin_scaled,
            'slope_start_min': slope_start_min,
            'slope_min_stop': slope_min_stop,
            'rh_bottom': rh_bottom,
            'rh_edap': rh_edap,
            'rh_min': rh_min,
            'rh_edep': rh_edep,
            'w_bottom': w_bottom,
            'w_edap': w_edap,
            'w_min': w_min,
            'w_edep': w_edep
        }

    return edzs


def extract_features(run_path, plot=False):
    # Load data
    with open(run_path, 'r') as f:
        run_dict = json.loads(f.read())
    working_dir = run_dict['geometry_directory']
    reach_path = run_dict['reach_meta_path']
    el_path = os.path.join(working_dir, 'el.csv')
    el_scaled_path = os.path.join(working_dir, 'el_scaled.csv')
    rh_path = os.path.join(working_dir, 'rh.csv')
    rh_prime_path = os.path.join(working_dir, 'rh_prime.csv')
    area_path = os.path.join(working_dir, 'area.csv')
    volume_path = os.path.join(working_dir, 'vol.csv')

    reach_data = pd.read_csv(reach_path)
    reach_data = reach_data.dropna(axis=0)
    reach_data['ReachCode'] = reach_data['ReachCode'].astype(np.int64).astype(str)
    reach_data = reach_data.set_index('ReachCode')

    el_data = pd.read_csv(el_path)
    el_data = el_data.dropna(axis=1)

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

    volume_data = pd.read_csv(volume_path)
    volume_data = volume_data.dropna(axis=1)

    # Get reaches
    valid_reaches = set(reach_data.index.tolist())
    valid_reaches = valid_reaches.intersection(el_data.columns)
    valid_reaches = valid_reaches.intersection(rh_data.columns)
    valid_reaches = valid_reaches.intersection(rh_prime_data.columns)
    valid_reaches = valid_reaches.intersection(area_data.columns)
    valid_reaches = valid_reaches.intersection(volume_data.columns)
    valid_reaches = valid_reaches.intersection(['4300103000354', '4300103001161', '4300103000982', '4300103003935', '4300103000593', '4300103001614', '4300103000851', '4300103000774', '4300103001615', '4300102001939', '4300102003887', '4300102003254', '4300102000550', '4300102002497', '4300102002905', '4300102000630', '4300102000677', '4300102000427', '4300102004925', '4300102007625', '4300102001305', '4300102007508', '4300102001169', '4300102007401', '4300102003558', '4300102005816', '4300102001982', '4300102007100', '4300102006177', '4300102003892', '4300102001790', '4300102003372', '4300102000063', '4300102003800', '4300102001600', '4300102000226', '4300102002356', '4300102000793', '4300102006167', '4300102000386', '4300102003504', '4300102000601', '4300102002065', '4300102000098', '4300102000099', '4300102000309', '4300102000219', '4300102000221', '4300107000762', '4300107000695', '4300107001240', '4300107001221', '4300107000086', '4300107000250', '4300107001574', '4300103000937', '4300103000749', '4300103000054', '4300103001199', '4300103001512', '4300108004810', '4300108006384', '4300108005994', '4300108000265', '4300108004999', '4300108006197', '4300108005916', '4300103003276', '4300103002522', '4300103000182', '4300103000516', '4300102006261', '4300102005415', '4300102005384', '4300102002132', '4300102005477', '4300100000000'])
    valid_reaches = sorted(valid_reaches)
    
    # Extract features
    features = list()
    for reach in list(valid_reaches):
        print(reach)
        # Subset data
        tmp_meta = reach_data.loc[reach]
        tmp_el = el_data[reach].to_numpy()
        tmp_el_scaled = el_scaled_data[reach].to_numpy()
        tmp_rh = rh_data[reach].to_numpy()
        tmp_rh_prime = rh_prime_data[reach].to_numpy()
        tmp_area = area_data[reach].to_numpy() / tmp_meta['length']
        tmp_volume = volume_data[reach].to_numpy() / tmp_meta['length']

        if plot:
            reach_plot = ReachPlot(run_dict['geometry_diagnostics'], reach)

        # Error handling
        if np.all(tmp_area == 0):
            if plot:
                reach_plot.no_geometry()
            tmp_features = ERROR_ARRAY.copy()
            tmp_features[0] = reach
            features.append(tmp_features)
            continue

        # Process
        thresh = 0.5
        max_stage = 2.5
        
        edzs = get_edzs(tmp_el, tmp_el_scaled, tmp_rh, tmp_rh_prime, tmp_area, thresh, max_stage)

        # Generate general stats
        edz_count = len(edzs)
        edz_vols = [edzs[e]['volume'] for e in edzs]
        cum_vol = sum(edz_vols)
        cum_height = sum([edzs[e]['height'] for e in edzs])
        bathymetry_break = np.argmax(tmp_area > tmp_area[0])  # will only work for rectangular cross sections
        el_bathymetry = tmp_el[bathymetry_break]
        el_bathymetry_scaled = tmp_el_scaled[bathymetry_break]
        ave = np.nanmean(tmp_rh_prime)
        stdev = np.nanstd(tmp_rh_prime)

        if edz_count == 0:
            tmp_features = ERROR_ARRAY.copy()
            tmp_features[0] = reach
            main_edz = None
            features.append(tmp_features)
        else:
            main_edz_ind = [i for v, i in sorted(zip(edz_vols, edzs.keys()))][0]
            main_edz = edzs[main_edz_ind]
            features.append([reach, ave, el_bathymetry, main_edz['start_el'], main_edz['min_el'], main_edz['stop_el'], el_bathymetry_scaled, main_edz['start_el_scaled'], main_edz['min_el_scaled'], main_edz['stop_el_scaled'], main_edz['height'], main_edz['height_scaled'], main_edz['volume'], main_edz['vol_scaled'], main_edz['min_val'], main_edz['slope_start_min'], main_edz['slope_min_stop'], main_edz['rh_bottom'], main_edz['rh_edap'], main_edz['rh_min'], main_edz['rh_edep'], main_edz['w_bottom'], main_edz['w_edap'], main_edz['w_min'], main_edz['w_edep'], 0])

        if plot:
            reach_plot.add_geometry(tmp_el_scaled, tmp_area, tmp_rh, tmp_rh_prime, ave)
            reach_plot.add_edzs(edzs, main_edz)
            q = (1 / np.repeat(0.035, len(tmp_el))) * tmp_volume * (tmp_rh ** (2 / 3)) * (tmp_meta['slope'] ** 0.5)
            da = tmp_meta['TotDASqKm']
            reach_plot.add_aeps(q, da)
            reach_plot.save()

    # Save
    # columns = ['ReachCode', 'ave_rhp', 'el_bathymetry', 'el_edap', 'el_min', 'el_edep', 'el_bathymetry_scaled', 'el_edap_scaled', 'el_min_scaled', 'el_edep_scaled', 'height', 'height_scaled', 'vol', 'vol_scaled', 'min_rhp', 'slope_start_min', 'slope_min_stop', 'rh_bottom', 'rh_edap', 'rh_min', 'rh_edep', 'w_bottom', 'w_edap', 'w_min', 'w_edep', 'invalid_geometry']
    # if os.path.exists(run_dict['muskingum_path']):
    #     merge_df = run_dict['muskingum_path']

    #     merge_df = pd.read_csv(merge_df)
    #     merge_df['ReachCode'] = merge_df['ReachCode'].astype(int).astype(str)

    #     out_df = pd.DataFrame(features, columns=columns)
    #     out_df = merge_df.merge(out_df, how='inner', on='ReachCode')
    # else:
    #     out_df = pd.DataFrame(features, columns=columns)
    # os.makedirs(os.path.dirname(run_dict['analysis_path']), exist_ok=True)
    # out_df.to_csv(run_dict['analysis_path'], index=False)


if __name__ == '__main__':
    run_path = r'/netfiles/ciroh/floodplainsData/runs/5/run_metadata.json'
    extract_features(run_path, plot=True)
