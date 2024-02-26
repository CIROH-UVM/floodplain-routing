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

        # Error handling
        if np.all(tmp_area == 0):
            if plot:
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.text(0.5, 0.5, 'NO GEOMETRY', fontsize=15, verticalalignment='center', horizontalalignment='center')
                fig.savefig(os.path.join(run_dict['geometry_diagnostics'], f'{reach}.png'), dpi=100)
                plt.close()
            tmp_features = ERROR_ARRAY.copy()
            tmp_features[0] = reach
            features.append(tmp_features)
            continue

        # Process
        bathymetry_break = np.argmax(tmp_area > tmp_area[0])  # will only work for rectangular cross sections
        el_bathymetry = tmp_el[bathymetry_break]
        el_bathymetry_scaled = tmp_el_scaled[bathymetry_break]

        ave = np.nanmean(tmp_rh_prime[bathymetry_break:])
        stdev = np.nanstd(tmp_rh_prime[bathymetry_break:])

        thresh = 0.5
        max_stage = 2.5
        max_stage_ind = np.argmax(tmp_el_scaled > max_stage)

        edz_bool = (tmp_rh_prime < thresh)
        edz_bool[:bathymetry_break] = False  # Enforce channel not being included in EDZ comparisons
        # Allow EDZs only for stages accessed in frequent floods
        if tmp_rh_prime[max_stage_ind] < thresh:
            if np.argmax(tmp_rh_prime[max_stage_ind:] > thresh) == 0:
                max_stage_ind = tmp_rh_prime.shape[0] - 1
            else:
                max_stage_ind = max_stage_ind + np.argmax(tmp_rh_prime[max_stage_ind:] > thresh)  # find first point after max_stage where rh' > thresh
        edz_bool[max_stage_ind:] = False  # Allow EDZs only for stages accessed in frequent floods

        transitions = edz_bool[:-1] != edz_bool[1:]
        if transitions.sum() != 0:
            edz_indices = np.argwhere(transitions)[:, 0]
            edzs = np.split(tmp_rh_prime, edz_indices)
            volumes = np.array([(thresh - e).sum() for e in edzs])
            heights = np.array([e.shape[0] for e in edzs])
            cum_vol = volumes[volumes > 0].sum()
            cum_height = heights[volumes > 0].sum()
            edz_count = (volumes > 0).sum()
        else:
            edz_count = 0
        if edz_count == 0:
            tmp_features = ERROR_ARRAY.copy()
            tmp_features[0] = reach
            features.append(tmp_features)
        else:
            # get main_edz_indices
            edz_indices = np.insert(edz_indices, 0, 0)
            edz_indices = np.append(edz_indices, max_stage_ind)
            start = edz_indices[np.argmax(volumes)]
            stop = edz_indices[np.argmax(volumes) + 1]

            start_el = tmp_el[start]
            stop_el = tmp_el[stop]

            start_el_scaled = tmp_el_scaled[start]
            stop_el_scaled = tmp_el_scaled[stop]

            height = stop_el - start_el
            height_scaled = stop_el_scaled - start_el_scaled

            argmin = np.argmin(tmp_rh_prime[start:stop]) + start
            min_val = tmp_rh_prime[argmin]
            el_argmin = tmp_el[argmin]
            el_argmin_scaled = tmp_el_scaled[argmin]

            del_el_meters = tmp_el[1:] - tmp_el[:-1]
            del_el_meters = np.append(del_el_meters, del_el_meters[-1])
            del_el_scaled = tmp_el_scaled[1:] - tmp_el_scaled[:-1]
            del_el_scaled = np.append(del_el_scaled, del_el_scaled[-1])
            vol = np.sum((ave - tmp_rh_prime[start:stop]) * del_el_meters[start:stop])
            vol_scaled = np.sum((ave - tmp_rh_prime[start:stop]) * del_el_scaled[start:stop])

            if argmin != start:
                # slope_start_min = (min_val - ave) / (argmin - start)  # Old
                slope_start_min = (min_val - ave) / (el_argmin - start_el)
            else:
                slope_start_min = 0
            if argmin != stop:
                # slope_min_stop = (ave - min_val) / (stop - argmin) # Old
                slope_min_stop = (ave - min_val) / (stop_el - el_argmin)
            else:
                slope_min_stop = 0

            rh_bottom = tmp_rh[max(bathymetry_break - 1, 0)]
            rh_edap = tmp_rh[start]
            rh_min = tmp_rh[argmin]
            rh_edep = tmp_rh[stop]

            w_bottom = tmp_area[max(bathymetry_break - 1, 0)]
            w_edap = tmp_area[start]
            w_min = tmp_area[argmin]
            w_edep = tmp_area[stop]

        if plot:

            # generate discharges
            q = (1 / np.repeat(0.035, len(tmp_el))) * tmp_volume * (tmp_rh ** (2 / 3)) * (tmp_meta['slope'] ** 0.5)

            # plot
            fig, (section_ax, rh_ax, rhp_ax) = plt.subplots(ncols=3, figsize=(10, 3), sharey=True)

            tmp_area = tmp_area / 2
            tmp_area = np.append(-tmp_area[::-1], tmp_area)
            tmp_area = tmp_area - min(tmp_area)
            section_el = np.append(tmp_el_scaled[::-1], tmp_el_scaled)
            section_ax.plot(tmp_area, section_el, c='k', lw=3)
            section_ax.fill_between([min(tmp_area), max(tmp_area)], [stop_el_scaled, stop_el_scaled], [start_el_scaled, start_el_scaled], fc='lightblue', alpha=0.9)
            section_ax.set(xlim=(min(tmp_area), max(tmp_area)), ylim=(0, 6), xlabel='Station (m)', ylabel='Stage / Bankfull Depth')
            section_ax.set_xlabel('Station (m)', fontsize=10)
            section_ax.set_ylabel('Stage / Bankfull Depth', fontsize=10)
            
            rh_ax.plot(tmp_rh, tmp_el_scaled, c='k', lw=3)
            rh_ax.fill_between([min(tmp_rh), max(tmp_rh)], [stop_el_scaled, stop_el_scaled], [start_el_scaled, start_el_scaled], fc='lightblue', alpha=0.9)
            rh_ax.set(xlim=(min(tmp_rh), max(tmp_rh)), ylim=(0, 6), xlabel=' ', ylabel='Stage / Bankfull Depth')
            rh_ax.set_xlabel(r'$R_{h}$', fontsize=10)

            rhp_ax.plot(tmp_rh_prime, tmp_el_scaled, c='k', lw=3)
            rhp_ax.axvline(ave, ls='dashed', c='k', alpha=0.2)
            rhp_ax.axvline(0.5, ls='dashed', c='k', alpha=0.7)
            if edz_count != 0:
                rhp_ax.fill_between([-1, 1], [stop_el_scaled, stop_el_scaled], [start_el_scaled, start_el_scaled], fc='lightblue', alpha=0.9)
                rhp_ax.fill_betweenx(tmp_el_scaled[start:stop], 0.5, tmp_rh_prime[start:stop])
            rhp_ax.set(xlim=(-1, 1), ylim=(0, 6), xlabel=' ', ylabel='Stage / Bankfull Depth')
            rhp_ax.set_xlabel(r"$R_{h}$'", fontsize=10)

            for 

            for reg in REGRESSIONS['peak_flowrate']:
                params = REGRESSIONS['peak_flowrate'][reg]
                q_ri = (params[0] * ((tmp_meta['TotDASqKm'] / 2.59) ** params[1])) / 35.3147
                norm_stage = np.interp(q_ri, q, tmp_el_scaled)

                section_ax.axhline(norm_stage, c='c', alpha=0.3, ls='dashed')
                section_ax.text(min(tmp_area), norm_stage, reg, horizontalalignment='left', verticalalignment='bottom', fontsize='xx-small')
                rh_ax.axhline(norm_stage, c='c', alpha=0.3, ls='dashed')
                rh_ax.text(min(tmp_rh), norm_stage, reg, horizontalalignment='left', verticalalignment='bottom', fontsize='xx-small')
                rhp_ax.axhline(norm_stage, c='c', alpha=0.3, ls='dashed')
                rhp_ax.text(-1, norm_stage, reg, horizontalalignment='left', verticalalignment='bottom', fontsize='xx-small')

            fig.tight_layout()
            fig.savefig(os.path.join(run_dict['geometry_diagnostics'], f'{reach}.png'), dpi=100)
            plt.close()

        features.append([reach, ave, el_bathymetry, start_el, el_argmin, stop_el, el_bathymetry_scaled, start_el_scaled, el_argmin_scaled, stop_el_scaled, height, height_scaled, vol, vol_scaled, min_val, slope_start_min, slope_min_stop, rh_bottom, rh_edap, rh_min, rh_edep, w_bottom, w_edap, w_min, w_edep, 0])

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
