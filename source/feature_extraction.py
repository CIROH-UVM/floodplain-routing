import numpy as np
import pandas as pd
import geopandas as gpd
import os
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import json

def scale_stage(reach_data, el_data):
    # el_data = el_data.iloc[:, 1:]
    el_scaled_data = el_data.copy()
    max_stage_equation = lambda da: 5 * (0.26 * (da ** 0.287))
    reaches = pd.DataFrame({'ReachCode': el_data.columns})
    reach_data = pd.merge(reach_data, reaches, right_on='ReachCode', left_index=True, how='right')
    max_stages = max_stage_equation(reach_data['TotDASqKm'].to_numpy())
    el_scaled_data.iloc[:, :] = (el_data.iloc[:, :] / max_stages) * 5
    el_scaled_data.iloc[:, 0] = el_data.iloc[:, 0]
    return el_scaled_data

def extract_features(run_path):
    # Load data
    with open(run_path, 'r') as f:
        run_dict = json.loads(f.read())
    working_dir = run_dict['out_directory']
    reach_path = run_dict['reach_meta_path']
    el_path = os.path.join(working_dir, 'el.csv')
    el_scaled_path = os.path.join(working_dir, 'el_scaled.csv')
    rh_path = os.path.join(working_dir, 'rh.csv')
    rh_prime_path = os.path.join(working_dir, 'rh_prime.csv')
    area_path = os.path.join(working_dir, 'area.csv')

    reach_data = pd.read_csv(reach_path)
    reach_data['ReachCode'] = reach_data['ReachCode'].astype(np.int64).astype(str)
    reach_data = reach_data.set_index('ReachCode')

    el_data = pd.read_csv(el_path)
    el_data = el_data.dropna(axis=1)

    if not os.path.exists(el_scaled_path):
        el_scaled_data = scale_stage(reach_data, el_data)
        el_scaled_data.to_csv(el_scaled_path)
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

    # Get reaches
    valid_reaches = set(reach_data.index.tolist())
    valid_reaches = valid_reaches.intersection(el_data.columns)
    valid_reaches = valid_reaches.intersection(rh_data.columns)
    valid_reaches = valid_reaches.intersection(rh_prime_data.columns)
    valid_reaches = valid_reaches.intersection(area_data.columns)
    valid_reaches = sorted(valid_reaches)
    # valid_reaches = ['4300103003055', '4300103003195', '4300103000939', '4300103000324', '4300103003071', '4300103000106', '4300103000749', '4300103002340', '4300103004093', '4300103001288', '4300103003459', '4300103003998', '4300103004141', '4300103001066', '4300103001189', '4300103000965', '4300103001512', '4300103003107', '4300103003995', '4300103005246']

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

        # Process
        bathymetry_break = np.argmax(tmp_area > tmp_area[0])
        el_bathymetry = tmp_el[bathymetry_break]
        el_bathymetry_scaled = tmp_el_scaled[bathymetry_break]

        ave = np.nanmean(tmp_rh_prime)

        le_ave = (tmp_rh_prime < ave)
        transitions = le_ave[:-1] != le_ave[1:]
        runs = np.cumsum(transitions)
        runs = np.insert(runs, 0, runs[0])
        runs[~le_ave] = -1
        unique, counts = np.unique(runs[runs != -1], return_counts=True)

        biggest_run = unique[np.argmax(counts)]
        start = np.argmax(runs == biggest_run)
        if start < bathymetry_break:
            start = bathymetry_break
        stop = (len(tmp_rh_prime) - np.argmax(runs[::-1] == biggest_run)) - 1

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

        plot = False
        if plot:
            fig, ax = plt.subplots()

            # Plot features
            ax.scatter(min_val, el_argmin, ec='r', fc='none', s=25)
            ax.text(min_val, el_argmin, 'Min', va='top', ha='right')

            ax.axvline(ave, c='r', ls='dashed', lw=1)
            ax.text(ave, tmp_el.max(), 'Average', va='top', ha='right', rotation=90)

            ax.axhline(start_el, c='r', ls='dotted', lw=1)
            ax.text(-3, start_el, 'EDAP', va='bottom', ha='left')

            ax.axhline(el_bathymetry, c='r', ls='dotted', lw=1)
            ax.text(-3, el_bathymetry, 'Bathymetry', va='bottom', ha='left')

            ax.axhline(stop_el, c='r', ls='dotted', lw=1)
            ax.text(-3, stop_el, 'EDEP', va='bottom', ha='left')

            ax.fill_betweenx(tmp_el[start:stop], tmp_rh_prime[start:stop], np.repeat(ave, stop - start), fc='r')
            ax.text((min_val + ave) / 2, (tmp_el[start] + tmp_el[stop]) / 2, 'volume', va='center', ha='center')

            ax.text(ave, (tmp_el[start] + tmp_el[stop]) / 2, 'height ED', va='center', ha='left', rotation=90)

            # ax.plot(p(tmp_el), tmp_el, ls='dashed', c='gray')

            ax.plot(tmp_rh_prime, tmp_el, c='k')

            plt.title(reach)
            ax.set_xlim(-3, 1)
            ax.set_ylim(0, tmp_el.max())
            ax.set_xlabel(r"$R_{h}$'")
            ax.set_ylabel('Stage (m)')
            fig.savefig(r'/netfiles/ciroh/floodplainsData/runs/3/outputs/feature_plots/{}.png'.format(reach))
            # plt.show()

        features.append([reach, ave, el_bathymetry, start_el, el_argmin, stop_el, el_bathymetry_scaled, start_el_scaled, el_argmin_scaled, stop_el_scaled, height, height_scaled, vol, vol_scaled, min_val, slope_start_min, slope_min_stop, rh_bottom, rh_edap, rh_min, rh_edep, w_bottom, w_edap, w_min, w_edep])

    # Save
    merge_df = run_dict['muskingum_path']

    merge_df = pd.read_csv(merge_df)
    merge_df['ReachCode'] = merge_df['ReachCode'].astype(int).astype(str)

    columns = ['ReachCode', 'ave_rhp', 'el_bathymetry', 'el_edap', 'el_min', 'el_edep', 'el_bathymetry_scaled', 'el_edap_scaled', 'el_min_scaled', 'el_edep_scaled', 'height', 'height_scaled', 'vol', 'vol_scaled', 'min_rhp', 'slope_start_min', 'slope_min_stop', 'rh_bottom', 'rh_edap', 'rh_min', 'rh_edep', 'w_bottom', 'w_edap', 'w_min', 'w_edep']
    out_df = pd.DataFrame(features, columns=columns)
    merge_df = merge_df.merge(out_df, how='inner', on='ReachCode')
    os.makedirs(os.path.dirname(run_dict['analysis_path']), exist_ok=True)
    merge_df.to_csv(run_dict['analysis_path'], index=False)


if __name__ == '__main__':
    run_path = r'/netfiles/ciroh/floodplainsData/runs/3/run_metadata.json'
    extract_features(run_path)
