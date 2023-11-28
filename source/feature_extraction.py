import numpy as np
import pandas as pd
import os
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

def extract_features():
    # Load data
    working_dir = r'G:\floodplainsData\runs\2\bathymetry_dev'
    reach_path = r"G:\floodplainsData\runs\2\reach_data.csv"
    el_path = os.path.join(working_dir, 'el.csv')
    rh_path = os.path.join(working_dir, 'rh.csv')
    rh_prime_path = os.path.join(working_dir, 'rh_prime.csv')
    area_path = os.path.join(working_dir, 'area.csv')

    reach_data = pd.read_csv(reach_path).dropna(axis=0).drop_duplicates(subset=['ReachCode'])
    reach_data['ReachCode'] = reach_data['ReachCode'].astype(np.int64).astype(str)
    reach_data = reach_data.set_index('ReachCode')
    el_data = pd.read_csv(el_path)
    el_data = el_data.dropna(axis=1)
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
        tmp_meta = reach_data.loc[reach]
        tmp_el = el_data[reach].to_numpy()
        tmp_rh = rh_data[reach].to_numpy()
        tmp_rh_prime = rh_prime_data[reach].to_numpy()
        tmp_area = area_data[reach].to_numpy() / tmp_meta['length']

        min_val = np.nanmin(tmp_rh_prime)
        argmin = np.nanargmin(tmp_rh_prime)
        el_argmin = tmp_el[argmin]
        w_argmin = tmp_area[argmin]
        argmax_after_min = argmin + np.nanargmax(tmp_rh_prime[argmin:])
        max_after_min = tmp_rh_prime[argmax_after_min]
        el_argmax_after_min = tmp_el[argmax_after_min]

        w_bottom = tmp_area[1]
        el_bathymetry = tmp_el[np.argmax(tmp_area > tmp_area[0])]

        # Get first flat after min
        second_derivative = np.append(tmp_rh_prime[1:], tmp_rh_prime[-1]) - tmp_rh_prime
        arg_0slope_after_min = argmin + np.nanargmax(second_derivative[argmin:] < 0)
        rhp_slope0_after_min = tmp_rh_prime[arg_0slope_after_min]
        el_slope0_after_min = tmp_el[arg_0slope_after_min]

        slope_start_min = (min_val - tmp_rh_prime[0]) / (argmin)
        if argmax_after_min == argmin:
            slope_min_max = 0
        else:
            slope_min_max = (max_after_min - min_val) / (argmax_after_min - argmin)

        ave = np.nanmean(tmp_rh_prime)

        le_ave = (tmp_rh_prime < ave)
        transitions = le_ave[:-1] != le_ave[1:]
        runs = np.cumsum(transitions)
        runs = np.insert(runs, 0, runs[0])
        runs[~le_ave] = -1
        unique, counts = np.unique(runs[runs != -1], return_counts=True)
        if unique.size != 0:
            biggest_run = unique[np.argmax(counts)]
            start = np.argmax(runs == biggest_run)
            stop = (len(tmp_rh_prime) - np.argmax(runs[::-1] == biggest_run)) - 1
            start_el = tmp_el[start]
            stop_el = tmp_el[stop]
            height = stop_el - start_el
            height_ind = stop - start

            vol = np.sum(ave - tmp_rh_prime[start:stop])
            vol_norm = np.sum((ave - tmp_rh_prime[start:stop]) * tmp_el[start:stop])

            if argmin != start:
                slope_start_min = (min_val - ave) / (argmin - start)
            else:
                slope_start_min = 0
            if argmin != stop:
                slope_min_stop = (ave - min_val) / (stop - argmin)
            else:
                slope_min_stop = 0
        else:
            start = 0
            stop = 1
            slope_start_min = 0
            slope_min_stop = 0
            height_ind = 1
            vol = 0
            vol_norm = 0

        rh_edap = tmp_rh[start]
        rh_min = tmp_rh[argmin]
        rh_edep = tmp_rh[stop]

        w_edap = tmp_area[start]
        w_min = tmp_area[argmin]
        w_edep = tmp_area[stop]

        # z = np.polyfit(tmp_el, tmp_rh_prime, 3)
        # p = np.poly1d(z)

        # fig, ax = plt.subplots()

        # # Plot features
        # ax.scatter(min_val, el_argmin, ec='r', fc='none', s=25)
        # ax.text(min_val, el_argmin, 'Min', va='top', ha='right')

        # ax.axvline(ave, c='r', ls='dashed', lw=1)
        # ax.text(ave, tmp_el.max(), 'Average', va='top', ha='right', rotation=90)

        # ax.axhline(start_el, c='r', ls='dotted', lw=1)
        # ax.text(-3, start_el, 'EDAP', va='bottom', ha='left')

        # ax.axhline(el_bathymetry, c='r', ls='dotted', lw=1)
        # ax.text(-3, el_bathymetry, 'Bathymetry', va='bottom', ha='left')

        # ax.axhline(stop_el, c='r', ls='dotted', lw=1)
        # ax.text(-3, stop_el, 'EDEP', va='bottom', ha='left')

        # ax.fill_betweenx(tmp_el[start:stop], tmp_rh_prime[start:stop], np.repeat(ave, height_ind), fc='r')
        # ax.text((min_val + ave) / 2, (tmp_el[start] + tmp_el[stop]) / 2, 'volume', va='center', ha='center')

        # ax.text(ave, (tmp_el[start] + tmp_el[stop]) / 2, 'height ED', va='center', ha='left', rotation=90)

        # # ax.plot(p(tmp_el), tmp_el, ls='dashed', c='gray')

        # ax.plot(tmp_rh_prime, tmp_el, c='k')

        # plt.title(reach)
        # ax.set_xlim(-3, 1)
        # ax.set_ylim(0, tmp_el.max())
        # ax.set_xlabel(r"$R_{h}$'")
        # ax.set_ylabel('Stage (m)')
        # plt.show()

        features.append([reach, ave, min_val, el_bathymetry, argmin, el_argmin, start, start_el, stop, stop_el, height, height_ind, vol, vol_norm, w_bottom, w_edap, w_min, w_edep, rh_edap, rh_min, rh_edep, slope_start_min, slope_min_stop])

    # Save
    merge_df = r"G:\floodplainsData\runs\2\muskingum-cunge\mc_out.csv"
    merge_df = pd.read_csv(merge_df)
    merge_df['ReachCode'] = merge_df['ReachCode'].astype(str)

    out_df = pd.DataFrame(features, columns=['ReachCode', 'ave_rhp', 'min_rhp', 'el_bathymetry', 'ind_min', 'el_min', 'ind_EDAP', 'el_EDAP', 'ind_EDEP', 'el_EDEP', 'height', 'height_ind', 'vol', 'vol_norm', 'w_bottom', 'w_EDAP', 'w_min', 'w_edep', 'rh_edap', 'rh_min', 'rh_edep', 'slope_edap-min', 'slope_min-edep'])
    merge_df = merge_df.merge(out_df, how='inner', on='ReachCode')
    merge_df.to_csv(r"G:\floodplainsData\runs\2\analysis\mc_topo_sig.csv", index=False)


if __name__ == '__main__':
    extract_features()
