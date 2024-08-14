import numpy as np
import pandas as pd
import sys
import argparse
import os
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splev, splrep
import matplotlib.pyplot as plt
import json

# Load regressions
reg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'regressions.json')
with open(reg_path) as in_file:
    REGRESSIONS = json.load(in_file)

FEATURE_NAMES = ['length', 'slope', 'DASqKm', 'wbody', 'ave_rhp', 'stdev_rhp', 'Ave_Rh', 'cumulative_volume', 'cumulative_height', 'valley_confinement', 'el_bathymetry', 'el_edap', 'el_min', 'el_edep', 'el_bathymetry_scaled', 'el_edap_scaled', 'el_min_scaled', 'el_edep_scaled', 'height', 'height_scaled', 'vol', 'vol_scaled', 'min_rhp', 'slope_start_min', 'slope_min_stop', 'rh_bottom', 'rh_edap', 'rh_min', 'rh_edep', 'w_bottom', 'w_edap', 'w_min', 'w_edep', 'w_edap_scaled', 'w_edep_scaled', 'edz_count', 'min_loc_ratio', 'rhp_pre', 'rhp_post', 'rhp_post_stdev', 'invalid_geometry', 'regression_valley_confinement', 'streamorder']
ERROR_ARRAY = [np.nan for i in FEATURE_NAMES]
ERROR_DICT = {k: np.nan for k in FEATURE_NAMES}

### PARSING ###
parser = argparse.ArgumentParser(description='Extract EDZ features from reach-scale geometry.')
parser.add_argument('meta_path', type=str, help='Path to run_metadata.json for this run.')
parser.add_argument('-plot', type=bool, help='Whether to generate diagnostic plots (optional)', required=False)
parser.add_argument('-subset', type=list, help='List of reaches to analyze (optional)', required=False)


class ReachPlot:

    def __init__(self, out_dir, reach, da, slope) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.reach = reach
        self.da = da
        self.slope = slope
        self.out_path = os.path.join(out_dir, f'{reach}.png')
        self.fig, (self.section_ax, self.rh_ax, self.rhp_ax) = plt.subplots(ncols=3, figsize=(10, 3), sharey=True)
        self.all_ax = (self.section_ax, self.rh_ax, self.rhp_ax)
        self.has_geom = False

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
        self.has_geom = True
        # convert semi-cross-section to pseudo-cross-section
        width = width / 2
        width = np.append(-width[::-1], width)
        width = width - min(width)
        # width = width / (2.44 * (self.da ** 0.34))
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

    def add_aeps(self, q):
        for reg in REGRESSIONS['peak_flowrate']:
            params = REGRESSIONS['peak_flowrate'][reg]
            q_ri = (params[0] * ((self.da / 2.59) ** params[1])) / 35.3147
            norm_stage = np.interp(q_ri, q, self.el)

            self.section_ax.axhline(norm_stage, c='c', alpha=0.3, ls='dashed')
            self.section_ax.text(min(self.width), norm_stage, reg, horizontalalignment='left', verticalalignment='bottom', fontsize='xx-small')
            self.rh_ax.axhline(norm_stage, c='c', alpha=0.3, ls='dashed')
            self.rh_ax.text(min(self.rh), norm_stage, reg, horizontalalignment='left', verticalalignment='bottom', fontsize='xx-small')
            self.rhp_ax.axhline(norm_stage, c='c', alpha=0.3, ls='dashed')
            self.rhp_ax.text(-1, norm_stage, reg, horizontalalignment='left', verticalalignment='bottom', fontsize='xx-small')


    def add_spline_rh(self, spl_dict):
        self.rh_ax.plot(spl_dict['rh_appr'], spl_dict['el_scaled'], c='darkorange', lw=1, alpha=0.7, zorder=4)
        self.rh_ax.scatter(spl_dict['simp_rh'], spl_dict['simp_el_scaled'], c='darkorange', alpha=0.7, zorder=3, s=4)


    def save(self, dpi=100):
        # update extents
        if self.has_geom:
            self.section_ax.set(xlim=(min(self.width), max(self.width)), ylim=(0, 6))
            self.rh_ax.set(xlim=(min(self.rh), max(self.rh)), ylim=(0, 6))
            self.rhp_ax.set(xlim=(-1, 1), ylim=(0, 6))

        # Export
        self.fig.suptitle(f'{self.reach} | {round(self.da, 1)} sqkm | {self.slope} m/m')
        self.fig.tight_layout()
        self.fig.savefig(self.out_path, dpi=dpi)
        plt.close()


def get_edzs(el, el_scaled, rh, rh_prime, widths, thresh=0.5, max_stage=2.5):
    # Initialize outputs
    edzs = dict()

    # Establish additional params
    max_stage_ind = np.argmax(el_scaled > max_stage)
    bathymetry_break = np.argmax(widths > widths[0]) - 1  # will only work for rectangular cross sections
    stage_inc = el[1] - el[0]
    stage_inc_scaled = el_scaled[1] - el_scaled[0]

    # Define potential EDZs
    edz_bool = (rh_prime < thresh)
    transitions = edz_bool[:-1] != edz_bool[1:]
    # transitions[0] = True
    transitions = np.insert(transitions, 0, True)
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

        argmin = max(1, np.argmin(tmp_rhp)) + start  # just updated, was min.  Whoops
        min_val = rh_prime[argmin]
        el_argmin = el[argmin]
        el_argmin_scaled = el_scaled[argmin]

        if argmin == start:
            slope_start_min = (min_val - thresh) / (el[argmin + 1] - start_el)
        else:
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

def perpendicular_distance(p, p1, p2):
    """
    Calculate the perpendicular distance from point p to the line segment defined by points p1 and p2.
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1

    if dx == dy == 0:
        # The line segment is a point.
        return np.hypot(p[0] - x1, p[1] - y1)

    u = ((p[0] - x1) * dx + (p[1] - y1) * dy) / (dx * dx + dy * dy)
    if u < 0:
        # Closest point is p1.
        return np.hypot(p[0] - x1, p[1] - y1)
    elif u > 1:
        # Closest point is p2.
        return np.hypot(p[0] - x2, p[1] - y2)
    else:
        # Closest point is along the segment.
        x = x1 + u * dx
        y = y1 + u * dy
        return np.hypot(p[0] - x, p[1] - y)

def douglas_peucker(points, tolerance):
    if len(points) <= 2:
        return points

    # Find the point with the maximum perpendicular distance
    max_distance = 0
    index = 0
    for i in range(1, len(points) - 1):
        distance = perpendicular_distance(points[i], points[0], points[-1])
        if distance > max_distance:
            max_distance = distance
            index = i

    # If the maximum perpendicular distance is greater than the tolerance, split the curve and recursively apply the algorithm
    if max_distance > tolerance:
        left_segment = douglas_peucker(points[:index + 1], tolerance)
        right_segment = douglas_peucker(points[index:], tolerance)
        return left_segment[:-1] + right_segment
    else:
        return [points[0], points[-1]]

def reprocess_rhp(el, rh, tol=0.01, smooth=True):
    points = [(i, j) for i, j in zip(el, rh)]
    simplified_points = douglas_peucker(points, tol)
    simp_rh = [i[1] for i in simplified_points]
    simp_el = [i[0] for i in simplified_points]
    # el_inc = np.median(el[1:] - el[:-1])
    # el_fit = np.arange(0, max(el), el_inc)
    el_fit = el
    rh_fit = np.interp(el_fit, simp_el, simp_rh)
    if smooth:
        spl = splrep(el_fit, rh_fit, t=simp_el[1:-1])
    else:
        spl = splrep(el_fit, rh_fit, s=0)
    rhp = splev(el, spl, der=1)

    spl_dict = {'spl': spl,
                'simp_rh': simp_rh,
                'simp_el': simp_el,
                'el': el,
                'rh_appr': splev(el, spl)}
    return rhp, spl_dict

def extract_features(run_path, plot=False, subset=None):
    # Load data
    with open(run_path, 'r') as f:
        run_dict = json.loads(f.read())
    working_dir = os.path.join(run_dict['run_directory'], 'geometry')
    reach_path = os.path.join(run_dict['run_directory'], 'network', 'reach_data.csv')
    el_path = os.path.join(working_dir, 'el.csv')
    el_scaled_path = os.path.join(working_dir, 'el_scaled.csv')
    rh_path = os.path.join(working_dir, 'rh.csv')
    rh_prime_path = os.path.join(working_dir, 'rh_prime.csv')
    area_path = os.path.join(working_dir, 'area.csv')
    volume_path = os.path.join(working_dir, 'vol.csv')

    reach_data = pd.read_csv(reach_path)
    # reach_data = reach_data.drop(columns=['_wbody', '_max_Slope'])
    # reach_data = reach_data.dropna(axis=0)
    reach_data[run_dict['id_field']] = reach_data[run_dict['id_field']].astype(np.int64).astype(str)
    reach_data = reach_data.set_index(run_dict['id_field'])

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
    if subset:
        valid_reaches = valid_reaches.intersection(subset)
    valid_reaches = sorted(valid_reaches)
    
    # Extract features
    features = pd.DataFrame(columns=FEATURE_NAMES, index=valid_reaches)
    counter = 0
    for reach in list(valid_reaches):
        if counter % 100 == 0:
            print(f'{counter} / {len(valid_reaches)} reaches processed')
        counter += 1
        # Subset data
        tmp_meta = reach_data.loc[reach]
        try:
            tmp_el = el_data[reach].to_numpy()
            tmp_el_scaled = el_scaled_data[reach].to_numpy()
            tmp_rh = rh_data[reach].to_numpy()
            tmp_rh_prime = rh_prime_data[reach].to_numpy()
            tmp_area = area_data[reach].to_numpy() / tmp_meta['length']
            tmp_volume = volume_data[reach].to_numpy() / tmp_meta['length']
        except KeyError:
            if plot:
                reach_plot.no_geometry()
            features.loc[reach, 'invalid_geometry'] = 1
            continue

        features.loc[reach, 'length'] = tmp_meta['length']
        features.loc[reach, 'slope'] = tmp_meta['slope']
        features.loc[reach, 'DASqKm'] = tmp_meta['TotDASqKm']
        features.loc[reach, 'wbody'] = tmp_meta['wbody']

        if plot:
            slope = tmp_meta['slope']
            da = tmp_meta['TotDASqKm']
            diagnostics_path = os.path.join(run_dict['run_directory'], 'geometry', 'diagnostics')
            os.makedirs(diagnostics_path, exist_ok=True)
            reach_plot = ReachPlot(diagnostics_path, reach, da, slope)

        # Error handling
        if np.all(tmp_area < 1):
            if plot:
                reach_plot.no_geometry()
            features.loc[reach, 'invalid_geometry'] = 1
            print(reach)
            continue
        
        # Process
        thresh = 0.5
        max_stage = 2.5
        
        edzs = get_edzs(tmp_el, tmp_el_scaled, tmp_rh, tmp_rh_prime, tmp_area, thresh, max_stage)
        q = (1 / 0.07) * tmp_volume * (tmp_rh ** (2 / 3)) * (tmp_meta['slope'] ** 0.5)
        q500 = REGRESSIONS['peak_flowrate']['Q500'][0] * ((tmp_meta['TotDASqKm'] / 2.59) ** REGRESSIONS['peak_flowrate']['Q500'][1]) * (1 / 35.3147)
        q100 = REGRESSIONS['peak_flowrate']['Q100'][0] * ((tmp_meta['TotDASqKm'] / 2.59) ** REGRESSIONS['peak_flowrate']['Q100'][1]) * (1 / 35.3147)
        q500_ind = np.argmax(q > q500)
        q100_ind = np.argmax(q > q100)
        if q500_ind == 0:
            q500_ind = len(q) - 1
        if q100_ind == 0:
            q100_ind = len(q) - 1
        q500_w = tmp_area[q500_ind]
        q100_w = tmp_area[q100_ind]
        bkf_w = 3.12 * (tmp_meta['TotDASqKm'] ** 0.415)
        regression_valley_confinement = q500_w / bkf_w

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
        ave_rh = np.nanmean(tmp_rh)

        features.loc[reach, 'ave_rhp'] = ave
        features.loc[reach, 'stdev_rhp'] = stdev
        features.loc[reach, 'Ave_Rh'] = ave_rh
        features.loc[reach, 'regression_valley_confinement'] = regression_valley_confinement
        features.loc[reach, 'streamorder'] = tmp_meta['s_order']
        features.loc[reach, 'q500_w'] = q500_w
        features.loc[reach, 'q100_w'] = q100_w
        features.loc[reach, 'invalid_geometry'] = 0
        if edz_count == 0:
            features.loc[reach, 'cumulative_volume'] = 0
            features.loc[reach, 'cumulative_height'] = 0
            features.loc[reach, 'valley_confinement'] = 1
            features.loc[reach, 'el_bathymetry'] = el_bathymetry
            features.loc[reach, 'el_bathymetry_scaled'] = el_bathymetry_scaled
            features.loc[reach, 'edz_count'] = 0
            main_edz = None
        else:
            main_edz_ind = [i for v, i in sorted(zip(edz_vols, edzs.keys()), reverse=True)][0]
            main_edz = edzs[main_edz_ind]
            valley_confinement = main_edz['w_edep'] / main_edz['w_edap']
            min_loc_ratio = (main_edz['min_el'] - main_edz['start_el']) / main_edz['height']
            rhp_pre = tmp_rh_prime[:main_edz['start_ind']].mean()
            rhp_post = tmp_rh_prime[main_edz['stop_ind']:].mean()
            rhp_post_stdev = tmp_rh_prime[main_edz['stop_ind']:].std()

            features.loc[reach, 'cumulative_volume'] = cum_vol
            features.loc[reach, 'cumulative_height'] = cum_height
            features.loc[reach, 'valley_confinement'] = valley_confinement
            features.loc[reach, 'el_bathymetry'] = el_bathymetry
            features.loc[reach, 'el_edap'] = main_edz['start_el']
            features.loc[reach, 'el_min'] = main_edz['min_el']
            features.loc[reach, 'el_edep'] = main_edz['stop_el']
            features.loc[reach, 'el_bathymetry_scaled'] = el_bathymetry_scaled
            features.loc[reach, 'el_edap_scaled'] = main_edz['start_el_scaled']
            features.loc[reach, 'el_min_scaled'] = main_edz['min_el_scaled']
            features.loc[reach, 'el_edep_scaled'] = main_edz['stop_el_scaled']
            features.loc[reach, 'height'] = main_edz['height']
            features.loc[reach, 'height_scaled'] = main_edz['height_scaled']
            features.loc[reach, 'vol'] = main_edz['volume']
            features.loc[reach, 'vol_scaled'] = main_edz['vol_scaled']
            features.loc[reach, 'min_rhp'] = main_edz['min_val']
            features.loc[reach, 'slope_start_min'] = main_edz['slope_start_min']
            features.loc[reach, 'slope_min_stop'] = main_edz['slope_min_stop']
            features.loc[reach, 'rh_bottom'] = main_edz['rh_bottom']
            features.loc[reach, 'rh_edap'] = main_edz['rh_edap']
            features.loc[reach, 'rh_min'] = main_edz['rh_min']
            features.loc[reach, 'rh_edep'] = main_edz['rh_edep']
            features.loc[reach, 'w_bottom'] = main_edz['w_bottom']
            features.loc[reach, 'w_edap'] = main_edz['w_edap']
            features.loc[reach, 'w_min'] = main_edz['w_min']
            features.loc[reach, 'w_edep'] = main_edz['w_edep']
            features.loc[reach, 'w_edap_scaled'] = main_edz['w_edap'] / (3.12 * (tmp_meta['TotDASqKm'] ** 0.415))
            features.loc[reach, 'w_edep_scaled'] = main_edz['w_edep'] / (3.12 * (tmp_meta['TotDASqKm'] ** 0.415))
            features.loc[reach, 'edz_count'] = edz_count
            features.loc[reach, 'min_loc_ratio'] = min_loc_ratio
            features.loc[reach, 'rhp_pre'] = rhp_pre
            features.loc[reach, 'rhp_post'] = rhp_post
            features.loc[reach, 'rhp_post_stdev'] = rhp_post_stdev

        if plot:
            reach_plot.add_geometry(tmp_el_scaled, tmp_area, tmp_rh, tmp_rh_prime, ave)
            reach_plot.add_edzs(edzs, main_edz)
            q = (1 / np.repeat(0.07, len(tmp_el))) * tmp_volume * (tmp_rh ** (2 / 3)) * (tmp_meta['slope'] ** 0.5)
            reach_plot.add_aeps(q)
            reach_plot.save()

    print(f'{len(valid_reaches)} / {len(valid_reaches)} reaches processed')

    # Save
    muskingum_path = os.path.join(run_dict['run_directory'], 'muskingum-cunge', 'mc_data.csv')
    if os.path.exists(muskingum_path):
        merge_df = muskingum_path

        merge_df = pd.read_csv(merge_df)
        merge_df[run_dict['id_field']] = merge_df[run_dict['id_field']].astype(int).astype(str)
        merge_df = merge_df.set_index(run_dict['id_field'])
        merge_df = merge_df.drop(columns=['DASqKm', 'slope'])

        out_df = merge_df.merge(features, how='inner', left_index=True, right_index=True)
    else:
        out_df = features
    analysis_path = os.path.join(run_dict['run_directory'], 'analysis', 'data.csv')
    os.makedirs(os.path.dirname(analysis_path), exist_ok=True)
    out_df.to_csv(analysis_path, index_label=run_dict['id_field'])


if __name__ == '__main__':
    args = parser.parse_args()
    extract_features(args.meta_path, plot=args.plot, subset=args.subset)
