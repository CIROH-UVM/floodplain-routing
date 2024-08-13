import json
import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
np.seterr(divide='ignore', invalid='ignore')

### Static Data ###
T_TP_ORDINATES = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 4.5, 5])
Q_QP_ORDINATES = np.array([0, 0.03, 0.1, 0.19, 0.31, 0.47, 0.66, 0.82, 0.93, 0.99, 1, 0.99, 0.93, 0.86, 0.78, 0.68, 0.56, 0.46, 0.39, 0.33, 0.28, 0.207, 0.147, 0.107, 0.077, 0.055, 0.04, 0.029, 0.021, 0.015, 0.011, 0.005, 0])
with open('source/regressions.json') as in_file:
    REGRESSIONS = json.load(in_file)

### PARSING ###
parser = argparse.ArgumentParser(description='Calculate flood-specific metrics for reaches.')
parser.add_argument('meta_path', type=str, help='Path to the run metadata file.')
parser.add_argument('-magnitudes', type=str, help='Comma-separated list of magnitudes to analyze. Ex. ["Q2","Q100"]', required=False)
parser.add_argument('-durations', type=str, help='Comma-separated list of durations to analyze.  Ex. ["Short","Medium"]', required=False)
parser.add_argument('-q_method', type=str, help='Method to calculate discharge. Options: 1-channel, 2-channel, 3-channel.', required=False, default='1-channel')

def get_hydrograph(da, magnitude, duration, resolution=None):
    # Get hydrograph ordinates from regressions
    q_peak = (REGRESSIONS['peak_flowrate'][magnitude][0] * ((da / 2.59) ** REGRESSIONS['peak_flowrate'][magnitude][1])) / 35.3147  # m^3/s
    t_peak = (REGRESSIONS['duration'][f'{magnitude}_{duration}'][0] * (da ** REGRESSIONS['duration'][f'{magnitude}_{duration}'][1])) * 60  # seconds
    q_values = q_peak * Q_QP_ORDINATES
    t_values = t_peak * T_TP_ORDINATES

    # resample
    if type(resolution) == int:
        new_t = np.linspace(0, t_values.max(), 1000)
        q_values = np.interp(new_t, t_values, q_values)
        t_values = new_t
    
    return q_values, t_values

class Reach:

    def __init__(self, da, slope, length, el, tw, area, p, radius, mannings, edap, edep, q_method='1-channel'):
        self.da = da
        self.slope = slope
        self.length = length
        self.el = el.values
        self.tw = tw.values
        self.p = p.values
        self.area = area.values
        self.radius = radius.values
        self.mannings = mannings.values
        self.weighted_mannings = np.cumsum(mannings * p) / np.cumsum(p)

        self.ch_width = np.interp(edap, el, tw)
        self.edz_width = np.interp(edep, el, tw) - self.ch_width
        self.ch_widths = tw.values.copy()
        self.ch_widths[el > edap] = np.repeat(self.ch_width, len(el))[el > edap]
        self.edz_widths = tw - self.ch_widths
        self.edz_widths[el > edep] = self.edz_width
        self.fp_widths = tw - (self.ch_widths + self.edz_widths)

        self.ch_p = np.interp(edap, el, p)
        self.edz_p = np.interp(edep, el, p) - self.ch_p
        self.ch_ps = p.values.copy()
        self.ch_ps[el > edap] = np.repeat(self.ch_p, len(el))[el > edap]
        self.edz_ps = p - self.ch_ps
        self.edz_ps[el > edep] = self.edz_p
        self.fp_ps = p - (self.ch_ps + self.edz_ps)

        self.ch_area = np.interp(edap, el, area)
        self.edz_area = np.interp(edep, el, area) - (self.ch_area + ((edep - edap) * self.ch_width))
        self.ch_areas = area.values.copy()
        self.ch_areas[el > edap] = (self.ch_area + ((el - edap) * self.ch_width))[el > edap]
        self.edz_areas = area.values - self.ch_areas
        self.edz_areas[el > edep] = (self.edz_area + ((el - edep) * self.edz_width))[el > edep]
        self.fp_areas = area.values - (self.ch_areas + self.edz_areas)

        self.ch_radius = self.ch_areas / self.ch_ps
        self.ch_radius = np.abs(np.nan_to_num(self.ch_radius, 0, posinf=0, neginf=0))
        self.edz_radius = self.edz_areas / self.edz_ps
        self.edz_radius = np.abs(np.nan_to_num(self.edz_radius, 0, posinf=0, neginf=0))
        self.fp_radius = self.fp_areas / self.fp_ps
        self.fp_radius = np.abs(np.nan_to_num(self.fp_radius, 0, posinf=0, neginf=0))

        self.ch_n = np.cumsum(mannings * self.ch_ps) / np.cumsum(self.ch_ps)
        self.edz_n = np.cumsum(mannings * self.edz_ps) / np.cumsum(self.edz_ps)
        self.fp_n = np.cumsum(mannings * self.fp_ps) / np.cumsum(self.fp_ps)

        if q_method == '1-channel':
            self.discharge = (1 / self.weighted_mannings) * (slope ** 0.5) * (radius ** (2/3)) * (area)
        elif q_method == '2-channel':
            n_channel_mod = ((self.ch_n * self.ch_ps) + (self.fp_n * self.fp_ps)) / (self.ch_ps + self.fp_ps)
            radius_channel_mod = (self.ch_areas + self.fp_areas) / (self.ch_ps + self.fp_ps)
            q_channel = (1 / n_channel_mod) * (slope ** 0.5) * (radius_channel_mod ** (2/3)) * (self.ch_areas + self.fp_areas)
            q_channel = np.nan_to_num(q_channel, 0)
            q_edz = (1 / self.edz_n) * (slope ** 0.5) * (self.edz_radius ** (2/3)) * (self.edz_areas)
            q_edz = np.nan_to_num(q_edz, 0)
            self.discharge = q_channel + q_edz
        elif q_method == '3-channel':
            q_channel = (1 / self.ch_n) * (slope ** 0.5) * (self.ch_radius ** (2/3)) * (self.ch_areas)
            q_channel = np.nan_to_num(q_channel, 0)
            q_edz = (1 / self.edz_n) * (slope ** 0.5) * (self.edz_radius ** (2/3)) * (self.edz_areas)
            q_edz = np.nan_to_num(q_edz, 0)
            q_fp = (1 / self.fp_n) * (slope ** 0.5) * (self.fp_radius ** (2/3)) * (self.fp_areas)
            q_fp = np.nan_to_num(q_fp, 0)
            self.discharge = q_channel + q_edz + q_fp
        
        self.discharge = np.nan_to_num(self.discharge, 0, posinf=0)
        increasing = self.discharge[1:] > np.maximum.accumulate(self.discharge)[:-1]
        increasing = np.insert(increasing, 0, True)
        for param in ['el', 'tw', 'area', 'p', 'radius', 'mannings', 'weighted_mannings', 'ch_widths', 'edz_widths', 'fp_widths', 'ch_ps', 'edz_ps', 'fp_ps', 'ch_areas', 'edz_areas', 'fp_areas', 'ch_radius', 'edz_radius', 'fp_radius', 'ch_n', 'edz_n', 'fp_n', 'discharge']:
            setattr(self, param, getattr(self, param)[increasing])
        self.velocity = self.discharge / self.area
        self.velocity = np.nan_to_num(self.velocity, 0, posinf=0)
    
    def analyze_hydrograph(self, q_vals, dt):
        if q_vals.max() > self.discharge.max():
            return None
        
        event_ch_a = np.interp(q_vals, self.discharge, self.ch_areas)
        event_edz_a = np.interp(q_vals, self.discharge, self.edz_areas)
        event_fp_a = np.interp(q_vals, self.discharge, self.fp_areas)
        event_a = np.interp(q_vals, self.discharge, self.area)

        event_velocity = np.interp(q_vals, self.discharge, self.velocity)

        event_volume_ch = np.sum(event_velocity * event_ch_a * dt)
        event_volume_edz = np.sum(event_velocity * event_edz_a * dt)
        event_volume_fp = np.sum(event_velocity * event_fp_a * dt)
        event_volume = event_volume_ch + event_volume_edz + event_volume_fp
        event_vol_check = np.sum(q_vals * dt)
        volume_conservation = 1 - (np.abs(event_volume - event_vol_check) / event_vol_check)

        event_ssp_ch = (9810 * self.slope * (self.velocity * self.ch_area)) / self.ch_widths
        event_ssp_ch = np.sum(np.nan_to_num(event_ssp_ch, 0, posinf=0, neginf=0))
        event_ssp_edz = (9810 * self.slope * (self.velocity * self.edz_area)) / self.edz_widths
        event_ssp_edz = np.sum(np.nan_to_num(event_ssp_edz, 0, posinf=0, neginf=0))
        event_ssp_fp = (9810 * self.slope * (self.velocity * self.fp_areas)) / self.fp_widths
        event_ssp_fp = np.sum(np.nan_to_num(event_ssp_fp, 0, posinf=0, neginf=0))
        event_ssp = event_ssp_ch + event_ssp_edz + event_ssp_fp

        tw_ch = np.interp(q_vals.max(), self.discharge, self.ch_widths)
        tw_edz = np.interp(q_vals.max(), self.discharge, self.edz_widths)
        tw_fp = np.interp(q_vals.max(), self.discharge, self.fp_widths)
        tw_sec = np.interp(q_vals.max(), self.discharge, self.tw)

        area_ch = np.interp(q_vals.max(), self.discharge, self.ch_areas)
        area_edz = np.interp(q_vals.max(), self.discharge, self.edz_areas)
        area_fp = np.interp(q_vals.max(), self.discharge, self.fp_areas)
        area_sec = np.interp(q_vals.max(), self.discharge, self.area)

        vol_ch = area_ch * self.length
        vol_edz = area_edz * self.length
        vol_fp = area_fp * self.length
        vol_sec = vol_ch + vol_edz + vol_fp

        peak_stage = np.interp(q_vals.max(), self.discharge, self.el)
        peak_stage_scaled = peak_stage / (0.26 * (self.da ** 0.287))

        metrics = {
            'event_volume': event_volume,
            'event_volume_ch': event_volume_ch,
            'event_volume_edz': event_volume_edz,
            'event_ssp': event_ssp,
            'event_ssp_ch': event_ssp_ch,
            'event_ssp_edz': event_ssp_edz,
            'tw_sec': tw_sec,
            'tw_ch': tw_ch,
            'tw_edz': tw_edz,
            'area_sec': area_sec,
            'area_ch': area_ch,
            'area_edz': area_edz,
            'vol_sec': vol_sec,
            'vol_ch': vol_ch,
            'vol_edz': vol_edz,
            'peak_stage': peak_stage,
            'peak_stage_scaled': peak_stage_scaled,
            'volume_conservation': volume_conservation
        }
        return metrics

def analyze_floods(meta_path, magnitudes=None, durations=None, q_method='1-channel'):
    # Load run config
    with open(meta_path, 'r') as f:
        run_dict = json.loads(f.read())

    # Load EDZ data
    feature_data = pd.read_csv(run_dict['analysis_path'])
    feature_data[run_dict['id_field']] = feature_data[run_dict['id_field']].astype(str)
    feature_data = feature_data.set_index(run_dict['id_field'])

    # Load reaches metadata
    reaches = pd.read_csv(run_dict['reach_meta_path'])
    reaches[run_dict['id_field']] = reaches[run_dict['id_field']].astype(str)
    reaches = reaches.set_index(run_dict['id_field'])
    units = reaches[run_dict['unit_field']].unique()

    # Load geometry
    el = pd.read_csv(os.path.join(run_dict['geometry_directory'], 'el.csv'))
    area = pd.read_csv(os.path.join(run_dict['geometry_directory'], 'area.csv'))
    volume = pd.read_csv(os.path.join(run_dict['geometry_directory'], 'vol.csv'))
    radius = pd.read_csv(os.path.join(run_dict['geometry_directory'], 'rh.csv'))
    perimeter = pd.read_csv(os.path.join(run_dict['geometry_directory'], 'p.csv'))
    if os.path.exists(os.path.join(run_dict['geometry_directory'], 'mannings.csv')):
        mannings = pd.read_csv(os.path.join(run_dict['geometry_directory'], 'mannings.csv'))
    else:
        mannings = pd.DataFrame(index=el.index, columns=el.columns, dtype=float)   
        mannings[:] = float(0.095)

    # define magnitudes and durations
    available_regressions = REGRESSIONS['duration'].keys()
    if magnitudes is None:
        magnitudes = list(set([x.split('_')[0] for x in available_regressions]))
        durations = list(set([x.split('_')[1] for x in available_regressions]))
    elif durations is None:
        durations = list(set([x.split('_')[1] for x in available_regressions if x.split('_')[0] in magnitudes]))

    # Make an empty dataframe to store results
    metric_list = ['event_volume', 'event_volume_ch', 'event_volume_edz', 'event_ssp', 'event_ssp_ch', 'event_ssp_edz', 'tw_sec', 'tw_ch', 'tw_edz', 'area_sec', 'area_ch', 'area_edz', 'vol_sec', 'vol_ch', 'vol_edz', 'peak_stage', 'peak_stage_scaled', 'volume_conservation']
    all_metrics = [f'{magnitude}_{duration}_{metric}' for magnitude in magnitudes for duration in durations for metric in metric_list]
    out_df = pd.DataFrame(index=feature_data.index, columns=all_metrics)

    # Process units
    for unit in units:
        reaches_in_unit = reaches[reaches[run_dict["unit_field"]] == unit]
        subunits = np.sort(reaches_in_unit[run_dict['subunit_field']].unique())

        t1 = time.perf_counter()
        print(f'Unit: {unit} | Subunits: {len(subunits)}')
        counter = 1
        for subunit in subunits:
            print(f'subunit {counter}: {subunit}')
            counter += 1

            reachesin_subunit = reaches_in_unit[reaches_in_unit[run_dict["subunit_field"]] == subunit]
            reach_list = reachesin_subunit.index.unique()

            # Run calculations
            for reach in reach_list:
                # subset geometry
                tmp_da = reaches.loc[reach, 'TotDASqKm']
                tmp_slope = reaches.loc[reach, 'slope']
                tmp_length = reaches.loc[reach, 'length']
                tmp_el = el[reach]
                tmp_area = area[reach] / tmp_length
                tmp_volume = volume[reach] / tmp_length
                tmp_p = perimeter[reach] / tmp_length
                tmp_radius = radius[reach]
                tmp_mannings = mannings[reach]

                # error catching
                if np.all(tmp_area < 1):
                    out_df.loc[reach, :] = np.nan
                    continue

                # get EDZ info
                tmp_edap = feature_data.loc[reach, 'el_edap']
                tmp_edep = feature_data.loc[reach, 'el_edep']
                if np.isnan(tmp_edap):
                    tmp_edap = tmp_el.max()
                    tmp_edep = tmp_el.max()
                elif np.isnan(tmp_edep):
                    tmp_edep = tmp_el.max()
                if tmp_edap == tmp_edep:
                    tmp_edap = tmp_el.max()
                    tmp_edep = tmp_el.max()

                # make a reach
                tmp_reach = Reach(tmp_da, tmp_slope, tmp_length, tmp_el, tmp_area, tmp_volume, tmp_p, tmp_radius, tmp_mannings, tmp_edap, tmp_edep, q_method=q_method)

                for magnitude in magnitudes:
                    for duration in durations:
                        hydrograph_q, hydrograph_t = get_hydrograph(tmp_da, magnitude, duration)
                        dt = hydrograph_t[1] - hydrograph_t[0]
                        tmp_metrics = tmp_reach.analyze_hydrograph(hydrograph_q, dt)

                        if tmp_metrics is None:
                            metric_subset = [f'{magnitude}_{duration}_{metric}' for metric in metric_list]
                            out_df.loc[reach, metric_subset] = np.nan
                        else:
                            for metric in tmp_metrics.keys():
                                out_df.loc[reach, f'{magnitude}_{duration}_{metric}'] = tmp_metrics[metric]

        print('\n'*3)
        print(f'Completed processing {unit} in {round((time.perf_counter() - t1) / 60, 1)} minutes')
        print('='*50)

    out_path = os.path.join(run_dict['analysis_directory'], 'flood_metrics.csv')
    out_df.to_csv(out_path)


if __name__ == '__main__':
    args = parser.parse_args()
    analyze_floods(**vars(args))