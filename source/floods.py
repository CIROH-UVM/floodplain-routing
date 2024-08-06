import json
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


### Static Data ###
T_TP_ORDINATES = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 4.5, 5])
Q_QP_ORDINATES = np.array([0, 0.03, 0.1, 0.19, 0.31, 0.47, 0.66, 0.82, 0.93, 0.99, 1, 0.99, 0.93, 0.86, 0.78, 0.68, 0.56, 0.46, 0.39, 0.33, 0.28, 0.207, 0.147, 0.107, 0.077, 0.055, 0.04, 0.029, 0.021, 0.015, 0.011, 0.005, 0])
with open('source/regressions.json') as in_file:
    REGRESSIONS = json.load(in_file)

def get_rating_curve(slope, area, radius, mannings):
    # Make mannings synthetic rating curve
    discharge = (1 / mannings) * (slope ** 0.5) * (radius ** (2/3)) * (area)
    discharge = np.nan_to_num(discharge, 0)
    # clean discharges for monotonicity
    # q_last = np.nan
    # for ind, q in enumerate(discharge):
    #     if q < q_last:
    #         discharge[ind] = q_last
    #     else:
    #         q_last = q
    # # still cleaning.  Do a backward pass to remove repeat values
    # q_last = discharge[-1]
    # for ind in range(len(discharge)):
    #     ind = len(discharge) - ind - 1
    #     if discharge[ind] == discharge[ind - 1]:
    #         discharge[ind] = (q_last + discharge[ind]) / 2
    #     q_last = discharge[ind]
    
    return discharge

def get_stage_hydrograph(da, el, discharge, magnitude, duration, resolution=None):
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

    # Calculate stage hydrograph
    el_values = np.interp(q_values, discharge, el)

    return el_values, q_values, t_values


def analyze_hydrograph(stages, el, width, area, velocity, discharge, edap, edep, dt, tmp_slope):
    # Break out parameters into channel, edz, and floodplain
    ch_width = np.interp(edap, el, width)
    edz_width = np.interp(edep, el, width)

    ch_widths = np.repeat(ch_width, len(el))
    ch_widths[el < edap] = width[el < edap]

    edz_widths = width - ch_widths
    edz_widths[el > edep] = edz_width

    fp_widths = width - (ch_widths + edz_widths)

    ch_area = np.interp(edap, el, area)
    edz_area = np.interp(edep, el, area)

    ch_areas = ch_area + ((el - edap) * ch_widths)
    ch_areas[el < edap] = area[el < edap]

    edz_areas = area - ch_areas
    edz_areas[el > edep] = edz_area + ((el - edep) * edz_widths)

    fp_areas = area - (ch_areas + edz_areas)

    # interpolate areas
    event_ch_a = np.interp(stages, discharge, ch_areas)
    event_edz_a = np.interp(stages, discharge, edz_areas)
    event_fp_a = np.interp(stages, discharge, fp_areas)
    event_a = np.interp(stages, discharge, area)

    # interpolate velocity
    event_velocity = np.interp(stages, discharge, velocity)

    # calculate metrics
    event_volume_ch = np.sum(event_velocity * event_ch_a * dt)
    event_volume_edz = np.sum(event_velocity * event_edz_a * dt)
    event_volume_fp = np.sum(event_velocity * event_fp_a * dt)
    # event_volume = np.sum(event_velocity * event_a * dt)
    event_volume = event_volume_ch + event_volume_edz + event_volume_fp

    tw_ch = np.interp(stages.max(), discharge, ch_widths)
    tw_edz = np.interp(stages.max(), discharge, edz_widths)
    tw_fp = np.interp(stages.max(), discharge, fp_widths)
    tw_sec = np.interp(stages.max(), discharge, width)

    area_ch = np.interp(stages.max(), discharge, ch_areas)
    area_edz = np.interp(stages.max(), discharge, edz_areas)
    area_fp = np.interp(stages.max(), discharge, fp_areas)
    area_sec = np.interp(stages.max(), discharge, area)

    # Package metrics into dictionary
    metrics = {
        'event_volume': event_volume,
        'event_volume_ch': event_volume_ch,
        'event_volume_edz': event_volume_edz,
        'tw_ch': tw_ch,
        'tw_edz': tw_edz,
        'tw_sec': tw_sec,
        'area_ch': area_ch,
        'area_edz': area_edz,
        'area_sec': area_sec
    }
    return metrics

def analyze_floods(meta_path):
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
    if os.path.exists(os.path.join(run_dict['geometry_directory'], 'mannings.csv')):
        mannings = pd.read_csv(os.path.join(run_dict['geometry_directory'], 'mannings.csv'))
    else:
        mannings = pd.DataFrame(index=el.index, columns=el.columns, dtype=float)   
        mannings[:] = float(0.095)

    # define magnitudes and durations
    magnitudes = ['Q2', 'Q10', 'Q50', 'Q100']
    durations = ['Short', 'Medium', 'Long']

    # Make an empty dataframe to store results
    metric_list = ['event_volume', 'event_volume_ch', 'event_volume_edz', 'tw_ch', 'tw_edz', 'tw_sec', 'area_ch', 'area_edz', 'area_sec', 'peak_stage_scaled', 'volume_conservation']
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
                tmp_radius = radius[reach]
                tmp_mannings = mannings[reach]
                tmp_bkf_depth = (0.26 * (tmp_da ** 0.287))

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

                discharge = get_rating_curve(tmp_slope, tmp_area, tmp_radius, tmp_mannings)
                # remove decreasing values from all series
                increasing = discharge[1:] > np.maximum.accumulate(discharge)[:-1]
                increasing = np.insert(increasing, 0, True)
                discharge = discharge[increasing]
                tmp_el = tmp_el[increasing]
                tmp_area = tmp_area[increasing]
                tmp_volume = tmp_volume[increasing]
                tmp_radius = tmp_radius[increasing]
                tmp_mannings = tmp_mannings[increasing]

                velocity = discharge / tmp_volume
                velocity = np.nan_to_num(velocity, 0, posinf=0)

                for magnitude in magnitudes:
                    for duration in durations:
                        hydrograph_el, hydrograph_q, hydrograph_t = get_stage_hydrograph(tmp_da, tmp_el, discharge, magnitude, duration)
                        out_df.loc[reach, f'{magnitude}_{duration}_peak_stage_scaled'] = hydrograph_el.max() / tmp_bkf_depth
                        if hydrograph_q.max() > discharge.max():
                            metric_subset = [f'{magnitude}_{duration}_{metric}' for metric in metric_list]
                            out_df.loc[reach, metric_subset] = np.nan
                            continue
                        dt = hydrograph_t[1] - hydrograph_t[0]
                        tmp_metrics = analyze_hydrograph(hydrograph_q, tmp_el, tmp_area, tmp_volume, velocity, discharge, tmp_edap, tmp_edep, dt, tmp_slope)

                        # error check
                        event_vol = np.sum(hydrograph_q * dt)
                        conservation_check = 1 - np.abs(event_vol - tmp_metrics['event_volume']) / event_vol
                        out_df.loc[reach, f'{magnitude}_{duration}_volume_conservation'] = conservation_check

                        # Add reach-volume instead of just area
                        tmp_metrics['volume_ch'] = tmp_metrics['area_ch'] * tmp_length
                        tmp_metrics['volume_edz'] = tmp_metrics['area_edz'] * tmp_length
                        tmp_metrics['volume_sec'] = tmp_metrics['area_sec'] * tmp_length

                        for metric in tmp_metrics.keys():
                            out_df.loc[reach, f'{magnitude}_{duration}_{metric}'] = tmp_metrics[metric]

        print('\n'*3)
        print(f'Completed processing {unit} in {round((time.perf_counter() - t1) / 60, 1)} minutes')
        print('='*50)

    out_path = os.path.join(run_dict['analysis_directory'], 'flood_metrics.csv')
    out_df.to_csv(out_path)


if __name__ == '__main__':
    meta_path = sys.argv[1]
    analyze_floods(meta_path)