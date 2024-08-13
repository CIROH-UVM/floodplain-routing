import os
import sys
import time
import pandas as pd
import numpy as np
import json
from utilities import subunit_hydraulics, add_bathymetry, nwm_subunit, calc_celerity


def extract_geometry(meta_path):
    # Load run config
    with open(meta_path, 'r') as f:
        run_dict = json.loads(f.read())

    # Set up run
    if run_dict['scaled_stages']:
        max_stage_equation = lambda da: 6 * (0.26 * (da ** 0.287))
    else:
        max_stage_equation = lambda da: 10

    # Load reaches/basins to run
    reaches = pd.read_csv(run_dict['reach_meta_path'], dtype={'12_code': str, run_dict['id_field']: int})
    reaches[run_dict['id_field']] = reaches[run_dict['id_field']].astype(str)
    units = reaches[run_dict['unit_field']].unique()

    # Initialize logging
    data_dict = {f: list() for f in run_dict['fields_of_interest']}

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
            reachesin_subunit = reachesin_subunit.groupby(reachesin_subunit[run_dict['id_field']]).agg(TotDASqKm=('TotDASqKm', 'max'), length=('length', 'sum'))
            reach_list = reachesin_subunit.index.to_list()
            reachesin_subunit['max_stage'] = reachesin_subunit['TotDASqKm'].apply(max_stage_equation)
            stages = np.array([np.linspace(0, max_stage_equation(dasqkm), 1000) for dasqkm in reachesin_subunit['TotDASqKm'].to_list()])

            if run_dict['geometry_source'] == 'HAND':
                hand_path = os.path.join(run_dict['data_directory'], unit, 'subbasins', subunit, 'rasters', 'HAND.tif')
                slope_path = os.path.join(run_dict['data_directory'], unit, 'subbasins', subunit, 'rasters', 'slope.tif')
                if not (os.path.exists(hand_path) and os.path.exists(slope_path)):
                    print(f'No data for {subunit} found')
                    continue

                su_data_dict = subunit_hydraulics(hand_path, run_dict['reach_path'], slope_path, stages, reach_field=run_dict['id_field'], reaches=reach_list, fields_of_interest=run_dict['fields_of_interest'], el_type='hand')
            elif run_dict['geometry_source'] == 'DEM':
                dem_path = os.path.join(run_dict['data_directory'], unit, 'subbasins', subunit, 'rasters', 'DEM.tif')
                slope_path = os.path.join(run_dict['data_directory'], unit, 'subbasins', subunit, 'rasters', 'slope.tif')
                if not (os.path.exists(hand_path) and os.path.exists(slope_path)):
                    print(f'No data for {subunit} found')
                    continue

                su_data_dict = subunit_hydraulics(dem_path, run_dict['reach_path'], slope_path, stages, reach_field=run_dict['id_field'], reaches=reach_list, fields_of_interest=run_dict['fields_of_interest'], el_type='dem')
            elif run_dict['geometry_source'] == 'NWM':
                length_list = reachesin_subunit['length'].values
                da_list = reachesin_subunit['TotDASqKm'].values

                su_data_dict = nwm_subunit(das=da_list, stages=stages, lengths=length_list, reaches=reach_list, fields_of_interest=run_dict['fields_of_interest'])

            for f in run_dict['fields_of_interest']:
                data_dict[f].append(su_data_dict[f])
        
        print('\n'*3)
        print(f'Completed processing {unit} in {round((time.perf_counter() - t1) / 60, 1)} minutes')
        print('='*50)

    print('Saving data')
    os.makedirs(run_dict['geometry_directory'], exist_ok=True)
    for f in run_dict['fields_of_interest']:
        data_dict[f] = pd.concat(data_dict[f], axis=1)
        data_dict[f].to_csv(os.path.join(run_dict['geometry_directory'], f'{f}.csv'), index=False)
    reaches[run_dict['id_field']] = reaches[run_dict['id_field']].astype(np.int64).astype(str)
    reaches = reaches.set_index(run_dict['id_field'])
    data_dict['el_scaled'] = scale_stages(reaches, data_dict['el'])
    data_dict['el_scaled'].to_csv(os.path.join(run_dict['geometry_directory'], 'el_scaled.csv'), index=False)
    print('Finished saving')

def batch_add_bathymetry(meta_path):
    # Load run config
    with open(meta_path, 'r') as f:
        run_dict = json.load(f)

    # Import data
    geometry = {'el': pd.read_csv(os.path.join(run_dict['geometry_directory'], 'el.csv')),
                'area': pd.read_csv(os.path.join(run_dict['geometry_directory'], 'area.csv')),
                'vol': pd.read_csv(os.path.join(run_dict['geometry_directory'], 'vol.csv')),
                'p': pd.read_csv(os.path.join(run_dict['geometry_directory'], 'p.csv')),
                'rh': pd.read_csv(os.path.join(run_dict['geometry_directory'], 'rh.csv')),
                'rh_prime': pd.read_csv(os.path.join(run_dict['geometry_directory'], 'rh_prime.csv'))}
    reach_data = pd.read_csv(run_dict['reach_meta_path'])
    reach_data = reach_data.dropna(subset=[run_dict['id_field']])
    reach_data[run_dict['id_field']] = reach_data[run_dict['id_field']].astype(np.int64).astype(str)
    reach_data = reach_data.set_index(run_dict['id_field'])

    # Clean input data
    valid_columns = set(reach_data.index)
    for col in geometry:
        valid_columns = valid_columns.intersection(geometry[col].columns)
    valid_columns = sorted(valid_columns)

    # Setup output frames
    counter = 1
    t_start = time.perf_counter()
    out_dfs = {i: pd.DataFrame() for i in geometry}
    out_dfs['rh'] = pd.DataFrame()
    out_dfs['rh_prime'] = pd.DataFrame()
    
    for reach in valid_columns:
        if counter % 100 == 0:
            print(f'{counter} / {len(valid_columns)} | {round((len(valid_columns) - counter) * ((time.perf_counter() - t_start) / counter), 1)} seconds left')
        counter += 1
        # Subset data
        tmp_geom = {i: geometry[i][reach].to_numpy() for i in geometry}
        tmp_meta = reach_data.loc[reach]
        slope = tmp_meta['slope']
        length = tmp_meta['length']
        da = tmp_meta['TotDASqKm']

        # handle waterbodies
        if tmp_meta['wbody']:
            for i in out_dfs:
                out_dfs[i][reach] = tmp_geom[i]
                out_dfs[i] = out_dfs[i].copy()
            continue

        # Convert 3D to 2D perspective
        tmp_geom['area'] = tmp_geom['area'] / length
        tmp_geom['vol'] = tmp_geom['vol'] / length
        tmp_geom['p'] = tmp_geom['p'] / length

        # error catching
        filter_arg = np.argmin(tmp_geom['el'] < 0.015)
        top_width = tmp_geom['area'][filter_arg]
        if np.all(tmp_geom['area'] == 0) or top_width < 1:
            for i in out_dfs:
                out_dfs[i][reach] = np.repeat(0, len(tmp_geom['area']))
                out_dfs[i] = out_dfs[i].copy()
            continue

        tmp_geom = add_bathymetry(tmp_geom, da, slope)

        tmp_geom['area'] = tmp_geom['area'] * length
        tmp_geom['vol'] = tmp_geom['vol'] * length
        tmp_geom['p'] = tmp_geom['p'] * length

        tmp_geom['rh'] = tmp_geom['vol'] / tmp_geom['p']
        tmp_geom['rh'] = np.nan_to_num(tmp_geom['rh'])
        tmp_geom['rh_prime'] = (tmp_geom['rh'][1:] - tmp_geom['rh'][:-1]) / (tmp_geom['el'][1:] - tmp_geom['el'][:-1])
        tmp_geom['rh_prime'] = np.append(tmp_geom['rh_prime'], tmp_geom['rh_prime'][-1])
        tmp_geom['rh_prime'] = np.nan_to_num(tmp_geom['rh_prime'])

        for i in out_dfs:
            out_dfs[i][reach] = tmp_geom[i]
            out_dfs[i] = out_dfs[i].copy()
    
    out_dfs['el_scaled'] = scale_stages(reach_data, out_dfs['el'])

    for i in out_dfs:
        out_dfs[i].to_csv(os.path.join(run_dict['geometry_directory'], f'{i}.csv'), index=False)
        continue
    
    with open(meta_path, 'w') as f:
        json.dump(run_dict, f)

    
def scale_stages(reach_data, el_data):
    el_scaled_data = el_data.copy()
    bkf_equation = lambda da: 0.26 * (da ** 0.287)
    reaches = pd.DataFrame({'ReachCode': el_data.columns.astype(str)})
    reach_data = pd.merge(reach_data, reaches, right_on='ReachCode', left_index=True, how='right')
    max_stages = bkf_equation(reach_data['TotDASqKm'].to_numpy())
    el_scaled_data.iloc[:, :] = (el_data.values / max_stages)
    el_scaled_data.iloc[:, 0] = el_data.iloc[:, 0]
    return el_scaled_data

if __name__ == '__main__':
    meta_path = sys.argv[1]
    extract_geometry(meta_path)
    batch_add_bathymetry(meta_path)
