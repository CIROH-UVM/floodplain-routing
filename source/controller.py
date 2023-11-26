import os
import time
import pandas as pd
import geopandas as gpd
import numpy as np
import json
from utilities import subunit_hydraulics, generate_geomorphons, add_bathymetry
from osgeo import ogr


def topographic_signatures(run_dict):
    # Set up run
    if run_dict['scaled_stages']:
        max_stage_equation = lambda da: 5 * (0.26 * (da ** 0.287))
    else:
        max_stage_equation = lambda da: 10

    with open(os.path.join(run_dict['run_directory'], 'run_metadata.json'), 'w') as f:
        json.dump(run_dict, f)

    # Load reaches/basins to run
    reaches = gpd.read_file(run_dict['reach_path'], ignore_geometry=True)
    units = reaches[run_dict['unit_field']].unique()

    # Initialize logging
    data_dict = {f: pd.DataFrame() for f in run_dict['fields_of_interest']}

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
            hand_path = os.path.join(run_dict['data_directory'], unit, 'subbasins', subunit, 'rasters', 'HAND.tif')
            slope_path = os.path.join(run_dict['data_directory'], unit, 'subbasins', subunit, 'rasters', 'slope.tif')
            if not (os.path.exists(hand_path) and os.path.exists(slope_path)):
                print(f'No data for {subunit} found')
                continue

            reachesin_subunit = reaches_in_unit[reaches_in_unit[run_dict["subunit_field"]] == subunit]
            reachesin_subunit = reachesin_subunit.groupby(reachesin_subunit[run_dict['id_field']]).agg(TotDASqKm=('TotDASqKm', 'max'))
            reach_list = reachesin_subunit.index.to_list()
            reachesin_subunit['max_stage'] = reachesin_subunit['TotDASqKm'].apply(max_stage_equation)
            stages = np.array([np.linspace(0, max_stage_equation(dasqkm), 1000) for dasqkm in reachesin_subunit['TotDASqKm'].to_list()])

            su_data_dict = subunit_hydraulics(hand_path, run_dict['reach_path'], slope_path, stages, reach_field=run_dict['id_field'], reaches=reach_list, fields_of_interest=run_dict['fields_of_interest'])

            for f in run_dict['fields_of_interest']:
                data_dict[f] = pd.concat([data_dict[f], su_data_dict[f]], axis=1)
        
        print('\n'*3)
        print(f'Completed processing {unit} in {round((time.perf_counter() - t1) / 60, 1)} minutes')
        print('='*50)

    print('Saving data')
    os.makedirs(run_dict['out_directory'], exist_ok=True)
    for f in run_dict['fields_of_interest']:
        data_dict[f].to_csv(os.path.join(run_dict['out_directory'], f'{f}.csv'), index=False)
    print('Finished saving')

def batch_add_bathymetry(run_dict):
    # Import data
    geometry = {'el': pd.read_csv(os.path.join(run_dict['out_directory'], 'el.csv')),
                'area': pd.read_csv(os.path.join(run_dict['out_directory'], 'area.csv')),
                'vol': pd.read_csv(os.path.join(run_dict['out_directory'], 'vol.csv')),
                'p': pd.read_csv(os.path.join(run_dict['out_directory'], 'p.csv'))}
    reach_data = pd.read_csv(run_dict['reach_meta_path'])
    reach_data['ReachCode'] = reach_data['ReachCode'].astype(np.int64).astype(str)
    reach_data = reach_data.set_index('ReachCode')

    # Clean input data
    valid_columns = set(reach_data.index)
    for col in geometry:
        dataset = geometry[col]
        tmp_cols = dataset.columns[(dataset != 0).any(axis=0)]
        valid_columns = valid_columns.intersection(tmp_cols)
    valid_columns = sorted(valid_columns)
    # valid_columns = ['4300103003055', '4300103003195', '4300103000939', '4300103000324', '4300103003071', '4300103000106', '4300103000749', '4300103002340', '4300103004093', '4300103001288', '4300103003459', '4300103003998', '4300103004141', '4300103001066', '4300103001189', '4300103000965', '4300103001512', '4300103003107', '4300103003995', '4300103005246']

    # Route
    counter = 1
    t_start = time.perf_counter()
    out_dfs = {i: pd.DataFrame() for i in geometry}
    out_dfs['rh'] = pd.DataFrame()
    out_dfs['rh_prime'] = pd.DataFrame()
    
    for reach in valid_columns:
        print(f'{counter} / {len(valid_columns)} | {round((len(valid_columns) - counter) * ((time.perf_counter() - t_start) / counter), 1)} seconds left')
        counter += 1
        # Subset data
        tmp_geom = {i: geometry[i][reach].to_numpy() for i in geometry}
        tmp_meta = reach_data.loc[reach]
        slope = tmp_meta['slope']
        length = tmp_meta['length']
        da = tmp_meta['TotDASqKm']

        # Convert 3D to 2D perspective
        tmp_geom['area'] = tmp_geom['area'] / length
        tmp_geom['vol'] = tmp_geom['vol'] / length
        tmp_geom['p'] = tmp_geom['p'] / length

        tmp_geom = add_bathymetry(tmp_geom, da, slope)

        tmp_geom['area'] = tmp_geom['area'] * length
        tmp_geom['vol'] = tmp_geom['vol'] * length
        tmp_geom['p'] = tmp_geom['p'] * length

        tmp_geom['rh'] = tmp_geom['vol'] / tmp_geom['p']
        tmp_geom['rh_prime'] = (tmp_geom['rh'][1:] - tmp_geom['rh'][:-1]) / (tmp_geom['el'][1:] - tmp_geom['el'][:-1])
        tmp_geom['rh_prime'] = np.append(tmp_geom['rh_prime'], tmp_geom['rh_prime'][-1])

        for i in out_dfs:
            out_dfs[i][reach] = tmp_geom[i]
            out_dfs[i] = out_dfs[i].copy()
    
    for i in out_dfs:
        out_dfs[i].to_csv(os.path.join(run_dict['out_directory'], f'{i}.csv'))


def batch_geomorphons(working_directory):
    run_list = ['WIN_0504']
    unit_dict = {'WIN': 'winooski', 'OTR': 'otter', 'LKC': 'champlain'}
    for run in run_list:
        print(f'Running basin {run}')
        tstart = time.perf_counter()
        unit = unit_dict[run[:3]]
        subunit = run[-4:]
        tmp_dir = os.path.join(working_directory, unit, 'subbasins', subunit, 'rasters')
        generate_geomorphons(tmp_dir, working_directory)
        print(f'Finished in {round((time.perf_counter() - tstart) / 60), 1} minutes')
        print('='*25)


if __name__ == '__main__':
    base_directory = r'/netfiles/ciroh/floodplainsData'
    run_id = '3'
    run_metadata = {'data_directory': base_directory,
                    'run_directory': os.path.join(base_directory, 'runs', run_id), 
                    'out_directory': os.path.join(base_directory, 'runs', run_id, 'outputs'), 
                    'reach_path': os.path.join(base_directory, 'runs', run_id, 'catchments.shp'),
                    'reach_meta_path': os.path.join(base_directory, 'runs', run_id, 'reach_data.csv'), 
                    'id_field': 'MergeCode', 
                    'unit_field': '8_name',
                    'subunit_field': '12_code',
                    'fields_of_interest': ['area', 'el', 'p', 'rh', 'rh_prime', 'vol'], 
                    'scaled_stages': True}
    
    topographic_signatures(run_metadata)
    batch_add_bathymetry(run_metadata)
