import os
import time
import pandas as pd
import numpy as np
import json
from utilities import subunit_hydraulics, generate_geomorphons, gage_areas_from_poly_gdal
from osgeo import ogr


def topographic_signatures(reach_path, aoi_path, working_directory, out_directory, id_field, fields_of_interest, scaling):
    # Set up run
    if scaling:
        max_stage_equation = lambda da: 5 * (0.26 * (da ** 0.287))
    else:
        max_stage_equation = lambda da: 10
    run_metadata = {'reach_path': reach_path, 
                    'aoi_path': aoi_path, 
                    'id_field': id_field, 
                    'fields_of_interest': fields_of_interest, 
                    'scaled_stages': scaling}
    with open(os.path.join(out_directory, 'run_metadata.json'), 'w') as f:
            json.dump(run_metadata, f)

    # Load reaches/basins to run
    reaches = pd.read_csv(reach_path, dtype={'subunit': 'str', 'unit': 'str'})
    units = reaches['unit'].unique()

    # Initialize logging
    data_dict = {f: pd.DataFrame() for f in fields_of_interest}

    # Process units
    for unit in units:
        reaches_in_unit = reaches.query(f'unit == "{unit}"')
        subunits = np.sort(reaches_in_unit['subunit'].unique())

        t1 = time.perf_counter()
        print(f'Unit: {unit} | Subunits: {len(subunits)}')
        counter = 1
        for subunit in subunits:
            print(f'subunit {counter}: {subunit}')
            counter += 1
            hand_path = os.path.join(working_directory, unit, 'subbasins', subunit, 'rasters', 'HAND.tif')
            slope_path = os.path.join(working_directory, unit, 'subbasins', subunit, 'rasters', 'slope.tif')
            if not (os.path.exists(hand_path) and os.path.exists(slope_path)):
                print(f'No data for {subunit} found')
                continue
            reachesin_subunit = reaches_in_unit.query(f'subunit == "{subunit}"')
            # reachesin_subunit  = reachesin_subunit.drop_duplicates(subset=id_field)
            reachesin_subunit = reachesin_subunit.groupby(reachesin_subunit[id_field]).agg(TotDASqKm=('TotDASqKm', 'max'), Slope=('Slope', 'mean'))
            # reach_list = reachesin_subunit[id_field].to_list()
            reach_list = reachesin_subunit.index.to_list()
            reachesin_subunit['max_stage'] = reachesin_subunit['TotDASqKm'].apply(max_stage_equation)
            stages = np.array([np.linspace(0, max_stage_equation(dasqkm), 1000) for dasqkm in reachesin_subunit['TotDASqKm'].to_list()])

            su_data_dict = subunit_hydraulics(hand_path, aoi_path, slope_path, stages, reach_field=id_field, reaches=reach_list, fields_of_interest=fields_of_interest)

            for f in fields_of_interest:
                data_dict[f] = pd.concat([data_dict[f], su_data_dict[f]], axis=1)
        
        print('\n'*3)
        print(f'Completed processing {unit} in {round((time.perf_counter() - t1) / 60, 1)} minutes')
        print('='*50)

    print('Saving data')
    out_directory = os.path.join(out_directory, "outputs")
    os.makedirs(out_directory, exist_ok=True)
    for f in fields_of_interest:
            data_dict[f].to_csv(os.path.join(out_directory, f'{f}.csv'), index=False)
    print('Finished saving')


def batch_geomorphons(working_directory):
    run_list = ['WIN_0101']
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

def geomorphon_stats(reach_path, aoi_path, working_directory, out_directory, id_field):
     # Load reaches/basins to run
    reaches = pd.read_csv(reach_path, dtype={'subunit': 'str', 'unit': 'str'})
    units = reaches['unit'].unique()

    # Initialize logging
    data_dict = {f: pd.DataFrame() for f in fields_of_interest}

    # Process units
    for unit in units:
        reaches_in_unit = reaches.query(f'unit == "{unit}"')
        subunits = np.sort(reaches_in_unit['subunit'].unique())

        t1 = time.perf_counter()
        print(f'Unit: {unit} | Subunits: {len(subunits)}')
        counter = 1
        for subunit in subunits:
            print(f'subunit {counter}: {subunit}')
            counter += 1
            geomorphon_path = os.path.join(working_directory, unit, 'subbasins', subunit, 'rasters', 'slope.tif')
            su_data_dict = subunit_hydraulics(hand_path, aoi_path, slope_path, stages, reach_field=id_field, reaches=reach_list, fields_of_interest=fields_of_interest)

            for f in fields_of_interest:
                data_dict[f] = pd.concat([data_dict[f], su_data_dict[f]], axis=1)
        
        print('\n'*3)
        print(f'Completed processing {unit} in {round((time.perf_counter() - t1) / 60, 1)} minutes')
        print('='*50)

    print('Saving data')
    out_directory = os.path.join(out_directory, "outputs")
    os.makedirs(out_directory, exist_ok=True)
    for f in fields_of_interest:
            data_dict[f].to_csv(os.path.join(out_directory, f'{f}.csv'), index=False)
    print('Finished saving')
        


if __name__ == '__main__':
    reach_path = r"C:\Users\klawson1\Documents\CIROH_Floodplains\runs\6-6-23\homework_reaches.csv"
    aoi_path = r"C:\Users\klawson1\Documents\CIROH_Floodplains\runs\reference\reaches.shp"
    id_field = 'MergedCode'
    working_directory = r'C:\Users\klawson1\Documents\CIROH_Floodplains'
    out_directory = r'C:\Users\klawson1\Documents\CIROH_Floodplains\runs\6-6-23'
    fields_of_interest = ['rh_prime', 'el', 'vol', 'p', 'area', 'rh', 'celerity']

    # topographic_signatures(reach_path, aoi_path, working_directory, out_directory, id_field, fields_of_interest, scaling=True)

    working_directory = r'/netfiles/ciroh/floodplains'
    batch_geomorphons(working_directory)
