import os
import time
import pandas as pd
import numpy as np
import json
from utilities import subunit_hydraulics


def topographic_signatures(reach_path, aoi_path, working_directory, id_field, fields_of_interest, scaling):
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

    # Load reaches/basins to run
    reaches = pd.read_csv(reach_path, dtype={'subunit': 'str', 'unit': 'str'})
    units = reaches['unit'].unique()

    # Process units
    for unit in units:
        reaches_in_unit = reaches.query(f'unit == "{unit}"')
        subunits = np.sort(reaches_in_unit['subunit'].unique())

        # Set up data logging
        data_dict = {f: pd.DataFrame() for f in fields_of_interest}
        out_path = os.path.join(working_directory, unit, 'reach_summaries')
        os.makedirs(out_path, exist_ok=True)

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
            reachesin_subunit = reachesin_subunit.groupby(reachesin_subunit[id_field]).agg(TotDASqKm=('TotDASqKm', 'max'), Slope=('Slope', 'mean'))
            reach_list = reachesin_subunit.index.to_list()
            reachesin_subunit['max_stage'] = reachesin_subunit['TotDASqKm'].apply(max_stage_equation)
            stages = np.array([np.linspace(0, max_stage_equation(dasqkm), 1000) for dasqkm in reachesin_subunit['TotDASqKm'].to_list()])

            su_data_dict = subunit_hydraulics(hand_path, aoi_path, slope_path, stages, reach_field=id_field, reaches=reach_list, fields_of_interest=fields_of_interest)

            for f in fields_of_interest:
                data_dict[f] = pd.concat([data_dict[f], su_data_dict[f]], axis=1)
    
        for f in fields_of_interest:
            data_dict[f].to_csv(os.path.join(out_path, f'{f}.csv'), index=False)
        with open(os.path.join(out_path, 'run_metadata.json'), 'w') as f:
            json.dump(run_metadata, f)
        
        print('\n'*3)
        print(f'Completed processing {unit} in {round((time.perf_counter() - t1) / 60, 1)} minutes')
        print('='*50)


if __name__ == '__main__':
    reach_path = r"reaches.csv"
    aoi_path = r"reaches.shp"
    id_field = 'MergedCode'
    working_directory = os.curdir()
    fields_of_interest = ['el', 'vol', 'p', 'area', 'rh', 'celerity']

    topographic_signatures(reach_path, aoi_path, working_directory, id_field, fields_of_interest, scaling=True)
