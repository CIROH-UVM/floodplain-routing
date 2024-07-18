import json
import sys
import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from utilities import map_edz, merge_rasters


def map_edzs(meta_path):
    # Load run config
    with open(meta_path, 'r') as f:
        run_dict = json.loads(f.read())

    # Load EDZ data
    feature_data = pd.read_csv(run_dict['analysis_path'])
    feature_data[run_dict['id_field']] = feature_data[run_dict['id_field']].astype(str)

    # Load reaches/basins to run
    reaches = gpd.read_file(run_dict['reach_path'], ignore_geometry=True)
    # reaches = reaches[reaches[run_dict['id_field']].isin(feature_data['ReachCode'])]
    units = reaches[run_dict['unit_field']].unique()

    # Process units
    out_paths = list()
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

            reachesin_subunit = reaches_in_unit[reaches_in_unit[run_dict["subunit_field"]] == subunit]
            reach_list = reachesin_subunit[run_dict["id_field"]].unique()

            reaches_in_subunit = feature_data[feature_data[run_dict["id_field"]].isin(reach_list)]
            reach_data = reaches_in_subunit[['el_edap', 'el_edep']].copy()
            if len(reaches_in_subunit) == 0:
                continue
            reach_data['el_edap'] = reach_data['el_edap'] - reaches_in_subunit['el_bathymetry']
            reach_data['el_edep'] = reach_data['el_edep'] - reaches_in_subunit['el_bathymetry']
            reach_data[run_dict["id_field"]] = reaches_in_subunit[run_dict["id_field"]]

            out_raster_path = map_edz(hand_path, run_dict["reach_path"], run_dict["id_field"], reach_data)

            out_raster_path = os.path.join(os.path.dirname(hand_path), 'edz.tif')
            out_paths.append(out_raster_path)
        print('\n'*3)
        print(f'Completed processing {unit} in {round((time.perf_counter() - t1) / 60, 1)} minutes')
        print('='*50)

    merge_rasters(out_paths, run_dict['edz_path'])


if __name__ == '__main__':
    meta_path = sys.argv[1]
    map_edzs(meta_path)