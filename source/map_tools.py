import json
import sys
import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from utilities import build_raster, merge_rasters, merge_polygons


def map_edzs(meta_path):
    # Load run config
    with open(meta_path, 'r') as f:
        run_dict = json.loads(f.read())

    # Load EDZ data
    feature_data = pd.read_csv(run_dict['analysis_path'])
    feature_data[run_dict['id_field']] = feature_data[run_dict['id_field']].astype(str)
    feature_data = feature_data.set_index(run_dict['id_field'])

    # Load reaches/basins to run
    reaches = pd.read_csv(run_dict['reach_meta_path'])
    reaches[run_dict['id_field']] = reaches[run_dict['id_field']].astype(str)
    reaches[run_dict['subunit_field']] = reaches[run_dict['subunit_field']].astype(str).str.rjust(4, '0')
    reaches = reaches.set_index(run_dict['id_field'])
    units = reaches[run_dict['unit_field']].unique()

    # Process units
    out_rasters = list()
    out_polys = list()
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
            reach_list = list(reachesin_subunit.index)

            reaches_in_subunit = feature_data[feature_data.index.isin(reach_list)]
            if len(reaches_in_subunit) == 0:
                continue
        
            reach_data = reaches_in_subunit[['el_edap', 'el_edep']].copy()
            reach_dict = dict()
            for r in reach_data.index:
                if not np.isnan(reach_data.loc[r, 'el_edap']) and not np.isnan(reach_data.loc[r, 'el_edep']):
                    ch_dict = {'label': 'channel', 'min_el': 0, 'max_el': reach_data.loc[r, 'el_edap']}
                    edz_dict = {'label': 'edz', 'min_el': reach_data.loc[r, 'el_edap'], 'max_el': reach_data.loc[r, 'el_edep']}
                    reach_dict[r] = {'ID': r, 'zones': [ch_dict, edz_dict]}

            build_raster(hand_path, run_dict["reach_path"], run_dict["id_field"], reach_dict, 'edz')

            out_raster_path = os.path.join(os.path.dirname(hand_path), 'edz.tif')
            out_rasters.append(out_raster_path)
            out_poly_path = os.path.join(os.path.dirname(os.path.dirname(hand_path)), 'vectors', 'edz.shp')
            out_polys.append(out_poly_path)
        print('\n'*3)
        print(f'Completed processing {unit} in {round((time.perf_counter() - t1) / 60, 1)} minutes')
        print('='*50)

    out_path = os.path.join(run_dict['analysis_directory'], 'edz.tif')
    merge_rasters(out_rasters, out_path)
    out_path = os.path.join(run_dict['analysis_directory'], 'edz.shp')
    merge_polygons(out_polys, out_path)

def map_floodplain(meta_path, magnitude):
    # Load run config
    with open(meta_path, 'r') as f:
        run_dict = json.loads(f.read())

    # Load flood data
    flood_data = pd.read_csv(os.path.join(run_dict['analysis_directory'], 'flood_metrics.csv'))
    flood_data[run_dict['id_field']] = flood_data[run_dict['id_field']].astype(str)
    flood_data = flood_data.set_index(run_dict['id_field'])

    # Load reaches/basins to run
    reaches = pd.read_csv(run_dict['reach_meta_path'])
    reaches[run_dict['id_field']] = reaches[run_dict['id_field']].astype(str)
    reaches[run_dict['subunit_field']] = reaches[run_dict['subunit_field']].astype(str).str.rjust(4, '0')
    reaches = reaches.set_index(run_dict['id_field'])
    units = reaches[run_dict['unit_field']].unique()

    # Process units
    out_rasters = list()
    out_polys = list()
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
            reach_list = list(reachesin_subunit.index)

            reaches_in_subunit = flood_data[flood_data.index.isin(reach_list)]
            if len(reaches_in_subunit) == 0:
                continue
            reach_data = reaches_in_subunit[[f'{magnitude}_Medium_peak_stage']]
            reach_dict = dict()
            for r in reach_data.index:
                mag_dict = {'label': magnitude, 'min_el': 0, 'max_el': reach_data.loc[r, f'{magnitude}_Medium_peak_stage']}
                reach_dict[r] = {'ID': r, 'zones': [mag_dict]}
            

            build_raster(hand_path, run_dict["reach_path"], run_dict["id_field"], reach_dict, magnitude)

            out_raster_path = os.path.join(os.path.dirname(hand_path), f'{magnitude}.tif')
            out_rasters.append(out_raster_path)
            out_poly_path = os.path.join(os.path.dirname(os.path.dirname(hand_path)), 'vectors', f'{magnitude}.shp')
            out_polys.append(out_poly_path)

        print('\n'*3)
        print(f'Completed processing {unit} in {round((time.perf_counter() - t1) / 60, 1)} minutes')
        print('='*50)

    out_path = os.path.join(run_dict['analysis_directory'], f'{magnitude}.tif')
    merge_rasters(out_rasters, out_path)
    out_path = os.path.join(run_dict['analysis_directory'], f'{magnitude}.shp')
    merge_polygons(out_polys, out_path)



def merge_subbasins(meta_path, file_name):
    # Load run config
    with open(meta_path, 'r') as f:
        run_dict = json.loads(f.read())

    print('Finding subbasin EDZ data...')
    # Load reaches/basins to run
    reaches = gpd.read_file(run_dict['reach_path'], ignore_geometry=True)
    units = reaches[run_dict['unit_field']].unique()

    # Process paths
    poly_paths = list()
    raster_paths = list()
    for unit in units:
        reaches_in_unit = reaches[reaches[run_dict["unit_field"]] == unit]
        subunits = np.sort(reaches_in_unit[run_dict['subunit_field']].unique())
        for subunit in subunits:
            shp_path = os.path.join(run_dict['data_directory'], unit, 'subbasins', subunit, 'vectors', f'{file_name}.shp')
            if os.path.exists(shp_path):
                poly_paths.append(shp_path)
            tif_path = os.path.join(run_dict['data_directory'], unit, 'subbasins', subunit, 'rasters', f'{file_name}.tif')
            if os.path.exists(tif_path):
                raster_paths.append(tif_path)
    
    print('Merging EDZ data...')
    out_path = os.path.join(run_dict['analysis_directory'], f'{file_name}.tif')
    merge_rasters(raster_paths, out_path)
    out_path = os.path.join(run_dict['analysis_directory'], f'{file_name}.shp')
    merge_polygons(poly_paths, out_path)


if __name__ == '__main__':
    meta_path = sys.argv[1]
    for m in ['Q2', 'Q5', 'Q10','Q25', 'Q50', 'Q100', 'Q200', 'Q500']:
        map_floodplain(meta_path, m)
    map_edzs(meta_path)
