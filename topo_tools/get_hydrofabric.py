import requests
import os
import json
import shutil
import geopandas as gpd
import pandas as pd

NAME_DICT = {
    'MSQ': 'missisquoi',
    'LAM': 'lamoille',
    'WIN': 'winooski',
    'OTR': 'otter',
    'MET': 'mettawee',
    'LKC': 'champlain'
    }


def download_data(run_dict, region):
    raise RuntimeError('This function is not currently supported.')
    url = f'https://lynker-spatial.s3.amazonaws.com/hydrofabric/v20.1/gpkg/nextgen_{region}.gpkg'
    out_path = os.path.join(run_dict['network_directory'], f'nextgen_{region}.gpkg')

    print('Starting download')
    with requests.get(url, stream=True) as r:
        with open(out_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    assert r.status_code == 200, f'Download failed with status code {r.status_code}'

    return out_path


def download_data_aws(run_dict, region):
    raise RuntimeError('This function is not currently supported.')
    # DO NOT USE
    """ Including this in case lynker ever takes down the public bucket """
    s3 = boto3.client('s3')
    file_name = f'nextgen_{region}.zip'
    out_path = os.path.join(run_dict['network_directory'], file_name)
    with open(out_path, 'wb') as of:
        s3.download_fileobj('lynker-spatial', f'hydrofabric/v20.1/gpkg/nextgen_{region}.zip', of)
    return out_path


def clip_to_study_area(gpkg_path, run_dict):
    # Clip flowlines and subcatchments to catchments of interest
    print('Loading subbasins')
    subbasins = gpd.read_file(run_dict['subunit_path'])

    print('Loading Flowlines')
    flowpaths = gpd.read_file(gpkg_path, driver='GPKG', layer='reference_flowline', bbox=subbasins)
    flowpaths['length'] = flowpaths['slopelenkm'] * 1000
    flowpaths = flowpaths[['COMID', 'geometry', 'TotDASqKM', 'slope', 'length', 'StreamOrde']]
    flowpaths = flowpaths.rename(columns={"TotDASqKM": "TotDASqKm", "StreamOrde": "s_order"})
    flowpaths = flowpaths.to_crs(subbasins.crs)

    print('Intersecting layers')
    intersected = gpd.overlay(flowpaths, subbasins[['geometry', 'Code_name']], how='intersection')

    print('Joining Merge Codes')
    intersected = intersected.rename(columns={"COMID": run_dict["id_field"]})
    intersected['tmp_length'] = intersected.length  # only being used for subunit membership.  Not for slope calculation
    grouped = intersected.groupby([run_dict["id_field"], 'Code_name'])['tmp_length'].sum().reset_index()
    max_length_subgroup = grouped.loc[grouped.groupby(run_dict["id_field"])['tmp_length'].idxmax()]
    intersected = intersected.drop(['Code_name'], axis=1)
    intersected = intersected.merge(max_length_subgroup[[run_dict["id_field"], 'Code_name']], on=run_dict["id_field"], how='left')
    intersected = intersected.merge(subbasins[[c for c in subbasins.columns if c not in ['geometry', 'AreaSqKm']]], on='Code_name', how='left')
    intersected = intersected.drop(['tmp_length'], axis=1)

    print('Saving to file')
    intersected.to_file(run_dict['flowline_path'])
    intersected = intersected.drop(['geometry'], axis=1)
    
    print('Loading NHD Catchments')
    catchments = gpd.read_file(gpkg_path, driver='GPKG', layer='reference_catchment', bbox=subbasins)
    catchments = catchments[['FEATUREID', 'geometry']]
    catchments = catchments.rename(columns={"FEATUREID": run_dict["id_field"]})

    print('Subsetting catchments')
    catchments = catchments.merge(intersected, how='inner', on=run_dict["id_field"])

    print('Saving to file')
    catchments.to_file(run_dict['reach_path'])

    print('Generating reach metadata table')
    catchments = catchments.drop(['geometry'], axis=1)
    catchments[['8_code', run_dict['subunit_field']]] = catchments['Code_name'].str.split('_', expand=True)
    catchments[run_dict['unit_field']] = catchments['8_code'].map(NAME_DICT)
    catchments = catchments.drop_duplicates(subset=[run_dict["id_field"]])
    catchments = catchments.sort_values(by=[run_dict["id_field"]])
    catchments.to_csv(run_dict['reach_meta_path'], index=False)


def run_all(meta_path):
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    # data_path = download_data(meta, '02')
    data_path = r"/netfiles/ciroh/floodplainsData/runs/hydrofabric/network/reference_02.gpkg"
    clip_to_study_area(data_path, meta)



if __name__ == '__main__':
    meta_path = r'/netfiles/ciroh/floodplainsData/runs/hydrofabric/run_metadata.json'
    run_all(meta_path)