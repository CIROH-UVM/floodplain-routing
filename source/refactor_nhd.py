import requests
import subprocess
import shutil
import zipfile
import os
import sys
import sqlite3
import geopandas as gpd
import pandas as pd
import json


OGR_PATH = os.path.join(sys.prefix, 'bin', 'ogr2ogr')
MANUAL_OVERRIDE = {'60000200057445': '4300102000489',
                   '60000200063685': '4300102000489'}

def download_data(run_dict):
    url = f'https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHDPlusHR/Beta/GDB/NHDPLUS_H_{run_dict["HUC4"]}_HU4_GDB.zip'
    zip_path = os.path.join(run_dict['network_directory'], 'NHD.zip')
    unzip_path = os.path.join(run_dict['network_directory'], 'NHD')

    print('Starting download')
    r = requests.get(url, stream=True)
    with requests.get(url, stream=True) as r:
        with open(zip_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    assert r.status_code == 200, f'Download failed with status code {r.status_code}'

    print('Download finished, unzipping')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:        
        zip_ref.extractall(unzip_path)
    
    out_path = [f for f in os.listdir(unzip_path) if f[-3:] == 'gdb'][0]
    out_path = os.path.join(unzip_path, out_path)
    return out_path


def merge_reaches(gdb_path, run_dict):
    # merge small DA reaches with d/s reachcode
    print('Exporting VAA table from GDB')
    subprocess.run([OGR_PATH, '-f', 'SQLite', run_dict['network_db_path'], gdb_path, 'NHDPlusFlowlineVAA'])

    in_file = open(os.path.join(os.path.dirname(__file__), 'generate_mainstems.sql'), 'r')
    mainstems = in_file.read().split(';')
    in_file.close()

    in_file = open(os.path.join(os.path.dirname(__file__), 'refactor_nhd.sql'), 'r')
    refactor = in_file.read().split(';')
    in_file.close()

    print('Merging reaches and gathering metadata')
    conn = sqlite3.connect(run_dict['network_db_path'])
    c = conn.cursor()

    [c.execute(q) for q in mainstems]
    [c.execute(q) for q in refactor]
    conn.commit()

    c.execute('DROP TABLE IF EXISTS reach_data')
    c.execute('UPDATE nhdplusflowlinevaa SET slopelenkm=0 WHERE slopelenkm<0')
    c.execute('CREATE TABLE reach_data (ReachCode REAL, TotDASqKm REAL, max_el REAL, min_el REAL, length REAL, slope REAL GENERATED ALWAYS AS (CASE WHEN ((max_el-min_el)/(length*100)) = 0 THEN 0.00001 ELSE ((max_el-min_el)/(length*100)) END) STORED)')
    c.execute('INSERT INTO reach_data (ReachCode, TotDASqKm, min_el, max_el, length) SELECT reachcode, max(TotDASqKm), min(minelevsmo), max(maxelevsmo), sum(slopelenkm)*1000 FROM nhdplusflowlinevaa GROUP BY reachcode')
    conn.commit()


def clip_to_study_area(gdb_path, run_dict):
    # Clip flowlines and subcatchments to catchments of interest
    print('Loading Merged VAA table')
    conn = sqlite3.connect(run_dict['network_db_path'])
    merged = pd.read_sql_query("SELECT * from merged", conn)
    meta = pd.read_sql_query("SELECT * from reach_data", conn)

    print('Loading subbasins')
    subbasins = gpd.read_file(run_dict['subunit_path'])

    print('Loading NHD Flowlines')
    nhd = gpd.read_file(gdb_path, driver='OpenFileGDB', layer='NHDFlowline')
    nhd = nhd[['NHDPlusID', 'geometry']]

    print('Intersecting layers')
    intersected = gpd.overlay(nhd, subbasins[['geometry', 'Code_name']], how='intersection')

    print('Joining Merge Codes')
    intersected = intersected.merge(merged, on='NHDPlusID', how='inner')
    intersected = intersected.merge(meta[['ReachCode', 'TotDASqKm', 'slope']], on='ReachCode', how='left')
    intersected = intersected.rename(columns={"ReachCode": "MergeCode"})
    intersected['length'] = intersected.length  # only being used for subunit membership.  Not for slope calculation
    grouped = intersected.groupby(['MergeCode', 'Code_name'])['length'].sum().reset_index()
    max_length_subgroup = grouped.loc[grouped.groupby('MergeCode')['length'].idxmax()]
    intersected = intersected.drop(['Code_name'], axis=1)
    intersected = intersected.merge(max_length_subgroup[['MergeCode', 'Code_name']], on='MergeCode', how='left')
    intersected = intersected.merge(subbasins[[c for c in subbasins.columns if c not in ['geometry', 'AreaSqKm']]], on='Code_name', how='left')
    intersected = intersected.drop(['length'], axis=1)

    print('Saving to file')
    intersected.to_file(run_dict['flowline_path'])
    intersected = intersected.drop(['geometry'], axis=1)
    
    print('Loading NHD Catchments')
    nhd = gpd.read_file(gdb_path, driver='OpenFileGDB', layer='NHDPlusCatchment')
    nhd = nhd[['NHDPlusID', 'geometry']]

    print('Subsetting catchments')
    nhd = nhd.merge(intersected, how='inner', on='NHDPlusID')

    print('Saving to file')
    nhd.to_file(run_dict['reach_path'])

    print('Generating reach metadata table')
    meta = meta[meta['ReachCode'].isin(nhd['MergeCode'].unique())]
    meta = meta.merge(nhd[['MergeCode', 'Code_name']], how='left', left_on='ReachCode', right_on='MergeCode')
    meta = meta.drop('MergeCode', axis=1)
    meta.to_csv(run_dict['reach_meta_path'], index=False)

def run_all(meta_path, update_metadata=True):
    # Load run config and initialize directory structure
    with open(meta_path, 'r') as f:
        run_dict = json.loads(f.read())
    if update_metadata:
        run_dict['network_directory'] = os.path.join(run_dict['run_directory'], 'network')
        run_dict['network_db_path'] = os.path.join(run_dict['network_directory'], 'vaa.db')
        run_dict['reach_path'] = os.path.join(run_dict['network_directory'], 'catchments.shp')
        run_dict['flowline_path'] = os.path.join(run_dict['network_directory'], 'flowlines.shp')
        run_dict['reach_meta_path'] = os.path.join(run_dict['network_directory'], 'reach_data.csv')
        with open(meta_path, 'w') as f:
            json.dump(run_dict, f)
    os.makedirs(run_dict['network_directory'], exist_ok=True)

    gdb_path = download_data(run_dict)
    merge_reaches(gdb_path, run_dict)
    clip_to_study_area(gdb_path, run_dict)

    print('Cleaning up')
    shutil.rmtree(os.path.join(run_dict['network_directory'], 'NHD'))
    os.remove(os.path.join(run_dict['network_directory'], 'NHD.zip'))


if __name__ == '__main__':
    meta_path = r'/netfiles/ciroh/floodplainsData/runs/4/run_metadata.json'
    run_all(meta_path)