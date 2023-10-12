import requests
import subprocess
import shutil
import zipfile
import os
import sys
import sqlite3
import geopandas as gpd
import pandas as pd


OGR_PATH = os.path.join(sys.prefix, 'bin', 'ogr2ogr')

def download_data(save_dir, huc4):
    url = f'https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHDPlusHR/Beta/GDB/NHDPLUS_H_{huc4}_HU4_GDB.zip'
    zip_path = os.path.join(save_dir, 'NHD.zip')
    unzip_path = os.path.join(save_dir, 'NHD')

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


def merge_reaches(gdb_path, out_path):
    # merge small DA reaches with d/s reachcode
    print('Exporting VAA table from GDB')
    subprocess.run([OGR_PATH, '-f', 'SQLite', out_path, gdb_path, 'NHDPlusFlowlineVAA'])

    in_file = open(os.path.join(os.path.dirname(__file__), 'refactor_nhd.sql'), 'r')
    refactor = in_file.read()
    in_file.close()

    print('Merging reaches and gathering metadata')
    conn = sqlite3.connect(out_path)
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS merged')
    c.execute('CREATE TABLE merged (NHDPlusID REAL, ReachCode REAL);')
    c.execute(refactor)
    c.execute('DELETE FROM merged WHERE rowid NOT IN (SELECT MIN(rowid) FROM  merged GROUP BY NHDPlusID)')
    conn.commit()

    c.execute('DROP TABLE IF EXISTS reach_data')
    c.execute('CREATE TABLE reach_data (ReachCode REAL, TotDASqKm REAL, max_el REAL, min_el REAL, length REAL, slope REAL GENERATED ALWAYS AS ((max_el-min_el)/(length*100)))')
    c.execute('INSERT INTO reach_data (ReachCode, TotDASqKm, min_el, max_el, length) SELECT reachcode, max(TotDASqKm), min(minelevsmo), max(maxelevsmo), sum(slopelenkm)*1000 FROM nhdplusflowlinevaa GROUP BY reachcode')
    conn.commit()


def clip_flowlines(clip_path, gdb_path, db_path, out_dir):
    # Clip flowlines and subcatchments to catchments of interest
    print('Loading Merged VAA table')
    conn = sqlite3.connect(db_path)
    merged = pd.read_sql_query("SELECT * from merged", conn)
    meta = pd.read_sql_query("SELECT * from reach_data", conn)

    print('Loading subbasins')
    subbasins = gpd.read_file(clip_path)

    print('Loading NHD Flowlines')
    nhd = gpd.read_file(gdb_path, driver='OpenFileGDB', layer='NHDFlowline')
    nhd = nhd[['NHDPlusID', 'geometry']]

    print('Intersecting layers')
    intersected = gpd.overlay(nhd, subbasins, how='intersection')

    print('Joining Merge Codes')
    intersected = intersected.merge(merged, on='NHDPlusID', how='left')
    intersected = intersected.merge(meta[['ReachCode', 'TotDASqKm']], on='ReachCode', how='left')
    intersected = intersected.rename(columns={"ReachCode": "MergeCode"})

    print('Saving to file')
    intersected.to_file(os.path.join(out_dir, 'flowlines.shp'))
    intersected = intersected.drop(['geometry'], axis=1)
    
    print('Loading NHD Catchments')
    nhd = gpd.read_file(gdb_path, driver='OpenFileGDB', layer='NHDPlusCatchment')
    nhd = nhd[['NHDPlusID', 'geometry']]

    print('Subsetting catchments')
    nhd = nhd.merge(intersected, how='inner', on='NHDPlusID')

    print('Saving to file')
    nhd.to_file(os.path.join(out_dir, 'catchments.shp'))


def run_all():
    save_path = r'/users/k/l/klawson1/netfiles/ciroh/slawson/ciroh_network'
    db_path = os.path.join(save_path, 'vaa.db')
    clip_path = os.path.join(save_path, 'subunits.shp')
    huc4 = '0430'

    gdb_path = download_data(save_path, huc4)
    merge_reaches(gdb_path, db_path)
    clip_flowlines(clip_path, gdb_path, db_path, save_path)

    print('Cleaning up')
    shutil.rmtree(os.path.join(save_path, 'NHD'))
    os.remove(os.path.join(save_path, 'NHD.zip'))


if __name__ == '__main__':
    run_all()