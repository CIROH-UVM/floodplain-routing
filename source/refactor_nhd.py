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
from collections import defaultdict

OGR_PATH = os.path.join(sys.prefix, 'bin', 'ogr2ogr')
MANUAL_OVERRIDE = {'60000200057445': '4300102000489',
                   '60000200063685': '4300102000489'}
NAME_DICT = {
    'MSQ': 'missisquoi',
    'LAM': 'lamoille',
    'WIN': 'winooski',
    'OTR': 'otter',
    'MET': 'mettawee',
    'LKC': 'champlain'
    }

def download_data(run_dict):
    # url = f'https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHDPlusHR/Beta/GDB/NHDPLUS_H_{run_dict["huc4"]}_HU4_GDB.zip'
    url = f'https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHDPlusHR/VPU/Current/GDB/NHDPLUS_H_{run_dict["huc4"]}_HU4_GDB.zip'
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


def merge_reaches(gdb_path, run_dict, merge_short=False):
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

    if merge_short:
        print('Merging short reaches')
        merge_short_reaches(conn, c)

    c.execute('DROP TABLE IF EXISTS reach_data')
    c.execute('UPDATE nhdplusflowlinevaa SET slopelenkm=0 WHERE slopelenkm<0')
    c.execute('CREATE TABLE reach_data (ReachCode REAL, TotDASqKm REAL, max_el REAL, min_el REAL, length REAL, s_order INTEGER, slope REAL GENERATED ALWAYS AS (CASE WHEN ((max_el-min_el)/(length*100)) = 0 THEN 0.00001 ELSE ((max_el-min_el)/(length*100)) END) STORED)')
    c.execute('INSERT INTO reach_data (ReachCode, TotDASqKm, min_el, max_el, length, s_order) SELECT uvm.reachcode, max(nhd.totdasqkm), min(nhd.minelevsmo), max(nhd.maxelevsmo), sum(nhd.slopelenkm)*1000, max(nhd.streamorde) FROM nhdplusflowlinevaa nhd JOIN merged uvm ON uvm.nhdplusid = nhd.nhdplusid WHERE nhd.mainstem=1 GROUP BY uvm.reachcode')
    conn.commit()

def merge_short_reaches(conn, c, length_threshold=500):
    # Traverse the network in a postorder fashion to identify reaches with length less than 300 meters and merges them with the downstream reach
    mainstems = pd.read_sql_query("SELECT fromnode, tonode, reachcode, slopelenkm from nhdplusflowlinevaa where mainstem = 1", conn)
    mainstems['fromnode'] = mainstems['fromnode'].astype(int).astype(str)
    mainstems['tonode'] = mainstems['tonode'].astype(int).astype(str)
    mainstems['reachcode'] = mainstems['reachcode'].astype(int).astype(str)
    # tonode is d/s end of reach, fromnode is u/s end of reach
    mainstems['length'] = mainstems['slopelenkm'] * 1000
    mainstems = mainstems.drop(['slopelenkm'], axis=1)

    # refactor to reachcode level and create new dictionary versions of the network
    edge_dict = defaultdict(list)
    meta_dict = dict()
    reachcodes = mainstems['reachcode'].unique()
    all_nodes = mainstems['fromnode'].unique()
    q = list()
    for r in reachcodes:
        tmp = mainstems[mainstems['reachcode'] == r]
        froms = set(tmp['fromnode'])
        tos = set(tmp['tonode'])
        us = froms.difference(tos).pop()
        ds = tos.difference(froms).pop()
        if ds not in all_nodes:
            q.append(r)
            meta_dict[r] = {'l': tmp_length}
        else:
            ds = mainstems[mainstems['fromnode'] == ds]['reachcode'].values[0]
            tmp_length = tmp['length'].sum()
            edge_dict[ds].append(r)
            meta_dict[r] = {'l': tmp_length, 'ds': ds}

    # Traverse the network in postorder
    conversions = dict()
    visited = set()
    merge_length = 0
    merge_ids = list()
    while q:
        cur_node = q[-1]
        children = edge_dict[cur_node]
        if all([n in visited for n in children]) or all([n not in edge_dict for n in children]):
            tmp_l = meta_dict[cur_node]['l']
            # Check for reaches in the stack
            if len(merge_ids) == 0:
                if tmp_l < length_threshold:
                    merge_length += tmp_l
                    merge_ids.append(cur_node)
            else:
                # Check if the reach is short or a confluence
                if tmp_l + merge_length >= length_threshold or len(children) > 1:
                    conversions[cur_node] = merge_ids
                    merge_length = 0
                    merge_ids = list()
                else:
                    merge_length += tmp_l
                    merge_ids.append(cur_node)        
            q.pop()
            visited.add(cur_node)
            continue
        for n in children:
            q.append(n)

    # Merge short reaches
    for k, v in conversions.items():
        c.execute(f"UPDATE merged SET ReachCode={k} WHERE ReachCode IN ({','.join([str(i) for i in v])})")
    conn.commit()

def clip_to_study_area(gdb_path, run_dict, water_toggle=0.05):
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
    intersected = intersected.rename(columns={"ReachCode": run_dict["id_field"]})
    intersected['length'] = intersected.length  # only being used for subunit membership.  Not for slope calculation
    grouped = intersected.groupby([run_dict["id_field"], 'Code_name'])['length'].sum().reset_index()
    max_length_subgroup = grouped.loc[grouped.groupby(run_dict["id_field"])['length'].idxmax()]
    intersected = intersected.drop(['Code_name'], axis=1)
    intersected = intersected.merge(max_length_subgroup[[run_dict["id_field"], 'Code_name']], on=run_dict["id_field"], how='left')
    intersected = intersected.merge(subbasins[[c for c in subbasins.columns if c not in ['geometry', 'AreaSqKm']]], on='Code_name', how='left')
    
    print('Checking waterbody intersections')
    wbodies = gpd.read_file(gdb_path, driver='OpenFileGDB', layer='NHDWaterbody')
    wbodies = wbodies[wbodies['AreaSqKm'] > 0.05]
    wbody_intersect = gpd.overlay(intersected, wbodies[['geometry', 'NHDPlusID']], how='intersection')
    wbody_intersect['w_length'] = wbody_intersect.length
    intersected['length'] = intersected.length
    wbody_intersect = wbody_intersect.groupby([run_dict["id_field"]])['w_length'].sum().reset_index()
    old_sums = intersected.groupby([run_dict["id_field"]])['length'].sum().reset_index()
    wbody_intersect = wbody_intersect.merge(old_sums, on=run_dict["id_field"], how='left')
    wbody_intersect['pct_water'] = wbody_intersect['w_length'] / wbody_intersect['length']
    wbody_intersect['wbody'] = wbody_intersect['pct_water'] > water_toggle
    intersected = intersected.merge(wbody_intersect[[run_dict["id_field"], 'wbody']], on=run_dict["id_field"], how='left')
    
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
    meta = meta[meta['ReachCode'].isin(nhd[run_dict["id_field"]].unique())]
    subunits = nhd[[run_dict["id_field"], 'Code_name']].drop_duplicates()
    meta = meta.merge(subunits, how='left', left_on='ReachCode', right_on=run_dict["id_field"])
    meta[['8_code', run_dict['subunit_field']]] = meta['Code_name'].str.split('_', expand=True)
    meta[run_dict['unit_field']] = meta['8_code'].map(NAME_DICT)
    meta = meta.merge(wbody_intersect[[run_dict["id_field"], 'wbody']], on=run_dict["id_field"], how='left').fillna(False)
    meta.to_csv(run_dict['reach_meta_path'], index=False)

def run_all(meta_path):
    # Load run config and initialize directory structure
    with open(meta_path, 'r') as f:
        run_dict = json.loads(f.read())

    gdb_path = download_data(run_dict)
    merge_reaches(gdb_path, run_dict, merge_short=True)
    clip_to_study_area(gdb_path, run_dict)

    print('Cleaning up')
    shutil.rmtree(os.path.join(run_dict['network_directory'], 'NHD'))
    os.remove(os.path.join(run_dict['network_directory'], 'NHD.zip'))


if __name__ == '__main__':
    meta_path = r'/netfiles/ciroh/floodplainsData/runs/8/run_metadata.json'
    run_all(meta_path)