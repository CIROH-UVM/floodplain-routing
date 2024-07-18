import os
import sys
import json


def make_run_template(base_directory='/path/to/data', run_id='1'):
    run_metadata = {
        "data_directory": base_directory,
        "run_directory": os.path.join(base_directory, 'runs', run_id),

        "analysis_directory": os.path.join(base_directory, 'runs', run_id, 'analysis'),
        "analysis_path": os.path.join(base_directory, 'runs', run_id, 'analysis', 'data.csv'),
        "edz_path": os.path.join(base_directory, 'runs', run_id, 'analysis', 'edz.tif'),

        "muskingum_directory": os.path.join(base_directory, 'runs', run_id, 'muskingum-cunge'),
        "muskingum_path": os.path.join(base_directory, 'runs', run_id, 'muskingum-cunge', 'mc_data.csv'),
        "muskingum_diagnostics": os.path.join(base_directory, 'runs', run_id, 'muskingum-cunge', 'diagnostics'),

        "network_directory": os.path.join(base_directory, 'runs', run_id, 'network'),
        "network_db_path": os.path.join(base_directory, 'runs', run_id, 'network', 'vaa.db'),
        "reach_path": os.path.join(base_directory, 'runs', run_id, 'network', 'catchments.shp'),
        "flowline_path": os.path.join(base_directory, 'runs', run_id, 'network', 'flowlines.shp'),
        "reach_meta_path": os.path.join(base_directory, 'runs', run_id, 'network', 'reach_data.csv'),

        "geometry_directory": os.path.join(base_directory, 'runs', run_id, 'geometry'),
        "geometry_diagnostics": os.path.join(base_directory, 'runs', run_id, 'geometry', 'diagnostics'),

        "id_field": "UVM_ID",
        "unit_field": "8_name",
        "subunit_field": "12_code",
        "fields_of_interest": ['area', 'el', 'p', 'rh', 'rh_prime', 'vol'],
        "scaled_stages": True,
        "huc4": "0430",
        "subunit_path": os.path.join(base_directory,'shared', 'subunits.shp'),
        "geometry_source": 'HAND'
        }

    for key in run_metadata:
        path = str(run_metadata[key])
        if not base_directory in path:
            continue
        last = os.path.split(path)[-1]
        if '.' in last:
            continue
        os.makedirs(run_metadata[key], exist_ok=True)
    meta_path = os.path.join(run_metadata['run_directory'], 'run_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(run_metadata, f)


if __name__ == '__main__':
    base_directory = sys.argv[1]
    run_id = sys.argv[2]
    make_run_template(base_directory, run_id)