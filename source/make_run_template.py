import os
import sys
import json


def make_run_template(base_directory='/path/to/data', run_id='1'):
    run_metadata = {
        "data_directory": base_directory,
        "run_directory": os.path.join(base_directory, 'runs', run_id),
        "id_field": "UVM_ID",
        "unit_field": "8_name",
        "subunit_field": "12_code",
        "scaled_stages": True,
        "huc4": "0430",
        "subunit_path": os.path.join(base_directory, 'shared', 'subunits.shp'),
        "geometry_source": 'HAND'
        }

    os.makedirs(run_metadata['run_directory'], exist_ok=True)
    meta_path = os.path.join(run_metadata['run_directory'], 'run_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(run_metadata, f, indent=4)


if __name__ == '__main__':
    base_directory = sys.argv[1]  # Folder of data directory
    run_id = sys.argv[2]  # unique identifier for run 
    make_run_template(base_directory, run_id)