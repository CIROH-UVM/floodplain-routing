import os
import sys
import json
import argparse

### PARSING ###
parser = argparse.ArgumentParser(description='Make a run metadata file from the template.')
parser.add_argument('data_directory', type=str, help='Path to folder containing all data.')
parser.add_argument('run_id', type=str, help='Unique ID for this run. (str)')

def make_run_template(base_directory='/path/to/data', run_id='1'):
    run_metadata = {
        "data_directory": base_directory,  # Directory containing all data
        "run_directory": os.path.join(base_directory, 'runs', run_id),  # Subdirectory for this run
        "id_field": "UVM_ID",  # Field used in several files to uniquely identify each reach
        "unit_field": "8_name",  # Field used to identify the first level of batch processing.  Typically the HUC8.
        "subunit_field": "12_code",  # Field used to identify the second level of batch processing.  Typically the HUC12.
        "scaled_stages": True,  # Whether to use 6xbankfull as the maximum stage (true) or 10 meters (false)
        "huc4": "0430",  # HUC4 code for this run.  Used to download NHD data.
        "subunit_path": os.path.join(base_directory, 'shared', 'subunits.shp'),  # Path to shapefile containing subunit boundaries.
        "geometry_source": 'HAND'  # Source of geometry data.  Either 'HAND' or 'DEM'.
        }

    os.makedirs(run_metadata['run_directory'], exist_ok=True)
    meta_path = os.path.join(run_metadata['run_directory'], 'run_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(run_metadata, f, indent=4)

    return meta_path

if __name__ == '__main__':
    args = parser.parse_args()
    make_run_template(args.data_directory, args.run_id)