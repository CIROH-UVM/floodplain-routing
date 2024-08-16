import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))  # Make sure root directory is in PYTHONPATH.  Not necessary, depending on how you set up your environment.
from topo_tools import *

# Create the configuration file
data_dir = os.path.join(os.getcwd(), 'samples')
run_id = 'test2'
meta_path = make_run_template(data_dir, run_id)

# Get reach delineations and metadata
get_reaches(meta_path, 1000)

# Extract geometry
extract_geometry(meta_path)
batch_add_bathymetry(meta_path)

# Extract features
extract_features(meta_path, plot=True)

# Analyze floods
analyze_floods(meta_path)

# Map floodplain
for m in ['Q2', 'Q5', 'Q10','Q25', 'Q50', 'Q100', 'Q200', 'Q500']:
    map_floodplain(meta_path, m)
map_edzs(meta_path)