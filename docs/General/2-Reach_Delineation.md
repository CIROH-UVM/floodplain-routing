# Refactoring NHDPlus-HR

The UVM workflow utilizes the NHDPlus-HR dataset for reach direct-drainage dilineations and metadata (slope, drainage area, etc).  To run the automated workflow, you'll need to know the HUC4 code for your study area and include that in your configuration file ([Tutorial 1](1-Metadata_File.md)).  You'll also need a shapefile with subunit delineations linked in the configuration file.

NHDPlus-Hr delineations can be very small at times, so these scripts include functions to "refactor" the network to a target reach length.  For this example, we'll use a target reach length of 1km.

## Running the scripts

From a shell command

```console
python topo_tools/refactor_nhd.py "path/to/run_metadata.json" --length 1000
```

or from python

```python
from topo_tools import get_reaches

get_reaches(meta_path, 1000)
```