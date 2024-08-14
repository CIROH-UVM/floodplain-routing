# The run_metadata.json file

The first step in running this workflow is creating a configuration file.  This file (run_metadata.json) contains paths and settings that will inform which options of the scripts are executed.  The file will look something like this

```json
{
    "data_directory": "/path/to/data",
    "run_directory": "/path/to/data/runs/run_id",
    "id_field": "UVM_ID",
    "unit_field": "8_name",
    "subunit_field": "12_code",
    "scaled_stages": true,
    "huc4": "0430",
    "subunit_path": "/path/to/other/data/subunits.shp",
    "geometry_source": "HAND"
}
```

Here are more details on those options

 - <b>data_directory:</b> This is a directory containing all your data (DEMs, slope rasters, HAND files).  The directory needs to have a specific structure.  An example is provided in the 'samples' folder of this repository.  The first level of folders in the data directory are names for the "unit" they have data for.  UVM does this for the HUC8 level, but you may use whatever is convenient for you.  The second level under the "unit" folder is a folder named "subbasins".  This folder contains folders for all "subbasins" or "subunits" within the "unit."  At UVM, the folders within "subbasins" represent the HUC12 level, but again, they may be whatever is convenient for your project.
 - <b>run_directory:</b> This is a folder where all the input and output files for a run live.  The default placement is within a folder called "runs" in the data directory, buwever, this can be adjusted.
 - <b>id_field:</b> This tool uses the reach as the fundamental unit of spatial aggregation.  Each reach within the run needs to have a unique ID (string).  Those IDs are stored within a field in several input and output files, and that field should be identified with the name provided here.
 - <b>unit_field:</b> The field that identifies which unit a reach is in.  When the script analyzes a unit, it uses this field to determine which reaches to run.
 - <b>subunit_field:</b> The field that identifies which subunit a reach is in.  When the script analyzes a subunit, it uses this field to determine which reaches to run.
 - <b>scaled_stages:</b> When true, the script will extract geometric information from stages ranging from 0 to 6 times bankfull depth.  When false, the script will extract geometric information from 0 to 10 meters.
 - <b>huc4:</b> This information is only used in the refactor_nhd.py scripts.  It defines which NHDPlus files to download from the USGS servers.
 - <b>subunit_path:</b> This is the path to a shapefile containing subunit boundaries.  Information from this file is intersected with NHDPlus data in refactor_nhd.py to determine subunit membership of reaches.
 - <b>geometry_source:</b> The geometry extraction steps may be run on HAND rasters ("HAND") or DEM rasters ("DEM").  When using the DEM option, reaches must be short enough such that the thalweg elevation change across the reach is negligible.

 ## Generating run_metadata.json

 If you want to make a boilerplate configuration file, you can run the following command in your system's shell

 ```console
python topo_tools/make_run_template.py "path/to/data" "run_id"
 ```

 or you may import this module into a python script and run

 ```python
from topo_tools import make_run_template

data_dir = r'path/to/data"
run_id = 'run_id'
make_run_template(data_dir, run_id)
 ```