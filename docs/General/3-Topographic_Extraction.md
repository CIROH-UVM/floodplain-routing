# Extracting Geometry from Rasters

Now that you have a configuration file ([Tutorial 1](1-Metadata_File.md)) and reach information ([Tutorial 2](2-Reach_Delineation.md)), you can start extracting reach-averaged geometric information.  Running this step will create a new folder in your run folder called "geometry."  That folder will contain individual csv files for each hydraulic geometry measure (top-width, cross-section area, wetted perimeter, etc). Columns begin with the reach ID and then contain a series representing the geometry measure at each stage analyzed.

While this script doesn't have any command line arguments, remember that there are two elements in the configuration file that will affect how the script executes: scaled_stages and geometry_source.  See [Tutorial 1](1-Metadata_File.md) for details.

## Running the scripts

From a shell command

```console
python -m topo_tools.controller "path/to/run_metadata.json"
```
**Note:_** controller.py imports other tools from the module, so it must be run as a module.

or from python

```python
from topo_tools import extract_geometry, batch_add_bathymetry

extract_geometry(meta_path)
batch_add_bathymetry(meta_path)
```