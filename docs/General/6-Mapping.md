# Creating Maps from the Analysis

Scripts are available to generate raster and shapefile extents of the EDZ as well as several design floods.  Using the map_floodplain function, the stage of each recurrence interval will be taken from flood_metrics.csv (made in [Tutorial 5](5-Flood_Metrics.md)) and converted into a raster and shapefile of inumdation extents. Using the map_edzs function, EDAP and EDEP stages will be taken from data.csv (made in [Tutorial 4](4-Feature_Extraction.md)).  Stages below EDAP will be marked as channel and stages between EDAP and EDEP will be marked as EDZ.  Raster and shapefile extents will then be generated with both zones.

## Running the scripts

From a shell command, you may generate all recurrence interval maps and the EDZ map in one go.

```console
python -m topo_tools.map_tools "path/to/run_metadata.json" 
```

From python, you have more control over what you map.  Adjust the code below as necessary.

```python
from topo_tools import map_floodplain, map_edzs

for m in ['Q2', 'Q5', 'Q10','Q25', 'Q50', 'Q100', 'Q200', 'Q500']:
    map_floodplain(meta_path, m)
map_edzs(meta_path)
```