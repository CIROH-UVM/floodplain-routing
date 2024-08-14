# Identifying Energy Dissipation Zones and Extracting Their Features

Once geometric information has been extracted ([Tutorial 3](3-Topographic_Extraction.md)), the feature_extraction.py script will identify Energy Dissipation Zones (EDZs) and derive numerous features to describe them.  A csv file with features organized by reach will be exported to a folder called "analysis" in the run folder.

The feature extraction function has two optional command line arguments: subset and plot.  Setting plot to true will generate diagnostic plots for each reach showing the cross-section, stage-Rh curve, and stage-Rh' curve.  The stage-Rh' curve will also have a highlighted area for the EDZ, if one exists.  The subset argument takes a list of reaches and will only extract features for those reaches.  This is largely useful for debugging.

## Running the script

From a shell command

```console
python topo_tools/feature_extraction.py "path/to/run_metadata.json" 
```


or from python

```python
from topo_tools import extract_features

extract_features(meta_path)
```