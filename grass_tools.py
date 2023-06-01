import os
import sys
import subprocess
from utilities import load_raster

### https://grasswiki.osgeo.org/wiki/Working_with_GRASS_without_starting_it_explicitly#Python:_GRASS_GIS_7_without_existing_location_using_metadata_only ###
GRASS7BIN = r"C:\OSGeo4W\bin\grass78.bat"
startcmd = [GRASS7BIN, '--config', 'path']
p = subprocess.run(startcmd, capture_output=True)
if p.returncode != 0:
    print(p.stderr, "ERROR: Cannot find GRASS GIS 7 start script ({})".format(startcmd))
    sys.exit(-1)
GISBASE = p.stdout.decode("utf-8").strip('\n\r')
os.environ['GISBASE'] = GISBASE
gpydir = os.path.join(GISBASE, "etc", "python")
sys.path.append(gpydir)
os.environ['GRASS_SH'] = os.path.join(GISBASE, 'msys', 'bin', 'sh.exe')
import grass.script as gscript
import grass.script.setup as gsetup


class GrassSession:

    def __init__(self, gisdb, location, mapset, epsg, ):
        grass_location = os.path.join(gisdb, location)
        if not os.path.exists(grass_location):
            cmd = f'"{GRASS7BIN}" -c "EPSG:{epsg}" -e "{grass_location}"'
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0:
                print(f'ERROR: {r.stderr}')
                print(f'ERROR: Cannot generate location ({grass_location})')
                sys.exit(-1)
            else:
                print(f'Created location {grass_location}')

        gsetup.init(GISBASE, gisdb, location, mapset)
        gscript.message('Current GRASS GIS 7 environment:')
        print(gscript.gisenv())

    def geomorphon(self, dem_path, out_path, search=3, skip=0, flat=1, dist=0):
        print(print(gscript.gisenv()))
        in_proj = gscript.read_command('g.proj', flags = 'jf')

        gscript.run_command('r.import', input=dem_path, output='DEM', flags='o')
        gscript.run_command('g.region rast=DEM')
        gscript.run_command('r.geomorphon', elevation="DEM", forms='geomorphon', search=search, skip=skip, flat=flat, dist=dist)
        gscript.run_command('r.out.gdal', input='geomorphon', output=out_path)

    def reclass(self, in_path, out_path, rules_path):
        print(print(gscript.gisenv()))
        
        gscript.run_command('r.import', input=in_path, output='input', flags='o')
        gscript.run_command('g.region rast=input')
        gscript.run_command('r.reclass', input="input", output='reclassed', rules=rules_path)
        gscript.run_command('r.out.gdal', input='reclassed', output=out_path)

