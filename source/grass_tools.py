import os
import sys
import subprocess

### https://grasswiki.osgeo.org/wiki/Working_with_GRASS_without_starting_it_explicitly#Python:_GRASS_GIS_7_without_existing_location_using_metadata_only ###
GRASS8BIN = r"C:\Program Files\GRASS GIS 8.2\grass82.bat"
startcmd = [GRASS8BIN, '--config', 'python_path']
p = subprocess.run(startcmd, capture_output=True)
if p.returncode != 0:
    print(p.stderr, "ERROR: Cannot find GRASS GIS 7 start script ({})".format(startcmd))
    sys.exit(-1)
gpydir = p.stdout.decode("utf-8").strip('\n\r')
sys.path.append(gpydir)
import grass.script as gs
import grass.script.setup as gsetup

# if you need to install r.pi (g.extension extension=r.pi)
# list addons g.extension -a operation=add   



class GrassSession:

    def __init__(self, gisdb, location, mapset, epsg, ):
        grass_location = os.path.join(gisdb, location)
        if not os.path.exists(grass_location):
            cmd = f'"{GRASS8BIN}" -c "EPSG:{epsg}" -e "{grass_location}"'
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0:
                print(f'ERROR: {r.stderr}')
                print(f'ERROR: Cannot generate location ({grass_location})')
                sys.exit(-1)
            else:
                print(f'Created location {grass_location}')

        gsetup.init(gisdb, location, mapset)
        gs.message('Current GRASS GIS 7 environment:')
        print(gs.gisenv())

    def geomorphon(self, dem_path, out_path, search=3, skip=0, flat=1, dist=0):
        gs.run_command('r.in.gdal', input=dem_path, output='DEM', overwrite=True)
        gs.run_command('g.region', rast='DEM')
        gs.run_command('r.geomorphon', elevation="DEM", forms='geomorphon', overwrite=True, search=search, skip=skip, flat=flat, dist=dist)
        gs.run_command('r.out.gdal', input='geomorphon', type='Int16', overwrite=True, output=out_path)

    def multiply(self, in_path_1, in_path_2, out_path):
        ### Output resolution will match i1 raster resolution ###
        gs.run_command('r.in.gdal', input=in_path_1, output='i1', overwrite=True)
        gs.run_command('r.in.gdal', input=in_path_2, output='i2', overwrite=True)
        gs.run_command('g.region', rast='i1')
        gs.run_command('r.mapcalc', expression="out = i1 * i2", overwrite=True)
        gs.run_command('r.out.gdal', input='out', output=out_path, overwrite=True)

    def reclass(self, in_path, out_path, rules_path):
        gs.run_command('r.in.gdal', input=in_path, output='input', overwrite=True)
        gs.run_command('g.region', rast='input')
        gs.run_command('r.reclass', input="input", output='reclassed', rules=rules_path, overwrite=True)
        gs.run_command('r.out.gdal', input='reclassed', output=out_path, overwrite=True)
    
    def clean_geomorphon(self, in_path, out_dir):
        gs.run_command('r.in.gdal', input=in_path, output='input', overwrite=True)
        gs.run_command('g.region', rast='input')

        # Majority filter
        gs.run_command('r.neighbors', input="input", output='maj_filter', method='mode', nprocs=6, overwrite=True)
        out_path = os.path.join(out_dir, 'maj_filter.tif')
        gs.run_command('r.out.gdal', input='maj_filter', output=out_path, type='Int16', overwrite=True)

        # Clump
        gs.run_command('r.clump', input='maj_filter', output='clumped', overwrite=True)
        out_path = os.path.join(out_dir, 'clumped.tif')
        gs.run_command('r.out.gdal', input='clumped', output=out_path, overwrite=True)

        # Edge Density
        gs.run_command('r.reclass.area', input='maj_filter', output='rmarea', value=0.003, mode='lesser', method='rmarea', overwrite=True)
        out_path = os.path.join(out_dir, 'rmarea.tif')
        gs.run_command('r.out.gdal', input='rmarea', output=out_path, overwrite=True)

