import subprocess
import os
import time


def update_system_path():
    env = os.environ
    path_list = env['PATH'].split(';')
    path_list = [p for p in path_list if 'gdal' not in p]
    path_list.append(os.path.join(env['HOMEDRIVE'], '\saga-8.5.1_x64'))
    env['PATH'] = ';'.join(path_list)
    return env

def run_command(command):
    errors = list()
    try:
        time_start = time.perf_counter()
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, env=SAGA_ENVIRON)
        print('process completed in {} seconds'.format(round(time.perf_counter() - time_start, 1)))
    except subprocess.CalledProcessError as e:
        errors.append(e)
    for e in errors:
        print(e)

def geomorphons(dem=None, geomorphons=None, threshold=1, radius=10000, method=1, dlevel=3):
    cmd = f'saga_cmd ta_lighting 8 -DEM "{dem}" -GEOMORPHONS "{geomorphons}" -THRESHOLD {threshold} -RADIUS {radius} -METHOD {method} -DLEVEL {dlevel}'
    run_command(cmd)

def clump_filter(grid=None, output=None, threshold=10):
    cmd = f'saga_cmd grid_filter 5 -GRID "{grid}" -OUTPUT "{output}" -THRESHOLD {threshold}'
    run_command(cmd)

def majority_filter(input=None, result=None, type=0, radius=1, threshold=0, kernel_type=1):
    cmd = f'saga_cmd grid_filter 6 -INPUT "{input}" -RESULT "{result}" -TYPE {type} -RADIUS {radius} -THRESHOLD {threshold} -KERNEL_TYPE {kernel_type}'
    run_command(cmd)

def morphological_filter(input=None, result=None, method=0, kernel_radius=2, kernel_type=1):
    cmd = f'saga_cmd grid_filter 8 -INPUT "{input}" -RESULT "{result}" -METHOD {method} -KERNEL_RADIUS {kernel_radius} -KERNEL_TYPE {kernel_type}'
    run_command(cmd)

def mesh_denoise(input=None, output=None, sigma=0.9, iter=5, viter=50, nb_cv=0, zonly=0):
    cmd = f'saga_cmd grid_filter 10 -INPUT "{input}" -OUTPUT "{output}" -SIGMA {sigma} -ITER {iter} -VITER {viter}, -NB_CV {nb_cv} -ZONLY {zonly}'
    run_command(cmd)

def geodesic_morphological_reconstruction(input_grid=None, object_grid=None, difference_grid=None, shift_value=5, border_yes_no=1, bin_yes_no=1, threshold=1):
    if difference_grid:
        difference_grid = f'-DIFFERENCE_GRID {difference_grid} '
    else:
        difference_grid = ''
    cmd = f'saga_cmd grid_filter 12 -INPUT_GRID "{input_grid}" -OBJECT_GRID "{object_grid}" {difference_grid}-SHIFT_VALUE {shift_value} -BORDER_YES_NO {border_yes_no} -BIN_YES_NO {bin_yes_no} -THRESHOLD {threshold}'
    run_command(cmd)

def binary_erosion_reconstruction(input_grid=None, output_grid=None, radius=3):
    cmd = f'saga_cmd grid_filter 13 -INPUT_GRID "{input_grid}" -OUTPUT_GRID "{output_grid}" -RADIUS {radius}'
    run_command(cmd)



SAGA_ENVIRON = update_system_path()
