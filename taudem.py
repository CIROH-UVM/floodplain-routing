import subprocess
import os
import time


def update_system_path():
    env = os.environ.copy()
    path_list = env['PATH'].split(';')
    path_list = [p for p in path_list if 'gdal' not in p]
    path_list.append(os.path.join(env['HOMEDRIVE'], 'GDAL'))
    path_list.append(os.path.join(env['PROGRAMFILES'], 'GDAL'))
    path_list.append(os.path.join(env['PROGRAMFILES'], 'TauDEM', 'TauDEM5Exe'))
    env['PATH'] = ';'.join(path_list)
    return env

def PitRemove(z=None, fel=None):
    cmd = f'mpiexec -n {str(os.cpu_count())} PitRemove -z "{z}" -fel "{fel}"'
    run_command(cmd)

def D8FlowDir(fel=None, p=None):
    cmd = f'mpiexec -n {str(os.cpu_count())} D8FlowDir -fel "{fel}" -p "{p}"'
    run_command(cmd)

def Aread8(p=None, ad8=None, nc=True):
    cmd = f'mpiexec -n {str(os.cpu_count())} Aread8 -p "{p}" -ad8 "{ad8}"'
    if nc:
        cmd += ' -nc'
    run_command(cmd)

def run_command(command):
    errors = list()
    try:
        time_start = time.perf_counter()
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, env=TAUDEM_ENVIRON)
        print('process completed in {} seconds'.format(round(time.perf_counter() - time_start, 1)))
    except subprocess.CalledProcessError as e:
        errors.append(e)
    for e in errors:
        print(e)

TAUDEM_ENVIRON = update_system_path()
