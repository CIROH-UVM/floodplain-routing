import os
import time
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from osgeo import gdal, ogr
from rasterio import features
from scipy.ndimage import gaussian_filter1d
# from whitebox import WhiteboxTools
gdal.UseExceptions()


def load_raster(in_path):
    raster = dict()

    dataset = gdal.Open(in_path)
    band = dataset.GetRasterBand(1)
    band.ComputeStatistics(0)
    raster['data'] = band.ReadAsArray()

    raster['nd_value'] = band.GetNoDataValue()
    raster['cols'] = dataset.RasterXSize
    raster['rows'] = dataset.RasterYSize
    raster['crs'] = dataset.GetProjectionRef()
    raster['transform'] = dataset.GetGeoTransform()
    raster['origin_x'] = raster['transform'][0]
    raster['origin_y'] = raster['transform'][3]
    raster['pixel_width'] = raster['transform'][1]
    raster['pixel_height'] = raster['transform'][5]
    
    x_max = raster['origin_x'] + (raster['cols'] * raster['pixel_width'])
    y_min = raster['origin_y'] + (raster['rows'] * raster['pixel_height'])
    raster['bbox'] = (raster['origin_x'], y_min, x_max, raster['origin_y'])

    return raster

def gage_areas_from_poly(shp_path, id_field, dem_filter, save_path=None):
    # Get some data from the raster template
    dtype = 'float64'
    bbox = dem_filter['bbox']
    affine = rasterio.Affine.from_gdal(*dem_filter['transform'])
    in_mask = (dem_filter['data'] == dem_filter['nd_value']).reshape(dem_filter['data'].shape)
    
    # Load gage polygons and reformat
    subbasins = gpd.read_file(shp_path, bbox=bbox)
    subbasins[id_field] = subbasins[id_field].astype(dtype)
    subbasins = list(subbasins[['geometry', id_field]].itertuples(index=False, name=None))

    # Rasterize polygons and return
    if save_path:
        with rasterio.open(save_path, 'w+', driver='GTiff', width=dem_filter['cols'], height=dem_filter['rows'], count=1, 
                        crs=dem_filter['crs'], transform=affine, dtype=dtype, nodata=dem_filter['nd_value']) as out:
            out_array = out.read(1)
            rasterized = features.rasterize(shapes=subbasins, fill=dem_filter['nd_value'], out=out_array, transform=out.transform)
            rasterized[in_mask] = dem_filter['nd_value']  # Mask by subbasin outline
            out_array[in_mask] = dem_filter['nd_value']  # Mask by subbasin outline
            out.write_band(1, rasterized)
    else:
        rasterized = features.rasterize(shapes=subbasins, fill=dem_filter['nd_value'], out=np.full(dem_filter['data'].shape, dem_filter['nd_value'], dtype=dtype), transform=affine)
        rasterized[in_mask] = dem_filter['nd_value']  # Mask by subbasin outline

    out_rast = dem_filter.copy()
    out_rast['data'] = rasterized
    return out_rast

def gage_areas_from_poly_gdal(shp_path, id_field, dem_filter, save_path=None, reaches=None):
    # Load gage polygons and reformat
    subunits = ogr.Open(shp_path)
    subunits_layer = subunits.GetLayer()
    if len(reaches) == 1:
        subunits_layer.SetAttributeFilter(f"{id_field} = '{reaches[0]}'")
    elif len(reaches) > 1:
        subunits_layer.SetAttributeFilter(f"{id_field} in {tuple(reaches)}")

    # Rasterize polygons and return
    nd_value = 0
    target_ds = gdal.GetDriverByName('MEM').Create('', dem_filter['cols'], dem_filter['rows'], 1, gdal.GDT_Int64)

    target_ds.SetGeoTransform((dem_filter['origin_x'], dem_filter['pixel_width'], 0, dem_filter['origin_y'], 0, dem_filter['pixel_height']))
    band = target_ds.GetRasterBand(1)
    target_ds.SetProjection(dem_filter['crs'])
    band.SetNoDataValue(nd_value)

    options = [f"ATTRIBUTE={id_field}", "outputType=gdal.GDT_Int64", f"noData={nd_value}", f"initValues={nd_value}"]
    gdal.RasterizeLayer(target_ds, [1], subunits_layer, options=options)

    if save_path:
        # option = gdal.TranslateOptions(creationOptions=['COMPRESS=DEFLATE'])
        # gdal.Translate(save_path, target_ds, options=option)
        gdal.Translate(save_path, target_ds)

    # This is kind of a hacky way to crete a thiessen raster object, but it's faster than calling load_raster()
    thiessen = dem_filter.copy()
    thiessen['data'] = band.ReadAsArray()

    return thiessen


def reach_hydraulics(r, thiessens, elevations, slope, el_nd, resolution, bins):
    mask = thiessens == int(r)  # Select cells within reach area of interest
    mask = np.logical_and(mask, elevations != el_nd)  # Select cells with valid HAND elevation
    mask = np.logical_and(mask, elevations < bins.max())  # Select cells with HAND elevation within range of interest
    tmp_elevations = elevations[mask]
    tmp_slope = np.arctan(slope[mask])
    projected_area = (resolution ** 2) / np.cos(tmp_slope)  # Wetted perimeter
    depth_change = bins[1] - bins[0]
    
    wrk_df = pd.DataFrame({'el': tmp_elevations, 'p': projected_area})
    wrk_df['bins'] = pd.cut(wrk_df['el'], bins=bins, labels=bins[:-1], include_lowest=True)  # Binning a la semivariogram
    wrk_df = wrk_df.groupby(wrk_df['bins'], observed=False).agg(el=('el', 'mean'),
                                                count=('el', 'count'),
                                                p=('p', 'sum'))
    
    wrk_df['el'] = (bins[:-1] + bins[1:]) / 2
    
    wrk_df['area'] = wrk_df['count'].cumsum()
    wrk_df['area'] -= (wrk_df['count'] * 0.5)  # Center
    tmp_p = wrk_df['p'].cumsum()
    tmp_p -= (wrk_df['p'] * 0.5)  # Center
    wrk_df['p'] = tmp_p

    vol_increase = depth_change * wrk_df['area']
    wrk_df['vol'] = np.cumsum(vol_increase)
    wrk_df['rh'] = wrk_df['vol'] / wrk_df['p']
    wrk_df['rh_prime'] = (wrk_df['rh'].shift(-1) - wrk_df['rh']) / depth_change
    wrk_df['rh_prime'] = np.nan_to_num(wrk_df['rh_prime'])
    
    p_prime = (wrk_df['p'].shift(-1) - wrk_df['p']) / depth_change
    k_prime = (5 / 3) - ((2 / 3) * (1 / wrk_df['area']) * wrk_df['rh'] * p_prime)
    wrk_df['celerity'] = k_prime * (wrk_df['rh'] ** (2 / 3))

    if np.all(wrk_df['area'] == 0):
        wrk_df['el'] = np.repeat(0, len(wrk_df['el']))
        wrk_df['rh'] = np.repeat(0, len(wrk_df['rh']))
        wrk_df['rh_prime'] = np.repeat(0, len(wrk_df['rh_prime']))

    return wrk_df

def nwm_geometry(da, stages):
    # Equations based on Read et al. (2023, JAWRA) https://onlinelibrary.wiley.com/doi/full/10.1111/1752-1688.13134
    tw = 2.44 * (da ** 0.34)
    a_ch = 0.75 * (da ** 0.53)
    bf = (a_ch / tw) * 1.25
    bw = ((2 * a_ch) / bf) - tw
    z = (tw - bw) / bf
    tw_cc = 3 * tw

    # Interpolate data
    wrk_df = pd.DataFrame({'el': stages})
    bf_ind = np.argmax(wrk_df['el'] > bf)

    area = bw + (wrk_df['el'] * z)
    area[bf_ind:] = tw_cc
    wrk_df['area'] = area

    h = ((wrk_df['el'] ** 2) * (1 + ((z ** 2) / 4))) ** 0.5
    p = bw + (2 * h)
    p[bf_ind:] = (p[bf_ind] + (tw_cc - tw) + ((wrk_df['el'] - bf) * 2)).iloc[bf_ind:]
    wrk_df['p'] = p

    vol = wrk_df['el'] * ((bw + (wrk_df['el'] * z)) * 0.5)
    vol[bf_ind:] = (vol[bf_ind] + ((wrk_df['el'] - bf) * tw_cc)).iloc[bf_ind:]
    wrk_df['vol'] = vol

    wrk_df['rh'] = wrk_df['vol'] / wrk_df['p']

    wrk_df['rh_prime'] = (wrk_df['rh'].shift(-1) - wrk_df['rh']) / (stages[1] - stages[0])
    wrk_df['rh_prime'] = np.nan_to_num(wrk_df['rh_prime'])

    return wrk_df


def subunit_hydraulics(hand_path, aoi_path, slope_path, stages, reach_field=None, reaches=None, fields_of_interest=None):
    elevations = load_raster(hand_path)
    slope = load_raster(slope_path)
    if aoi_path[-3:] == 'tif':
        thiessens = load_raster(aoi_path)
    elif aoi_path[-3:] == 'shp':
        thiessens = gage_areas_from_poly_gdal(aoi_path, reach_field, elevations, reaches=reaches)

    resolution = elevations['pixel_width'] * elevations['pixel_height']

    data_dict = {k: pd.DataFrame() for k in fields_of_interest}

    counter = 1
    t1 = time.perf_counter()
    for r, s in zip(reaches, stages):
        print(f'{counter} / {len(reaches)}', end="\r")
        wrk_df = reach_hydraulics(r, thiessens['data'], elevations['data'], slope['data'], elevations['nd_value'], resolution, s)

        for k in data_dict:
            # data_dict[k] = pd.concat((data_dict[k], wrk_df[k].rename(r)), axis=1, ignore_index=True)
            data_dict[k][r] = wrk_df[k].reset_index(drop=True)
            # data_dict[k] = data_dict[k].reset_index(drop=True)

        counter += 1
    print('')
    print(f'Completed processing in {round(time.perf_counter() - t1, 1)} seconds')
    return data_dict

def nwm_subunit(das, reaches, stages, lengths, fields_of_interest=None):
    data_dict = {k: pd.DataFrame() for k in fields_of_interest}

    counter = 1
    t1 = time.perf_counter()
    for r, s, da, l in zip(reaches, stages, das, lengths):
        print(f'{counter} / {len(reaches)}', end="\r")

        wrk_df = nwm_geometry(da, s)
        wrk_df['area'] = wrk_df['area'] * l
        wrk_df['vol'] = wrk_df['vol'] * l
        wrk_df['p'] = wrk_df['p'] * l

        for k in data_dict:
            data_dict[k][r] = wrk_df[k].reset_index(drop=True)

        counter += 1
    print('')
    print(f'Completed processing in {round(time.perf_counter() - t1, 1)} seconds')
    return data_dict


def extract_topographic_signature(hand_path, aoi_path, slope_path, reaches=None, max_el=10, nstages=1000, show=False, save_path=None, reach_field=None):
    elevations = load_raster(hand_path)
    slope = load_raster(slope_path)
    if aoi_path[-3:] == 'tif':
        thiessens = load_raster(aoi_path)
    elif aoi_path[-3:] == 'shp':
        thiessens = gage_areas_from_poly(aoi_path, reach_field, elevations)

    resolution = elevations['pixel_width'] * elevations['pixel_height']
    stages = np.linspace(0, max_el, nstages, endpoint=True)

    all_reaches = np.unique(thiessens['data'])
    if reaches == None:
        reaches = [r for r in all_reaches if r != thiessens['nd_value']]  # filter out nodata
    else:
        reaches = list(set(all_reaches).intersection([int(i) for i in reaches]))

    for r in reaches:
        print(r)
        wrk_df = reach_hydraulics(r, thiessens['data'], elevations['data'], slope['data'], elevations['nd_value'], resolution, stages)

        fig, ax = plt.subplots()
        ax.plot(wrk_df['el'], wrk_df['rh_prime'], label='raw')
        ax.plot(wrk_df['el'], gaussian_filter1d(wrk_df['rh_prime'], 3), label='smoothed')
        ax.set_xlabel('Stage (m)')
        ax.set_ylabel(r"$R_{h}$ '")
        ax.set_ylim(-3, 1)
        ax.set_title(r)
        plt.legend()
        if show:
            plt.show()
        if save_path:
            tmp_path = os.path.join(save_path, f'{r}.png')
            fig.savefig(tmp_path, dpi=300)
        
        plt.close(fig)

def add_bathymetry(geom, da, slope):
    dim = geom['el'].shape[0]
    # 0.015 meters is a reasonable threshold to extract the lidar-based wetted top-width.  Sensitivity analysis by UVM Fall 2023
    filter_arg = np.argmin(geom['el'] < 0.015)
    filter_arg = max(filter_arg, 2)  # need at least two 
    top_width = geom['area'][filter_arg]
    #  Use regression of Read et al 2023. https://onlinelibrary.wiley.com/doi/full/10.1111/1752-1688.13134
    bottom_width = min(top_width, (2.44 * (da ** 0.34)))  # try to use NWM channel top-width, unless it would lead to decreasing top-width
    flowrate = (0.4962 * da) / 35.3147  # Diehl Estimate
    n = 0.01
    max_space = 2 * (0.26 * (da ** 0.287))  # Cap at 2xbkfl
    stage_inc = np.median(geom['el'][1:] - geom['el'][:-1])
    stage_space = np.arange(0, max_space, stage_inc)
    width_space = np.linspace(bottom_width, top_width, stage_space.shape[0])
    dw = width_space[1:] - width_space[:-1]
    ds = stage_space[1:] - stage_space[:-1]
    dp = np.sqrt((dw ** 2) + (ds ** 2))
    dp = np.insert(dp, 0, dp[0])

    # Assume rectangular channel
    area = (stage_space * top_width)
    perimeter = (bottom_width + (2 * dp))

    flowrate_space = (1 / n) * (stage_space * top_width) * (slope ** 0.5) * ((area / perimeter) ** (2 / 3))
    channel_ind = np.argmax(flowrate_space > flowrate)
    channel_ind = max(channel_ind, filter_arg)  # need at least two points of bathy for a square.  Need at least 0.025m to replace cut out

    stage_space = stage_space[:channel_ind]
    area = area[:channel_ind]
    perimeter = perimeter[:channel_ind]
    area_diff = geom['area'][filter_arg] - top_width

    geom['area'] = geom['area'][filter_arg:]
    geom['area'] = np.insert(geom['area'], 0, np.repeat(top_width, channel_ind))
    geom['area'] = geom['area'][:dim]

    geom['el'] -= geom['el'][filter_arg - 1]  # first data point should be at top of bathymetry
    geom['el'] = geom['el'][filter_arg:]
    geom['el'] += stage_space[-1]
    geom['el'] = np.insert(geom['el'], 0, stage_space)
    geom['el'] = geom['el'][:dim]

    geom['vol'] -= geom['vol'][filter_arg - 1]
    geom['vol'] = geom['vol'][filter_arg:]
    geom['vol'] += area[-1]
    geom['vol'] = np.insert(geom['vol'], 0, area)
    geom['vol'] = geom['vol'][:dim]

    geom['p'] -= geom['p'][filter_arg - 1]
    geom['p'] = geom['p'][filter_arg:]
    geom['p'] += perimeter[-1] + area_diff
    geom['p'] = np.insert(geom['p'], 0, perimeter)
    geom['p'] = geom['p'][:dim]

    return geom

def calc_celerity(geom, slope):
    dp = geom['p'][1:] - geom['p'][:-1]
    dy = geom['el'][1:] - geom['el'][:-1]
    dp_dy = dp / dy
    dp_dy[0] = dp_dy[1]
    dp_dy[np.isnan(dp_dy)] = 0.0001
    dp_dy = gaussian_filter1d(dp_dy, 15)
    dp_dy[dp_dy < 0.0001] = 0.0001
    dp_dy = np.append(dp_dy, dp_dy[-1])
    k_prime = (5 / 3) - ((2 / 3)*(geom['vol'] / (geom['area'] * geom['p'])) * dp_dy)
    q = (1 / 0.07) * geom['vol'] * ((geom['vol'] / geom['p']) ** (2 / 3)) * (slope ** 0.5)
    geom['vol'][0] = geom['vol'][1]
    celerity = k_prime * (q / geom['vol'])
    celerity[0] = celerity[1]
    celerity = np.nan_to_num(celerity)
    celerity[celerity <= 0] = 0.0001
    return celerity


def map_edz(hand_path, aoi_path, reach_field, reach_data):
    reaches = reach_data[reach_field].unique()
    elevations = load_raster(hand_path)
    if aoi_path[-3:] == 'tif':
        thiessens = load_raster(aoi_path)
    elif aoi_path[-3:] == 'shp':
        thiessens = gage_areas_from_poly_gdal(aoi_path, reach_field, elevations, reaches=reaches)

    out_data = np.zeros(elevations['data'].shape)
    counter = 1
    t1 = time.perf_counter()
    for r in reaches:
        print(f'{counter} / {len(reaches)}', end="\r")
        tmp_reach = reach_data[reach_data[reach_field] == r]

        reach_mask = (thiessens['data'] == int(r))
        edz_mask = np.logical_and((elevations['data'] > tmp_reach['el_edap'].values[0]), (elevations['data'] < tmp_reach['el_edep'].values[0]))
        combo_mask = np.logical_and(reach_mask, edz_mask)

        out_data[combo_mask] = 1
        
        counter += 1

    cols = elevations['cols']
    rows = elevations['rows']
    originX = elevations['origin_x']
    originY = elevations['origin_y']

    driver = gdal.GetDriverByName('GTiff')
    out_path = os.path.join(os.path.dirname(hand_path), 'edz.tif')
    if os.path.exists(out_path):
        os.remove(out_path)
    if os.path.exists(out_path):
        os.remove(out_path + '.aux.xml')
    outRaster = driver.Create(out_path, cols, rows, 1, gdal.GDT_Byte, options=['COMPRESS=LZW'])
    outRaster.SetGeoTransform((originX, elevations['pixel_width'], 0, originY, 0, elevations['pixel_height']))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(out_data.astype(np.int8))
    outband.SetNoDataValue(0)
    outRaster.SetProjection(elevations['crs'])
    outRaster.FlushCache()
    gdal.Dataset.__swig_destroy__(outRaster)
    outband = outRaster = None
    print('')
    print(f'Completed processing in {round(time.perf_counter() - t1, 1)} seconds')
    return out_path

def merge_rasters(paths, out_path):
    tmp_path = out_path.replace('tif', 'vrt')
    gdal.BuildVRT(tmp_path, paths)
    gdal.Translate(out_path, tmp_path, creationOptions=['COMPRESS=LZW'])
    os.remove(tmp_path)


def plot_rh_curve(hand_path, aoi_path, slope_path, reaches=None, max_el=10, nstages=1000, show=False, save_path=None, reach_field=None):
    elevations = load_raster(hand_path)
    slope = load_raster(slope_path)
    if aoi_path[-3:] == 'tif':
        thiessens = load_raster(aoi_path)
    elif aoi_path[-3:] == 'shp':
        thiessens = gage_areas_from_poly(aoi_path, reach_field, elevations)

    resolution = elevations['pixel_width'] * elevations['pixel_height']

    all_reaches = np.unique(thiessens['data'])
    if reaches == None:
        reaches = [r for r in all_reaches if r != thiessens['nd_value']]  # filter out nodata
    else:
        reaches = list(set(all_reaches).intersection([int(i) for i in reaches]))

    for r in reaches:
        print(r)
        wrk_df = reach_hydraulics(r, thiessens['data'], elevations['data'], slope['data'], elevations['nd_value'], nstages, resolution, max_el)

        fig, ax = plt.subplots()
        ax.plot(wrk_df['el'], wrk_df['rh'])
        ax.set_xlabel('Stage (m)')
        ax.set_ylabel(r"$R_{h}$")
        ax.set_ylim(0, 10)
        ax.set_title(r)
        if show:
            plt.show()
        if save_path:
            tmp_path = os.path.join(save_path, f'{r}.png')
            fig.savefig(tmp_path, dpi=300)
        
        plt.close(fig)


def extract_celerity_signature(hand_path, aoi_path, slope_path, reaches=None, max_el=10, nstages=1000, show=False, save_path=None):
    elevations, el_meta = load_raster(hand_path)
    thiessens, thiessen_meta = load_raster(aoi_path)
    slope, slope_meta = load_raster(slope_path)

    resolution = el_meta['pixel_width'] * el_meta['pixel_height']

    all_reaches = np.unique(thiessens)
    if reaches == None:
        reaches = [r for r in all_reaches if r != thiessen_meta['nd_value']]  # filter out nodata
    else:
        reaches = list(set(all_reaches).intersection([int(i) for i in reaches]))

    for r in reaches:
        print(r)
        mask = thiessens == r
        mask = np.logical_and(mask, elevations != el_meta['nd_value'])
        mask = np.logical_and(mask, elevations < max_el)
        tmp_elevations = elevations[mask]
        tmp_slope = np.arctan(slope[mask])
        projected_area = (resolution ** 2) / np.cos(tmp_slope)

        bins = np.linspace(0, 10, nstages, endpoint=True)
        wrk_df = pd.DataFrame({'el': tmp_elevations, 'p': projected_area})
        wrk_df['bins'] = pd.cut(wrk_df['el'], bins=bins, labels=bins[:-1], include_lowest=True)
        wrk_df = wrk_df.groupby(wrk_df['bins']).agg(el=('el', 'mean'),
                                                    count=('el', 'count'),
                                                    p=('p', 'sum'))
        
        wrk_df['area'] = wrk_df['count'].cumsum()
        wrk_df['area'] -= (wrk_df['count'] * 0.5)
        wrk_df['p'] = wrk_df['p'].cumsum()

        depth_change = bins[1] - bins[0]
        vol_increase = depth_change * wrk_df['area']
        wrk_df['vol'] = np.cumsum(vol_increase)
        wrk_df['rh'] = wrk_df['vol'] / wrk_df['p']
        wrk_df['rh_prime'] = (wrk_df['rh'].shift(-1) - wrk_df['rh']) / depth_change

        wrk_df['p_prime'] = (wrk_df['p'].shift(-1) - wrk_df['p']) / depth_change
        k_prime = (5 / 3) - ((2 / 3) * (1 / wrk_df['area']) * wrk_df['rh'] * wrk_df['p_prime'])
        wrk_df['celerity'] = k_prime * (wrk_df['rh'] ** (2 / 3))

        fig, ax = plt.subplots()
        ax.plot(wrk_df['el'], wrk_df['celerity'], label='raw')
        ax.plot(wrk_df['el'], gaussian_filter1d(wrk_df['celerity'], 3), label='smoothed')
        ax.set_xlabel('Stage (m)')
        ax.set_ylabel('Kinematic Celertiy')
        ax.set_title(r)
        plt.legend()
        if show:
            plt.show()
        if save_path:
            tmp_path = os.path.join(save_path, f'{r}.png')
            fig.savefig(tmp_path, dpi=300)
        plt.close(fig)

def generate_geomorphons(raster_dir, working_dir):
    # Set up workspace
    location = 'grass_tmp'
    mapset = 'permanent'
    epsg = 32145

    dem_path = os.path.join(raster_dir, 'DEM.tif')
    valley_bottom_path = os.path.join(raster_dir, 'valley_bottom.tif')
    geomorphon_path = os.path.join(raster_dir, 'geomorphon_raw.tif')
    wrk_path = os.path.join(raster_dir, 'geomorphon_tmp.tif')
    sieve_path = os.path.join(raster_dir, 'geomorphon_tmp_sieve.tif')
    out_path = os.path.join(raster_dir, 'geomorphon_clean.tif')
    
    tstart = time.perf_counter()
    wbt = WhiteboxTools()

    # Generate geomorphon
    wbt.geomorphons(dem_path, geomorphon_path, search=34, skip=12, threshold=1, fdist=0)

    # Reclassify
    reclass_vals = '1;5;1;6;1;7;2;2;2;3;2;4;3;8;3;9;3;10'
    wbt.reclass(geomorphon_path, wrk_path, reclass_vals, assign_mode=True)

    # Majority filter
    wbt.majority_filter(wrk_path, wrk_path, filterx=5, filtery=5)

    # Clump
    wbt.clump(wrk_path, sieve_path, diag=True, zero_back=False)

    # Edge proportion
    wbt.edge_proportion(sieve_path, sieve_path)

    # Filter
    reclass_vals = '-999;0.75;1.1;999;-0.1;0.75'
    wbt.reclass(sieve_path, sieve_path, reclass_vals, assign_mode=False)

    # Sieve
    wbt.min(sieve_path, wrk_path, wrk_path)
    os.remove(sieve_path)

    # Enforce nodata
    wbt.set_nodata_value(wrk_path, wrk_path, -999)

    # Fill gaps (nibble)
    wbt.fill_missing_data(wrk_path, wrk_path, filter=25, weight=2, no_edges=True)

    # Cleanup majority filter
    wbt.majority_filter(wrk_path, wrk_path, filterx=5, filtery=5)

    # Clip to valley bottoms
    wbt.multiply(wrk_path, valley_bottom_path, out_path)

    os.remove(wrk_path)

    print(f' - geomorphon reclassed and cleaned in {round(time.perf_counter() - tstart, 1)} seconds')

def reclass_geomorphons_channel(subbasin, thalweg_path, subbasin_path):
    geomorphon_path = os.path.join(subbasin, 'rasters', 'geomorphon_clean.tif')
    geomorphon_poly_path = os.path.join(subbasin, 'shapefiles', 'geomorphon.shp')
    geomorphon_reclass_path = os.path.join(subbasin, 'rasters', 'geomorphon_clean_reclass_2.tif')

    # Polygonize
    polygonize = True
    if polygonize:
        print('polygonizing')
        dataset = gdal.Open(geomorphon_path)
        band = dataset.GetRasterBand(1)
        crs = dataset.GetProjectionRef()
        srs = ogr.osr.SpatialReference()
        srs.ImportFromWkt(crs)

        poly_ds = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource(geomorphon_poly_path)
        poly_layer = poly_ds.CreateLayer('geomorphon', srs=srs)
        field_def = ogr.FieldDefn('gmorph_cls', ogr.OFTInteger)
        poly_layer.CreateField(field_def)

        gdal.Polygonize(band, None, poly_layer, 0, [])
        poly_ds = None

    # Intersect with thalweg
    intersect = True
    if intersect:
        print('intersecting')
        polys = gpd.read_file(geomorphon_poly_path)
        polys = polys.query('gmorph_cls > 0').copy()
        polys['PID'] = polys.index.to_list()
        _ = polys.total_bounds
        thalweg = gpd.read_file(thalweg_path, bbox=polys)
        thalweg = thalweg[['geometry']]
        concaves = polys.query('gmorph_cls == 3')
        intersects = concaves.geometry.map(lambda x: x.intersects(thalweg.geometry).any())
        tmp_df = pd.DataFrame({'PID': concaves['PID'], 'channel': intersects})
        polys = polys.merge(tmp_df, how='left', on='PID')
        polys['channel'] = polys['channel'].fillna(0)
        polys.to_file(geomorphon_poly_path)
        polys = None
        thalweg = None
    
    # Process largest
    geoprocess = True
    if geoprocess:
        print('geoprocessing')
        polys = gpd.read_file(geomorphon_poly_path)
        polys = polys.query('gmorph_cls > 0')
        _ = polys.total_bounds
        subbasins = gpd.read_file(subbasin_path, bbox=polys)
        subbasins = subbasins[['geometry', 'MergedCode']]
        subbasins = subbasins.dissolve(by='MergedCode')
        subbasins['Code'] = subbasins.index.to_list()
        polys = polys.overlay(subbasins, how='intersection')
        polys['PID'] = polys.index.to_list()
        polys['area'] = polys.geometry.area
        concaves = polys.query('gmorph_cls == 3')
        group = concaves.groupby('Code')
        max_rows = group['area'].transform(max) == concaves['area']
        max_area = np.zeros(len(concaves))
        max_area[max_rows] = 1
        tmp_df = pd.DataFrame({'PID': concaves['PID'], 'largest_area': max_area})
        print(len(polys))
        polys = polys.merge(tmp_df, how='left', on='PID')
        polys['largest_area'] = polys['largest_area'].fillna(0)
        print(len(polys))
        polys.to_file(geomorphon_poly_path)

    # Reclassify channel to class_4
    reclassify = True
    if reclassify:
        print('reclassifying')
        polys = gpd.read_file(geomorphon_poly_path)
        tmp_classes = polys['gmorph_cls'].to_numpy()
        tmp_classes[polys['channel'] == "1"] = 4
        polys['gmorph_cls'] = tmp_classes
        polys.to_file(geomorphon_poly_path)

    # Re-rasterize
    rasterize = True
    if rasterize:
        print('rasterizing')
        subunits = ogr.Open(geomorphon_poly_path)
        subunits_layer = subunits.GetLayer()

        template_raster = load_raster(geomorphon_path)
        
        nd_value = 0
        target_ds = gdal.GetDriverByName('MEM').Create('', template_raster['cols'], template_raster['rows'], 1, gdal.GDT_Byte)

        target_ds.SetGeoTransform((template_raster['origin_x'], template_raster['pixel_width'], 0, template_raster['origin_y'], 0, template_raster['pixel_height']))
        band = target_ds.GetRasterBand(1)
        target_ds.SetProjection(template_raster['crs'])
        band.SetNoDataValue(nd_value)

        id_field = 'gmorph_cls'
        options = [f"ATTRIBUTE={id_field}", "outputType=gdal.GDT_Int64", f"noData={nd_value}", f"initValues={nd_value}"]
        gdal.RasterizeLayer(target_ds, [1], subunits_layer, options=options)

        gdal.Translate(geomorphon_reclass_path, target_ds)

