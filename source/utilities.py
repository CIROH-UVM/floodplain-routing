import os
import time
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from osgeo import gdal, ogr, osr
from rasterio import features
from scipy.ndimage import gaussian_filter1d
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

def reach_hydraulics(r, thiessens, elevations, slope, el_nd, resolution, bins, el_type='hand'):
    mask = thiessens == int(r)  # Select cells within reach area of interest
    mask = np.logical_and(mask, elevations != el_nd)  # Select cells with valid HAND elevation
    if el_type == 'dem':
        tmp_elevations = elevations - elevations[mask].min()
        mask = np.logical_and(mask, tmp_elevations < bins.max())  # Select cells with HAND elevation within range of interest
        tmp_elevations = elevations[mask]
        tmp_elevations = tmp_elevations - tmp_elevations.min()
    else:
        mask = np.logical_and(mask, elevations < bins.max())  # Select cells with HAND elevation within range of interest
        tmp_elevations = elevations[mask]
    tmp_slope = np.arctan(slope[mask])
    projected_area = resolution / np.cos(tmp_slope)  # Wetted perimeter
    depth_change = bins[1] - bins[0]
    
    wrk_df = pd.DataFrame({'el': tmp_elevations, 'p': projected_area})
    wrk_df['bins'] = pd.cut(wrk_df['el'], bins=bins, labels=bins[:-1], include_lowest=True)  # Binning a la semivariogram
    wrk_df = wrk_df.groupby(wrk_df['bins'], observed=False).agg(el=('el', 'mean'),
                                                count=('el', 'count'),
                                                p=('p', 'sum'))
    
    wrk_df['el'] = (bins[:-1] + bins[1:]) / 2
    
    wrk_df['area'] = wrk_df['count'].cumsum() * resolution
    wrk_df['area'] -= (wrk_df['count'] * 0.5 * resolution)  # Center
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

def subunit_hydraulics(hand_path, aoi_path, slope_path, stages, reach_field=None, reaches=None, fields_of_interest=None, el_type='hand'):
    elevations = load_raster(hand_path)
    slope = load_raster(slope_path)
    if aoi_path[-3:] == 'tif':
        thiessens = load_raster(aoi_path)
    elif aoi_path[-3:] == 'shp':
        thiessens = gage_areas_from_poly_gdal(aoi_path, reach_field, elevations, reaches=reaches)

    resolution = abs(elevations['pixel_width'] * elevations['pixel_height'])

    data_dict = {k: pd.DataFrame() for k in fields_of_interest}

    counter = 1
    t1 = time.perf_counter()
    for r, s in zip(reaches, stages):
        print(f'{counter} / {len(reaches)}', end="\r")
        wrk_df = reach_hydraulics(r, thiessens['data'], elevations['data'], slope['data'], elevations['nd_value'], resolution, s, el_type=el_type)

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

    poly_path = os.path.join(os.path.dirname(os.path.dirname(hand_path)), 'vectors')
    os.makedirs(poly_path, exist_ok=True)
    poly_path = os.path.join(poly_path, 'edz.shp')
    # Load AOI vector
    if len(reaches) == 1:
        query = f"{reach_field} = '{reaches[0]}'"
    elif len(reaches) > 1:
        query = f"{reach_field} in {tuple(reach_data[reach_field].values)}"
    reach_polys = gpd.read_file(aoi_path, where=query)[[reach_field, 'geometry']]
    polygonize_raster(out_path, poly_path, reach_polys)

    print('')
    print(f'Completed processing in {round(time.perf_counter() - t1, 1)} seconds')
    return out_path


def build_raster(hand_path, aoi_path, reach_field, reach_data, out_name):
    """ A multi-purpose version of map_edz() that can be used to map any number of zones to a raster. """

    reaches = list(reach_data.keys())
    elevations = load_raster(hand_path)
    if aoi_path[-3:] == 'tif':
        thiessens = load_raster(aoi_path)
    elif aoi_path[-3:] == 'shp':
        thiessens = gage_areas_from_poly_gdal(aoi_path, reach_field, elevations, reaches=reaches)

    # make an internal ID
    zones = list()
    for r in reaches:
        zones.extend([i['label'] for i in reach_data[r]['zones']])
    iids = {z: i+1 for i, z in enumerate(np.unique(zones))}
    reverse_iids = {v: k for k, v in iids.items()}
    
    # initialize empty raster and go
    out_data = np.zeros(elevations['data'].shape)
    counter = 1
    t1 = time.perf_counter()
    for r in reaches:
        print(f'{counter} / {len(reaches)}', end="\r")

        reach_mask = (thiessens['data'] == int(r))

        for z in reach_data[r]['zones']:
            z_mask = np.logical_and((elevations['data'] >= z['min_el']), (elevations['data'] < z['max_el']))
            combo_mask = np.logical_and(reach_mask, z_mask)
            out_data[combo_mask] = iids[z['label']]

        counter += 1

    cols = elevations['cols']
    rows = elevations['rows']
    originX = elevations['origin_x']
    originY = elevations['origin_y']

    driver = gdal.GetDriverByName('GTiff')
    out_path = os.path.join(os.path.dirname(hand_path), f'{out_name}.tif')
    if os.path.exists(out_path):
        os.remove(out_path)
    if os.path.exists(out_path):
        os.remove(out_path + '.aux.xml')
    outRaster = driver.Create(out_path, cols, rows, 1, gdal.GDT_Byte, options=['COMPRESS=LZW'])
    outRaster.SetGeoTransform((originX, elevations['pixel_width'], 0, originY, 0, elevations['pixel_height']))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(out_data.astype(np.uint8))
    outband.SetNoDataValue(0)
    outRaster.SetProjection(elevations['crs'])
    outRaster.FlushCache()
    gdal.Dataset.__swig_destroy__(outRaster)
    outband = outRaster = None

    poly_path = os.path.join(os.path.dirname(os.path.dirname(hand_path)), 'vectors')
    os.makedirs(poly_path, exist_ok=True)
    poly_path = os.path.join(poly_path, f'{out_name}.shp')
    # Load AOI vector
    if len(reaches) == 1:
        query = f"{reach_field} = '{reaches[0]}'"
    elif len(reaches) > 1:
        query = f"{reach_field} in {tuple(reaches)}"
    reach_polys = gpd.read_file(aoi_path, where=query)[[reach_field, 'geometry']]
    reach_polys = reach_polys.dissolve(by=reach_field)
    polygonize_raster(out_path, poly_path, reach_polys, rename_dict=reverse_iids)

    print('')
    print(f'Completed processing in {round(time.perf_counter() - t1, 1)} seconds')
    return out_path

def merge_rasters(paths, out_path):
    tmp_path = out_path.replace('tif', 'vrt')
    gdal.BuildVRT(tmp_path, paths)
    gdal.Translate(out_path, tmp_path, creationOptions=['COMPRESS=LZW'])
    os.remove(tmp_path)

def merge_polygons(paths, out_path):
    print('Loading shapefiles...')
    shp_list = list()  # could condense the next few line into list comprehension, but doing this way so progress can be printed
    for ind, path in enumerate(paths):
        print(f'{ind+1}/{len(paths)}', end='\r')
        shp_list.append(gpd.read_file(path))
    print()
    print('Merging shapefiles...')
    gdf = gpd.GeoDataFrame(pd.concat(shp_list, ignore_index=True))
    gdf.to_file(out_path, driver='ESRI Shapefile')
    print(f'EDZ shapefile saved to {out_path}')

def polygonize_raster(in_path, out_path, reaches=None, rename_dict=None):
    # Open the raster file
    raster = gdal.Open(in_path)
    band = raster.GetRasterBand(1)
    crs = raster.GetProjectionRef()

    # Create an output vector file
    fname = os.path.splitext(out_path)[0]
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(out_path):
        driver.DeleteDataSource(out_path)
    out_datasource = driver.CreateDataSource(out_path)
    out_layer = out_datasource.CreateLayer(fname, srs=None)

    # Add a field to the output shapefile
    new_field = ogr.FieldDefn('DN', ogr.OFTInteger)
    out_layer.CreateField(new_field)

    # Polygonize the raster
    gdal.Polygonize(band, None, out_layer, 0, [], callback=None)

    # Close datasets
    out_datasource = None
    raster = None

    # Clean up polygon
    gdf = gpd.read_file(out_path)
    gdf = gdf[gdf['DN'] != 0]
    if rename_dict is not None:
        gdf['zone'] = gdf['DN'].map(rename_dict)

    if reaches is not None:
        gdf = gpd.GeoDataFrame(gdf).set_crs(crs)
        gdf = gdf.dissolve(by='zone', as_index=False)
        reaches = reaches.to_crs(crs)
        gdf = gdf.overlay(reaches, how='intersection')

    gdf = gdf.drop(columns='DN')

    gdf.to_file(out_path, driver='ESRI Shapefile')

    
