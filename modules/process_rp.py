"""
Statistical functions for MARISA IDF Analysis
IDF curve inversion, GEV fitting, and statistical tests
"""

from tabnanny import check
from zipfile import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.stats import genextreme
import lmoments3 as lm
from lmoments3 import distr

from modules import data_io, config

import geopandas as gpd
import regionmask
from rasterio import features
from shapely.geometry import mapping

from typing import Optional, Literal, Union

### Main function to calculate adjustment factors and return period values for a given model, scenario, and time period ###
def calc_adj_factors(model, scenario, subset_time, DIRECTORY, n_b = 100, s_var = 'pds_peak_values', regionalization = True):
  period = f"{str(subset_time[0])}-{str(subset_time[1])}"

  result_hist = xr.open_dataset(f'{DIRECTORY}/{model}.historical.{str(1950)}-{str(2000)}_processed.zarr', zarr_format=2, consolidated=False)
  dims_stack = [d for d in result_hist.dims if d not in ['x', 'y', 'year']]
  
  if regionalization:
    #r_hist, gev_paras_log_hist = lmoments_county(result_hist[s_var])
    r_hist, gev_paras_log_hist = calc_adj_factors_centroid(result_hist, s_var, n_b = n_b)
  else:
    r_hist, gev_paras_log_hist = process_rp_values(result_hist[s_var], dims_stack = dims_stack)


  result = xr.open_dataset(f'{DIRECTORY}/{model}.{scenario}.{str(subset_time[0])}-{str(subset_time[1])}_processed.zarr', zarr_format=2, consolidated=False)
  dims_stack = [d for d in result.dims if d not in ['x', 'y', 'year']]
  if regionalization:
    r_mod, gev_paras_log_mod = calc_adj_factors_centroid(result, s_var, n_b = n_b)
  else:
    r_mod, gev_paras_log_mod = process_rp_values(result[s_var], dims_stack = dims_stack)

  r = r_mod/r_hist

  r = r.expand_dims({'model': [model], 'scenario': [scenario], 'time_period' : [period]})
  r_mod = r_mod.expand_dims({'model': [model], 'scenario': [scenario], 'time_period' : [period]})
  gev_mod = gev_paras_log_mod.expand_dims({'model': [model], 'scenario': [scenario], 'time_period' : [period]})

  r.name = 'adj_factor'
  r_mod.name = 'return_precip'
  gev_mod.name = 'GEV_parameters'
  return r, r_mod, gev_mod
def calc_adj_factors_centroid(ds, var_name, n_b, lat_dim='lat', lon_dim='lon', fit_type = 'regional_lmom'):
    from pathlib import Path
    import lmoments3 as lm
    from lmoments3 import distr
    import regionmask
    import geopandas as gpd

    def county_centroid_gdf():
      #Load Counties
      gdf_counties = data_io.load_counties()
      #Calculate centroids

      gdf_projected = gdf_counties.to_crs("EPSG:5070")  # Project to a metric CRS for accurate distance calculations
      centroids = gdf_projected.geometry.centroid  # GeoSeries in EPSG:5070
      gdf_counties['centroid'] = gpd.GeoSeries(centroids, crs="EPSG:5070").to_crs("EPSG:4326")

      return gdf_counties

    def gdf_xarray(gdf):

      county_ids = gdf.index.astype(str) 
      centroids = gdf['centroid']
      ds = xr.Dataset(
          coords={
              "county": county_ids,
              "lat": ("county", centroids.y.values),
              "lon": ("county", centroids.x.values),
          }
      )

      # Attach geometry as a non-index coordinate for reference
      ds = ds.assign_coords(geometry=("county", gdf.geometry.values))
      return ds

    def fit_gev_shape_scale(data):
      """
      Fit a GEV distribution to the data using L-moments 
      return: shape and scale parameters 
      """
      try:
          fit_paras = distr.gev.lmom_fit(data)
          shape = -fit_paras['c']
          scale = fit_paras['scale']
      except:
          shape = np.nan
          scale = np.nan
      return shape, scale
    
    def safe_lmom(x):
      """
      Compute L-moments for a given array, handling missing values.
      """
      x_valid = x[np.isfinite(x)]
      try:
          return np.array(lm.lmom_ratios(x_valid, nmom=4))
      except Exception:
          return np.full(4, np.nan)
                    
    def extract_7x7(da_gridded, lons, lats):
      """
      Extract 7x7 windows around each county centroid.
      da_gridded: (y, x) or (ensemble, peak, y, x)
      lons, lats: (county,)
      returns: (county, ensemble, peak, cell) or (county, peak, cell) where cell is the flattened 7x7 window
      """
      half = 3
      windows = []
      
      for county_id, lon, lat in zip(ds_county['county'].values, lons, lats):
          i_lat = np.abs(da_gridded.y - lat).argmin().item()
          i_lon = np.abs(da_gridded.x - lon).argmin().item()
          
          y_start = max(0, i_lat - half)
          y_stop  = min(da_gridded.sizes['y'], i_lat + half + 1)
          x_start = max(0, i_lon - half)
          x_stop  = min(da_gridded.sizes['x'], i_lon + half + 1)
          
          window = da_gridded.isel(
              y=slice(y_start, y_stop),
              x=slice(x_start, x_stop),
          )
          window = window.pad(
              y=(max(0, half - i_lat), max(0, (i_lat + half + 1) - da_gridded.sizes['y'])),
              x=(max(0, half - i_lon), max(0, (i_lon + half + 1) - da_gridded.sizes['x'])),
              mode='constant', constant_values=np.nan
          )
          if 'ensemble' in da_gridded.dims:
              window = window.transpose('ensemble', 'peak', 'y', 'x')  # ensure correct order
          else:
              window = window.transpose('peak', 'y', 'x')  # ensure correct order
          values = window.values                                     # (ensemble, peak, 7, 7)
          values_flat = values.reshape(*values.shape[:-2], -1)      # (ensemble, peak, 49)
          windows.append(values_flat)
      
      if 'ensemble' in da_gridded.dims:
        return xr.DataArray(
          np.stack(windows, axis=0),
          dims=['county', 'ensemble', 'peak', 'cell'],
          coords={'county': ds_county['county'].values}
      )
      else:
        return xr.DataArray(
          np.stack(windows, axis=0),
          dims=['county', 'peak', 'cell'],
          coords={'county': ds_county['county'].values}
      )
    def fit_gev_grid(lmoms_4d):
      """
      Fit GEV to each cell in a 5x5 grid from pre-computed L-moments.
      lmoms_4d: (4, 5, 5) where dim 0 is [L1, L2, t3, t4]
      returns: c, loc, scale each (5, 5)
      """
      ny, nx = lmoms_4d.shape[1], lmoms_4d.shape[2]
      c_out     = np.full((ny, nx), np.nan)
      loc_out   = np.full((ny, nx), np.nan)
      scale_out = np.full((ny, nx), np.nan)

      for i in range(ny):
          for j in range(nx):
              lmoms = lmoms_4d[:, i, j]
              if not np.isfinite(lmoms).all():
                  continue
              try:
                  paras = distr.gev.lmom_fit(lmom_ratios=lmoms.tolist())
                  c_out[i, j]     = paras['c']
                  loc_out[i, j]   = paras['loc']
                  scale_out[i, j] = paras['scale']
              except:
                  pass

      return c_out, loc_out, scale_out

    def ds_5x5_lmom(data, n_b=100):
      """
      input:
      data: 7x7 grid of data around each county centroid (ensemble, peak, 49) or (peak, 49) 
      n_b: number of bootstrap iterations

      computes L-moments for the 5x5 grid around the focal cell, 
      fits a GEV distribution to those moments, 
      bootstraps return period thresholds from that fitted distribution. 

      return:
      r_out: (9, n_b+1, n_return_periods) return period thresholds for each of the 9 cells in the 3x3 grid around the centroid (including the centroid itself)
      p_out: (9, n_b+1, 3) GEV parameters (shape, loc, scale) for each of the 9 cells in the 3x3 grid around the centroid (including the centroid itself)) for each bootstrap iteration
      """

      # Flatten ensemble and peak into single time axis
      data = data.reshape(-1, data.shape[-1])  # (ensemble*peak, 49)
      data_2d = data.reshape(data.shape[0], 7, 7)  # (time, 7, 7)

      all_r, all_params = [], []
      #For each cell in a 3x3 (for smoothing at the end)
      for i in range(2, 5):
          for j in range(2, 5):
              focal_cell = data_2d[:, i, j]               # (time,)
              pool = data_2d[:, i-2:i+3, j-2:j+3]        # (time, 5, 5)

              if fit_type == "regional_lmom":
                #Regional L-moments: Compute L-moments for the 5x5 grid around the focal cell and fit a GEV to those moments
                pool = data_2d[:, i-2:i+3, j-2:j+3]             # (time, 5, 5)
                data_flat = pool.reshape(pool.shape[0], -1)       # (time, 25)
                #Compute L-Moments for each cell in the 5x5 grid
                lmoms = np.apply_along_axis(safe_lmom, axis=0, arr=data_flat)  # (4, 25)
                #Smoothing: Handle any NaN values in L-moments by replacing them with the mean of the non-NaN values across the grid 
                lmoms[0] = lmoms[0][np.isnan(lmoms[0])] = np.nanmean(lmoms[0])
                lmoms[1] = lmoms[1][np.isnan(lmoms[1])] = np.nanmean(lmoms[1])
                #Smooth higher order moments (mean across grid) to ensure stable fitting even if some cells have extreme values or insufficient data
                lmoms[2] = np.ones(lmoms[2].shape)*np.nanmean(lmoms[2])
                lmoms[3] = np.ones(lmoms[3].shape)*np.nanmean(lmoms[3])

                lmoms_4d = lmoms.reshape(4, 5, 5)
                shape_grid, loc_grid, scale_grid = fit_gev_grid(lmoms_4d)

              elif fit_type == "regional_stack":
                #Stack all values in the 5x5 grid to fit a single GEV (Increases sample size for fitting, but assumes all cells are from the same distribution, ignores spatial dependence, leads to overconfident estimates)
                pool = pool.reshape(-1)       # (time, 25)
                valid = np.isfinite(pool)       # (25,)
                pool = pool[valid]                        # (time, n_valid)
                shape, scale = fit_gev_shape_scale(pool)     
              
              #Baseline GEV parameters for bootstrapping (use focal cell parameters if regional stack, or 3x3 smoothed parameters if regional_lmom)
              fit_paras_base = {'c': shape_grid[2,2], 'loc': loc_grid[2,2], 'scale': scale_grid[2,2]}
              # Bootstap return period thresholds, GEV parameters from the baseline GEV distribution
              r, gev_params = bootstrap_gev(n_b, fit_paras_base, num_years=50)
              all_r.append(np.asarray(r))
              all_params.append(np.asarray(gev_params))
      r_out = np.stack(all_r, axis=0)        # (9, n_b+1, n_return_periods)
      p_out = np.stack(all_params, axis=0)   # (9, n_b+1, 3)
      return r_out, p_out
    
    ds_county = gdf_xarray(county_centroid_gdf())
        
    da_gridded = ds[var_name] if isinstance(ds, xr.Dataset) else ds

    da_pooled = extract_7x7(da_gridded, ds_county['lon'].values, ds_county['lat'].values)

    if 'ensemble' in da_pooled.dims:
      in_dims = ['ensemble', 'peak', 'cell']
    else:
      in_dims = ['peak', 'cell']

    r, gev_params = xr.apply_ufunc(
      ds_5x5_lmom,
      da_pooled,
      kwargs={"n_b": n_b},
      input_core_dims=[in_dims],
      output_core_dims=[['centroid_cell', 'n_b', 'return_periods'], 
                        ['centroid_cell', 'n_b', 'GEV_paras']],
      output_dtypes=[np.float64, np.float64],
      vectorize=True,
      dask="parallelized",
      dask_gufunc_kwargs={'output_sizes': {
          'return_periods': 6, 
          'n_b': n_b+1, 
          'GEV_paras': 3,
          'centroid_cell': 9
      }}
      )

      

    gev_params = gev_params.assign_coords(GEV_paras = ['shape', 'loc', 'scale']) 
    r = r.assign_coords(return_periods = [2, 5, 10, 25, 50, 100]) 

    return r, gev_params

def county_centroid(ds, gdf_counties):
  """
  Assigns each county the grid cell closest to its centroid as its representative location for GEV fitting and return period calculation. 

  """
  import numpy as np
  from scipy.spatial import cKDTree

  # stack grid coordinates
  da_stacked = ds.stack(location=['y', 'x'])
  all_lons_2d, all_lats_2d = np.meshgrid(ds.x.values, ds.y.values)
  all_lats = all_lats_2d.ravel()
  all_lons = all_lons_2d.ravel()

  tree = cKDTree(np.column_stack([all_lats, all_lons]))

  # find nearest grid cell to each county centroid
  gdf_projected = gdf_counties.to_crs("EPSG:5070")  # Project to a metric CRS for accurate distance calculations
  centroids = gdf_projected.geometry.centroid  # GeoSeries in EPSG:5070
  centroids_wgs84 = gpd.GeoSeries(centroids, crs="EPSG:5070").to_crs("EPSG:4326")
  centroid_coords = np.column_stack([centroids_wgs84.y, centroids_wgs84.x])

  _, nearest_idx = tree.query(centroid_coords)

  # build mapping
  gdf_counties["rep_location"] = da_stacked.location.values[nearest_idx]

  # select those cells from the dataset
  rep_locations = gdf_counties["rep_location"].unique()
  ds_county_rep = da_stacked.sel(location=rep_locations)
  return ds_county_rep

def thin_grid(ds, dist = 50):
  #Thin grid to maintain independence of grid cells for statistical analysis (e.g., GEV fitting)

  #Randomly select an offset for lat and lon to ensure different grid cells are selected each time
  stride = round(dist / 6) #Assuming original grid is 6km, adjust stride to achieve desired distance between grid cells
  rng = np.random.default_rng(seed=42)
  lat_offset = rng.integers(0, stride)
  lon_offset = rng.integers(0, stride)

  ds_thinned = ds.isel(lat=slice(lat_offset, None, stride),
                      lon=slice(lon_offset, None, stride))
  return ds_thinned

def bootstrap_gev(n_b, fit_paras_base, num_years = 50, return_periods = config.RETURN_PERIODS):
  #From a baseline GEV, resample w/t data length
  """
  input:
  n_b: number of bootstrap iterations
  fit_paras_base: parameters of the baseline GEV distribution to resample from
  num_years: length of the data record to resample (e.g., 50 years)
  return_periods: list of return periods to calculate thresholds for
  
  returns: 
  thresholds_cp (n_b+1, n_return_periods)
  parameters_cp (n_b+1, 3)
  """
  
  thresholds_cp = np.zeros([(n_b+1), len(return_periods)])
  parameters_cp = np.zeros([(n_b+1), 3])

  fitted_gev_base = distr.gev(**fit_paras_base)
  #New PDS (50-yr record)
  thresholds = []
  parameters_log = []
  parameters_log = np.append(parameters_log, list(fit_paras_base.values()))

  for rp in return_periods:
    exceedance_prob = 1 / rp  # Probability of exceeding this threshold in a given year
    threshold_base_hist = fitted_gev_base.ppf(1 - exceedance_prob)
    thresholds = np.append(thresholds, threshold_base_hist)

  #Resample the fitted gev n_b times
  for b in range(n_b):
    try:
      pds_sample = fitted_gev_base.rvs(size=num_years)
      fit_paras = distr.gev.lmom_fit(pds_sample) #Fit the distribution to the data
      fitted_gev = distr.gev(**fit_paras)
      parameters_log = np.append(parameters_log, list(fit_paras.values()))

      for rp in return_periods:
        exceedance_prob = 1 / rp  # Probability of exceeding this threshold in a given year
        thresholds = np.append(thresholds, fitted_gev.ppf(1 - exceedance_prob))
    except:
      for rp in return_periods:
        exceedance_prob = 1 / rp  # Probability of exceeding this threshold in a given year
        thresholds = np.append(thresholds, np.nan)
  threshold = thresholds

  # Store the computed threshold
  thresholds_cp = np.array(threshold, dtype=float).reshape((n_b+1), len(return_periods))
  # Store GEV parameters
  if len(parameters_log) <= 3:
    parameters_log = np.full(((n_b+1), 3), np.nan)
  parameters_cp = np.array(parameters_log, dtype=float).reshape((n_b+1), 3)

  return thresholds_cp, parameters_cp

#### Main function to calculate return period values for a given dataset using either empirical or L-moments fitting. Returns both the return period thresholds and the GEV parameters for each bootstrap iteration. ###
def calc_rp_values(data, 
                   type = 'lmom',
                   return_periods = config.RETURN_PERIODS,
                   duration= ['24-hr'],
                   bootstrap=100):
  """
  Calculates return period values for precipitation data.

  Args:
    data: A pandas Series containing DAILY precipitation values.

  Returns:
    A pandas DataFrame containing return period values for durations and return periods.
  """
  data = data[~np.isnan(data)]
  #Initialize DataFrame to return threshold values
  num_years = int(data.size/365)
  p_i = np.arange(1,num_years+1)/(num_years+1)
  ### Bootstrap Parameters ###

  #Initialize Bootstrap Instances
  if bootstrap>0:
    n_b = bootstrap
    thresholds_cp = np.zeros([(n_b+1), len(return_periods)])
    parameters_cp = np.zeros([(n_b+1), 3])
  else:
    thresholds_cp = np.zeros(len(return_periods))

  i=0
  ### Empirical Weibull Fit ###
  if type == 'empirical':
    # Compute threshold using Weibull dist. percentile (inverse of exceedance probability) using first n-values of precipitation given n-year long record
    #threshold = np.percentile(r_sum[:num_years], 100 - (exceedance_prob * 100))
    threshold = data[np.where(p_i <= exceedance_prob)].min().item()
  ### L-Moments Fit ###
  elif type == 'lmom':
    #Compute threshold according to L-moments fitted GEV distribution
    #Resample the fitted gev 1000 times
    if bootstrap > 0:
      fit_paras_base = distr.gev.lmom_fit(data) #Fit the distribution to the data
      fitted_gev_base = distr.gev(**fit_paras_base)

      #New PDS (50-yr record)
      thresholds = []
      parameters_log = []
      parameters_log = np.append(parameters_log, list(fit_paras_base.values()))

      for rp in return_periods:
        exceedance_prob = 1 / rp  # Probability of exceeding this threshold in a given year
        threshold_base_hist = fitted_gev_base.ppf(1 - exceedance_prob)
        thresholds = np.append(thresholds, threshold_base_hist)

      #Resample the fitted gev n_b times
      for b in range(n_b):
        try:
          pds_sample = fitted_gev_base.rvs(size=num_years)
          fit_paras = distr.gev.lmom_fit(pds_sample) #Fit the distribution to the data
          fitted_gev = distr.gev(**fit_paras)
          parameters_log = np.append(parameters_log, list(fit_paras.values()))

          for rp in return_periods:
            exceedance_prob = 1 / rp  # Probability of exceeding this threshold in a given year
            thresholds = np.append(thresholds, fitted_gev.ppf(1 - exceedance_prob))
        except:
          for rp in return_periods:
            exceedance_prob = 1 / rp  # Probability of exceeding this threshold in a given year
            thresholds = np.append(thresholds, np.nan)
      threshold = thresholds
    else:
      fit_paras = distr.gev.lmom_fit(data) #Fit the distribution to the data
      fitted_gev = distr.gev(**fit_paras)
      parameters_log = list(fit_paras.values())
      for rp in return_periods:
        exceedance_prob = 1 / rp  # Probability of exceeding this threshold in a given year
        threshold = np.append(thresholds, fitted_gev.ppf(1 - exceedance_prob))
  ### Maximum Likelihood Fit ###
  elif type == 'mle':
    #Compute threshold according to MLE fitted GEV distribution
    fit_paras_mle = genextreme.fit(data) #Fit the distribution to the data
    #fit_paras_mle = uniform_MLE(r_sum[:num_years].fillna(0).values) #Fit uniform bin_MLE
    threshold = genextreme.ppf(1 - exceedance_prob, *fit_paras_mle)

  # Store the computed threshold
  thresholds_cp = np.array(threshold, dtype=float).reshape((n_b+1), len(return_periods))
  # Store GEV parameters
  if len(parameters_log) <= 3:
    parameters_log = np.full(((n_b+1), 3), np.nan)
  parameters_cp = np.array(parameters_log, dtype=float).reshape((n_b+1), 3)

  return thresholds_cp, parameters_cp

def input_data_byCounty(xarray_data):
  #Load Counties
  from pathlib import Path

# Get the directory where your notebook/script is located
  NOTEBOOK_DIR = Path.cwd()
  COUNTY_SHAPEFILE = f'{NOTEBOOK_DIR}/data/Boundaries/MARISA_domain.shp'

  #Load Counties
  gdf_counties = data_io.load_counties(crs = xarray_data.rio.crs)
  
  # Create mask
  mask = regionmask.mask_geopandas(
      gdf_counties,
      lon_or_obj=xarray_data.x,
      lat=xarray_data.y
  )

  # Create dataset
  ds = xarray_data
  # Broadcast mask
  mask_broadcast = mask.broadcast_like(ds)
  ds['county'] = mask_broadcast

  # Group by county only
  grouped = ds.groupby('county')

  def concat_county_peaks(group):
      """Flatten all dimensions into one"""
      # Just flatten the data, ignore coordinate structure
      flat_data = group.values.flatten()
      
      # Create new DataArray with simple index
      return xr.DataArray(
          flat_data,
          dims=['peak'],
          coords={'peak': range(len(flat_data))}
      )

  result = grouped.map(concat_county_peaks)
  result = result.rename({'peak': 'time'})
  return result

def lmoments_county(xarray_data, n_b = 100, thin = 'centroid', lmom_fit = True):
  #Load Counties
  from pathlib import Path
  import lmoments3 as lm
  from lmoments3 import distr
  import regionmask

  def lmom_grid(data):
    #Calculate L-moments for each grid cell
    return np.array(lm.lmom_ratios(data, nmom=4))

  def fit_gev_lmom(lmoms, n_b):
    #Fit GEV distribution using L-moments and calculate return period thresholds
    gev_paras = distr.gev.lmom_fit(lmoms)

    thresholds_cp, parameters_cp = bootstrap_gev(n_b, gev_paras)

    return thresholds_cp, parameters_cp
  def concat_ensemble(ds):
      """Flatten all dimensions into one"""
      # Just flatten the data, ignore coordinate structure
      flat_data = ds.values.flatten()
      
      # Create new DataArray with simple index
      return xr.DataArray(
          flat_data,
          dims=['peak'],
          coords={'peak': range(len(flat_data))}
      )
  #Load Counties
  gdf_counties = data_io.load_counties(crs = xarray_data.rio.crs)

  if 'ensemble' in xarray_data.dims:
      # If ensemble dimension exists, concatenate it with time to create a single dimension for L-moment calculation
      da_by_county = xarray_data.map(concat_ensemble, input_core_dims=[['ensemble', 'time']], output_core_dims=[['time']])
  
  if 'peak' in xarray_data.dims:
    xarray_data = xarray_data.rename({'peak': 'time'})
  
  if thin == 'grid':
    xarray_data = thin_grid(xarray_data, dist = 50)
  elif thin == 'centroid':
    xarray_county = county_centroid(xarray_data, gdf_counties)

  if lmom_fit:
    #Calculate lmoments for each grid cell
    lmom = xr.apply_ufunc(
      lmom_grid, 
      xarray_county, 
      xarray_data,
      input_core_dims=[['time']], 
      output_core_dims=[['moments']],
      output_dtypes=[np.float64], 
      exclude_dims={'time'}, 
      vectorize=True, 
      dask="parallelized",
      dask_gufunc_kwargs={'output_sizes': {'moments': 4}}
      )
    
    lmom = lmom.assign_coords(moments = ['L1', 'L2', 'L3', 'L4'])

    ds = lmom

    
    r, gev_params = xr.apply_ufunc(
        fit_gev_lmom, 
        ds,
        kwargs={"n_b": n_b},
        input_core_dims=[['moments']], 
        output_core_dims=[(['n_b', 'return_periods']), (['n_b','GEV_paras'])],
        output_dtypes=[np.float64, np.float64], 
        vectorize=True, 
        dask="parallelized",
        dask_gufunc_kwargs={'output_sizes': {'return_periods': 6, 'n_b': n_b+1, 'GEV_paras': 3}}
    )
  gev_params = gev_params.assign_coords(GEV_paras = ['shape', 'loc', 'scale']) 
  r = r.assign_coords(return_periods = [2, 5, 10, 25, 50, 100]) 

  r = r.rename({'location': 'county'})
  gev_params = gev_params.rename({'location': 'county'})

  return r, gev_params

### Alternative to lmoments_county that applies GEV fitting and return period calculation directly to the data without calculating L-moments first. This is faster but less robust, and may fail for some grid cells with insufficient data or extreme values ###
def process_rp_values(data, n_b, dims_stack = []):
  data = data.compute()
  input_data = input_data_byCounty(data)
  r, gev_paras_log = xr.apply_ufunc(
    calc_rp_values, 
    input_data, 
    kwargs={"n_b": n_b},
    input_core_dims=[['time']], 
    output_core_dims=[(['n_b', 'return_periods']), (['n_b', 'GEV_paras'])],
    output_dtypes=[np.float64, np.float64], 
    vectorize=True, 
    dask="parallelized",
    dask_gufunc_kwargs={'output_sizes': {'return_periods': 6, 'n_b': n_b+1, 'GEV_paras': 3}}
    )
  gev_paras_log = gev_paras_log.assign_coords(GEV_paras = ['c', 'loc', 'scale'])
  r = r.assign_coords(return_periods = [2, 5, 10, 25, 50, 100])

  return r, gev_paras_log













      


