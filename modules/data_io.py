"""
Data I/O functions for MARISA IDF Analysis
Functions for loading model data, processed datasets, and saving results
"""

import time
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import glob
import os
from typing import List, Optional, Union
from .config import MODELS_LOCA2, MODELS_LOCA, MM_TO_INCHES, SECONDS_PER_DAY
from modules import config


def get_atlas14(gdf: gpd.GeoDataFrame, dur: str = '24-hr', out_dir: str = None) -> pd.DataFrame:
    """
    Download NOAA Atlas 14 24-hr precipitation frequency estimates for each
    county in a GeoDataFrame and return a DataFrame with mean, upper,
    and lower confidence intervals.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame of counties. Expected columns: 'NAME', 'STATE', 'FIPS'.
        Geometry must be in a CRS that supports .centroid (e.g. EPSG:4326).
    out_dir : str, optional
        Directory to save downloaded CSV files. If None, uses 'atlas14_cache' in current working directory.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame indexed by (county_name, CI) where CI is one of
        ['mean', 'upper', 'lower']. Columns are ARI return periods (years).
        Intensity values are in inches/day 

    Notes
    -----
    Uses pfdf's atlas14.download()
    pfdf Atlas 14 API docs:
        https://ghsc.code-pages.usgs.gov/lhp/pfdf/api/data/noaa/atlas14.html

    Downloads are cached to a local directory to avoid redundant requests.
    A 1-second sleep is added between requests
    """
    # !pip install pfdf -i https://code.usgs.gov/api/v4/groups/859/-/packages/pypi/simple
    import pfdf
    from pfdf.data.noaa import atlas14

    ##########

    if out_dir is None:
        out_dir = 'atlas14_data'
    os.makedirs(out_dir, exist_ok=True)

    dur = dur+":"

    records = []

    for idx, county in gdf.iterrows():
        try:
            centroid = county.geometry.centroid
            lon, lat = centroid.x, centroid.y

            county_name = county.get('NAME')
            state_name = county.get('STATE')
            county_fips = county.get('FIPS', None)

            csv_path = atlas14.download(
                lat=lat,
                lon=lon,
                series='pds',
                statistic='all',
                data='intensity',
                units='english',
                parent=out_dir,
                name=f"{county_name}_{state_name}_{idx}.csv",
                overwrite=False
            )

            df = pd.read_csv(csv_path, header=11)
            df.set_index('by duration for ARI (years):', inplace=True)

            data = df.loc[dur].copy() * 24
            data.index = pd.Index(['mean', 'upper', 'lower'], name='CI')

            data['county_name'] = county_name
            data['county_fips'] = county_fips
            data['lon'] = lon
            data['lat'] = lat

            records.append(data)
            print(f"Downloaded {county_name}: extracted {len(data)} {dur[:-1]} records")
            time.sleep(1)

        except Exception as e:
            print(f"Error for county {county.get('NAME', idx)}: {e}")
            continue

    if not records:
        raise ValueError("No Atlas 14 data was successfully retrieved.")

    result = (
        pd.concat(records)
        .reset_index()
        .set_index(['county_name', 'CI'])
    )

    return result

def load_loca_precipitation(
    ds: xr.Dataset,
    convert_to_inches: bool = True,
    subset_time: Optional[tuple] = None,
    variable: str = 'pr'
) -> xr.Dataset:
    """
    Load LOCA precipitation data and optionally convert units
    
    Parameters
    ----------
    filepath : str
        Path to the NetCDF file
    convert_to_inches : bool, default=True
        If True, convert from mm/day to inches/day
    subset_time : tuple, optional
        Tuple of (start_date, end_date) to subset time dimension
        
    Returns
    -------
    xr.Dataset
        Precipitation dataset
    """
   
    if convert_to_inches:
      unit = ds[variable].units
      if unit == 'kg m-2 s-1':
        ds[variable] = ds[variable] * SECONDS_PER_DAY * MM_TO_INCHES
      elif unit == 'mm/day':
        ds[variable] = ds[variable] * MM_TO_INCHES
      ds[variable].attrs['units'] = 'inches/day'
    
    if 'Time' in ds.dims:
        ds = ds.rename({'Time': 'time'})

    if subset_time is not None:
        ds = ds.sortby('time')
        ds = ds.sel(time=slice(str(subset_time[0]), str(subset_time[1])))
    
    return ds


def load_model(
  directory: str,
  model: str,
  scenario: str,
  subset_time: str,
  variable: str = 'pr',
  pattern: Optional[str] = None
) -> xr.Dataset:
    """
    Load precipitation data from a single climate model
    
    Parameters
    ----------
    directory : str
        Base directory containing model files
    model : list
        List of model names to load
    scenario : str
        Climate scenario (historical, ssp245, ssp370, ssp585)
    variable : str
        Variable name (default: 'pr')
    pattern : str, optional
        Custom file pattern. If None, uses default pattern
        
    Returns
    -------
    xr
        
    """
    # Load Data
    def preprocess(ds, file):
      filename = os.path.basename(file)  
      parts = filename.split('.')
      ensemble = int(parts[3:4][0][1])
      ds = ds.assign_coords(ensemble=ensemble)
      ds = ds.expand_dims('ensemble')
      return ds

    if pattern:
      file_pattern = pattern
    else:
      file_pattern = f"{directory}/{variable}.{model}.{scenario}.*.nc"

    files = glob.glob(file_pattern)

    if len(files) == 0:
        raise ValueError(f"No files found matching pattern: {file_pattern}")
    elif len(files) == 1:
        ds = xr.open_dataset(
            files[0]
        )
    elif len(files) > 1:
        ds = xr.open_mfdataset(
            files,
            combine='by_coords',
            preprocess=lambda x: preprocess(x, x.encoding['source'])
        )


    ds = load_loca_precipitation(ds, subset_time = subset_time, variable = variable)
    return ds


def load_processed_zarr(
    zarr_path: str,
    chunks: Optional[dict] = None
) -> xr.Dataset:
    """
    Load processed data from Zarr format
    
    Parameters
    ----------
    zarr_path : str
        Path to Zarr store
    chunks : dict, optional
        Chunk specification for dask
        
    Returns
    -------
    xr.Dataset
        Loaded dataset
    """
    if chunks is None:
        return xr.open_zarr(zarr_path)
    else:
        return xr.open_zarr(zarr_path, chunks=chunks)

def save_to_netcdf(
    dataset: xr.Dataset,
    output_path: str,
    compute: bool = True,
    **kwargs
) -> None:
    """
    Save xarray dataset to NetCDF format
    
    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to save
    output_path : str
        Path for output NetCDF file
    compute : bool
        If True, compute immediately
    **kwargs
        Additional arguments passed to to_netcdf
    """
    write_job = dataset.to_netcdf(output_path, compute=compute, **kwargs)
    
    if not compute:
        return write_job
    else:
        print(f"Saved to {output_path}")


def get_valid_grid_point(dataset) -> tuple:
    """
    Find a grid point with maximum non-NaN data coverage
    
    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset
    var : str
        Variable name to check
        
    Returns
    -------
    tuple
        (latitude, longitude) of the point with most data
    """
    non_nan_counts = dataset.count(dim='time')
    lat_idx, lon_idx = np.where(non_nan_counts == non_nan_counts.max())
    
    lat_sel = dataset.y[lat_idx[0]].values
    lon_sel = dataset.x[lon_idx[0]].values
    
    return lat_sel, lon_sel


def export_to_csv(
    data: Union[pd.DataFrame, xr.Dataset],
    output_path: str,
    **kwargs
) -> None:
    """
    Export data to CSV format
    
    Parameters
    ----------
    data : pd.DataFrame or xr.Dataset
        Data to export
    output_path : str
        Output file path
    **kwargs
        Additional arguments for to_csv
    """
    if isinstance(data, xr.Dataset):
        data = data.to_dataframe()
    
    data.to_csv(output_path, **kwargs)
    print(f"Exported to {output_path}")

def load_counties(crs = None):
  import geopandas as gpd
  from pathlib import Path
   # Get the directory where your notebook/script is located
  NOTEBOOK_DIR = Path.cwd()
  COUNTY_SHAPEFILE = f'{NOTEBOOK_DIR}/data/Boundaries/MARISA_domain.shp'

  gdf_counties = gpd.read_file(COUNTY_SHAPEFILE)
  gdf_counties['STATE'] = gdf_counties['STATEFP'].map({'42': 'PA', '24': 'MD', '11': 'DC', '51': 'VA', '10' : 'DE', '36': 'NY', '54': 'WV'})
  if crs is not None:
    gdf_counties = gdf_counties.to_crs(crs)
  else:
    gdf_counties = gdf_counties.to_crs(epsg=4326)
  
  return gdf_counties

def combine_models(ver, suffix, FINAL_DIR = "/content/drive/MyDrive/Research/MARISA_IDF/data/FINAL" , zarr_vars = None, save_vars = None ):
    """
    Combine processed Zarr files across models and scenarios into a single dataset
    """

    import os
    time_periods = ['1950-2014', '2020-2070', '2050-2100']
    if ver == 'LOCA2':
        DIRECTORY = "/content/drive/MyDrive/Research/MARISA_IDF/data/LOCA2/MARISA/"
        SAVE_DIR = "/content/drive/MyDrive/Research/MARISA_IDF/data/FINAL/LOCA2/Processed"
        #scenarios = ['ssp245', 'ssp370', 'ssp585']
        scenarios = ['ssp245']
        models = config.MODELS_LOCA2
    elif ver == 'LOCA':
        DIRECTORY = "/content/drive/MyDrive/Research/MARISA_IDF/data/LOCA/"
        SAVE_DIR = "/content/drive/MyDrive/Research/MARISA_IDF/data/FINAL/LOCA"
        scenarios = ['rcp45', 'rcp85']
        models = config.MODELS_LOCA

    #zarr_vars = ['adj_factor', 'return_precip', 'GEV_paras']
    #save_vars = ['adj_factors', 'return_precip', 'gevParas']


    for scenario in scenarios:
        files = glob.glob(f'{DIRECTORY}/*.{scenario}.*_processed.zarr')
        valid_files = []
        for file in files:
            zarr_path = os.path.join(file, zarr_vars[0]) if zarr_vars else file
            if os.path.exists(zarr_path):  # Check if return_periods exists in zarr structure
                valid_files.append(file)
            else:
                print(f"Skipping {file} - no data")

        #temp = xr.open_mfdataset(valid_files, combine='nested',
        #                                    concat_dim='model',
        #                                    consolidated=False,
        #                                    errors = 'ignore')
        
        temp = xr.open_mfdataset(valid_files, combine='by_coords',
                                            consolidated=False,
                                            errors = 'ignore')

        temp.to_zarr(f'{FINAL_DIR}/baseline_data_combined_{ver}_{scenario}_{suffix}.zarr', zarr_format=2, consolidated = False, mode='w')
        print(f"Saved combined dataset for {scenario} to {FINAL_DIR}/baseline_data_combined_{ver}_{scenario}_{suffix}.zarr")
    return True