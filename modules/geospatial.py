"""
Geospatial functions for MARISA IDF Analysis
County statistics, spatial smoothing, and rasterization
"""

import numpy as np
import xarray as xr
import geopandas as gpd
import regionmask
from rasterio import features
from shapely.geometry import mapping
from typing import Optional, Union

from modules import config, data_io


def smooth_spatial_rolling(
    xarray_data: xr.DataArray,
    window_size: int = 5
) -> xr.DataArray:
    """
    Apply rolling mean spatial smoothing
    Works efficiently with dask arrays
    
    Parameters
    ----------
    xarray_data : xr.DataArray
        Input data with spatial dimensions (x, y)
    window_size : int
        Size of rolling window (default: 5)
        
    Returns
    -------
    xr.DataArray
        Smoothed data
    """
    # Apply rolling mean on spatial dimensions
    smoothed = xarray_data.rolling(
        y=window_size,
        x=window_size,
        center=True
    ).mean()
    
    # Fill NaNs at edges with original values
    smoothed = smoothed.fillna(xarray_data)
    
    return smoothed


def calculate_county_stats_xarr(xarray_data):
    """
    Most efficient version using xarray groupby directly
    """
    all_results = []

    for rp in xarray_data.return_periods.values:
        print(f"\nProcessing {rp}-year return period...")

        # Select this return period
        rp_data = xarray_data.sel(return_periods=rp)


        dims = list(rp_data.dims)[0]
        # Calculate stats
        mean_vals = rp_data.mean(dim=dims)
        quantile_vals = rp_data.quantile(
            [0.1, 0.25, 0.5, 0.75, 0.9],
            dim=dims
        )


        mean_vals = mean_vals.compute()
        quantile_vals = quantile_vals.compute()

        # Extract individual quantiles
        p10 = quantile_vals.sel(quantile=0.1)
        p25 = quantile_vals.sel(quantile=0.25)
        p50 = quantile_vals.sel(quantile=0.5)
        p75 = quantile_vals.sel(quantile=0.75)
        p90 = quantile_vals.sel(quantile=0.9)

        # Convert to dataframe
        df = pd.DataFrame({
            'county': mean_vals.county.values,
            f'{rp}-yr mean': mean_vals.values,
            f'{rp}-yr 10th ptile': p10.values,
            f'{rp}-yr 25th ptile': p25.values,
            f'{rp}-yr 50th ptile': p50.values,
            f'{rp}-yr 75th ptile': p75.values,
            f'{rp}-yr 90th ptile': p90.values,
        })

        all_results.append(df.set_index('county'))

    # Combine all return periods
    result_df = pd.concat(all_results, axis=1).reset_index()
    result_df = result_df.rename(columns={'index': 'county_mask'})

    # Merge with geodataframe
    COUNTY_SHAPEFILE = '/content/Adj_IDF/data/Boundaries/MARISA_domain.shp'

    gdf_counties = gpd.read_file(COUNTY_SHAPEFILE)
    gdf_counties['STATE'] = gdf_counties['STATEFP'].map({'42': 'PA', '24': 'MD', '11': 'DC', '51': 'VA', '10' : 'DE', '36': 'NY', '54': 'WV'})

    gdf_counties['COUNTY_ID'] = gdf_counties.index
    gdf_merged = gdf_counties.merge(df, left_on='COUNTY_ID', right_on='county', how = 'left')
    gdf_merged = interp_nearest_neighbors(gdf_merged)
    
    return gdf_merged

def calculate_county_stats_optimized(
    xarray_data: xr.DataArray,
    gdf_counties: gpd.GeoDataFrame,
    stats: list = ['mean', 'std', 'min', 'max']
) -> gpd.GeoDataFrame:
    """
    Calculate statistics for each county using xarray groupby
    Efficient version using regionmask
    
    Parameters
    ----------
    xarray_data : xr.DataArray
        Gridded data with spatial dimensions
    gdf_counties : gpd.GeoDataFrame
        County boundaries
    stats : list
        Statistics to calculate (default: ['mean', 'std', 'min', 'max'])
        
    Returns
    -------
    gpd.GeoDataFrame
        Counties with calculated statistics
    """
    # Reproject counties to match data CRS
    gdf_counties = gdf_counties.to_crs(xarray_data.rio.crs)
    
    # Create mask for counties
    mask = regionmask.mask_geopandas(
        gdf_counties,
        lon_or_obj=xarray_data.x,
        lat=xarray_data.y
    )
    
    all_results = []
    
    # Smooth data if it has high resolution
    if len(xarray_data.x) > 1000 or len(xarray_data.y) > 1000:
        xarray_data = smooth_spatial_rolling(xarray_data, window_size=3)
    
    # Calculate statistics for each county
    for idx in range(len(gdf_counties)):
        county_mask = (mask == idx)
        county_data = xarray_data.where(county_mask, drop=True)
        
        result = {'county_idx': idx}
        
        if 'mean' in stats:
            result['mean'] = float(county_data.mean().values)
        if 'std' in stats:
            result['std'] = float(county_data.std().values)
        if 'min' in stats:
            result['min'] = float(county_data.min().values)
        if 'max' in stats:
            result['max'] = float(county_data.max().values)
        if 'median' in stats:
            result['median'] = float(county_data.median().values)
        if 'count' in stats:
            result['count'] = int(county_data.count().values)
        
        all_results.append(result)
    
    # Merge results back to GeoDataFrame
    stats_df = pd.DataFrame(all_results)
    gdf_result = gdf_counties.copy()
    
    for col in stats_df.columns:
        if col != 'county_idx':
            gdf_result[col] = stats_df[col].values
    
    return gdf_result


def rasterize_shapes(
    gdf: gpd.GeoDataFrame,
    reference_data: xr.DataArray,
    column: Optional[str] = None,
    fill_value: float = np.nan,
    all_touched: bool = False
) -> xr.DataArray:
    """
    Rasterize vector shapes to match a reference grid
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Vector data to rasterize
    reference_data : xr.DataArray
        Reference grid to match
    column : str, optional
        Column to use for raster values (if None, uses sequential integers)
    fill_value : float
        Value for pixels outside shapes
    all_touched : bool
        If True, all pixels touched by geometries will be burned
        
    Returns
    -------
    xr.DataArray
        Rasterized data
    """
    # Ensure CRS match
    gdf = gdf.to_crs(reference_data.rio.crs)
    
    # Get transform and shape from reference
    transform = reference_data.rio.transform()
    shape = reference_data.shape
    
    # Prepare shapes for rasterization
    if column is not None:
        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[column]))
    else:
        shapes = ((geom, idx) for idx, geom in enumerate(gdf.geometry))
    
    # Rasterize
    rasterized = features.rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=fill_value,
        all_touched=all_touched,
        dtype='float32'
    )
    
    # Create DataArray
    result = xr.DataArray(
        rasterized,
        coords={'y': reference_data.y, 'x': reference_data.x},
        dims=['y', 'x']
    )
    
    return result


def extract_point_values(
    xarray_data: xr.DataArray,
    points: gpd.GeoDataFrame,
    method: str = 'nearest'
) -> gpd.GeoDataFrame:
    """
    Extract values from gridded data at point locations
    
    Parameters
    ----------
    xarray_data : xr.DataArray
        Gridded data
    points : gpd.GeoDataFrame
        Point locations
    method : str
        Interpolation method ('nearest' or 'linear')
        
    Returns
    -------
    gpd.GeoDataFrame
        Points with extracted values
    """
    # Ensure CRS match
    points = points.to_crs(xarray_data.rio.crs)
    
    # Extract coordinates
    x_coords = points.geometry.x.values
    y_coords = points.geometry.y.values
    
    # Extract values
    values = xarray_data.sel(x=xr.DataArray(x_coords), 
                            y=xr.DataArray(y_coords),
                            method=method)
    
    # Add to GeoDataFrame
    points_result = points.copy()
    points_result['extracted_value'] = values.values
    
    return points_result


def calculate_spatial_percentiles(
    xarray_data: xr.DataArray,
    percentiles: list = [10, 50, 90],
    dim: Union[str, list] = 'time'
) -> xr.Dataset:
    """
    Calculate spatial percentiles over a dimension
    
    Parameters
    ----------
    xarray_data : xr.DataArray
        Input data
    percentiles : list
        Percentiles to calculate
    dim : str or list
        Dimension(s) over which to calculate percentiles
        
    Returns
    -------
    xr.Dataset
        Dataset with percentile values
    """
    result_dict = {}
    
    for p in percentiles:
        result_dict[f'p{p}'] = xarray_data.quantile(p/100, dim=dim)
    
    return xr.Dataset(result_dict)


def mask_by_shape(
    xarray_data: xr.DataArray,
    gdf: gpd.GeoDataFrame,
    buffer: float = 0
) -> xr.DataArray:
    """
    Mask xarray data by vector boundaries
    
    Parameters
    ----------
    xarray_data : xr.DataArray
        Data to mask
    gdf : gpd.GeoDataFrame
        Boundaries for masking
    buffer : float
        Buffer distance to apply to boundaries
        
    Returns
    -------
    xr.DataArray
        Masked data
    """
    # Ensure CRS match
    gdf = gdf.to_crs(xarray_data.rio.crs)
    
    # Apply buffer if specified
    if buffer != 0:
        gdf = gdf.copy()
        gdf.geometry = gdf.geometry.buffer(buffer)
    
    # Clip data
    masked = xarray_data.rio.clip(gdf.geometry, gdf.crs, drop=True)
    
    return masked


def aggregate_to_polygon(
    xarray_data: xr.DataArray,
    gdf: gpd.GeoDataFrame,
    aggregation: str = 'mean',
    weights: Optional[xr.DataArray] = None
) -> gpd.GeoDataFrame:
    """
    Aggregate gridded data to polygon boundaries
    
    Parameters
    ----------
    xarray_data : xr.DataArray
        Gridded data
    gdf : gpd.GeoDataFrame
        Polygon boundaries
    aggregation : str
        Aggregation method ('mean', 'sum', 'max', 'min', 'median')
    weights : xr.DataArray, optional
        Weights for weighted aggregation
        
    Returns
    -------
    gpd.GeoDataFrame
        Polygons with aggregated values
    """
    gdf = gdf.to_crs(xarray_data.rio.crs)
    
    result_values = []
    
    for idx, row in gdf.iterrows():
        # Clip to polygon
        clipped = xarray_data.rio.clip([mapping(row.geometry)], gdf.crs)
        
        # Aggregate
        if weights is not None:
            clipped_weights = weights.rio.clip([mapping(row.geometry)], gdf.crs)
            value = (clipped * clipped_weights).sum() / clipped_weights.sum()
        else:
            if aggregation == 'mean':
                value = clipped.mean().values
            elif aggregation == 'sum':
                value = clipped.sum().values
            elif aggregation == 'max':
                value = clipped.max().values
            elif aggregation == 'min':
                value = clipped.min().values
            elif aggregation == 'median':
                value = clipped.median().values
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        result_values.append(float(value))
    
    gdf_result = gdf.copy()
    gdf_result['aggregated_value'] = result_values
    
    return gdf_result

from scipy.spatial import cKDTree

def interp_nearest_neighbors(gdf):
  import geopandas as gpd
  from shapely.ops import unary_union

  columns = gdf.columns
  for column in columns:
    missing_mask = gdf[column].isna()

    # For each missing region, interpolate from neighbors
    for idx in gdf[missing_mask].index:
        # Get the geometry of the missing region
        missing_geom = gdf.loc[idx, 'geometry']

        # Find neighbors (regions that touch or are within distance)
        neighbors = gdf[~missing_mask][gdf[~missing_mask].geometry.touches(missing_geom)]

        # If no touching neighbors, use nearest neighbors
        if len(neighbors) == 0:
            neighbors = gdf[~missing_mask].copy()
            neighbors['dist'] = neighbors.geometry.centroid.distance(missing_geom.centroid)
            neighbors = neighbors.nsmallest(5, 'dist')  # 5 nearest

        # Interpolate (e.g., mean of neighbors)
        gdf.loc[idx, column] = neighbors[column].mean()
  return gdf

def merged_xarr_county(xarr, interpolate = False):
  import geopandas as gpd
  import xarray as xr

  gdf_counties = data_io.load_counties()
  df = xarr.to_series().reset_index()
  df['county'] = df['county'].astype(int)
  gdf_counties['COUNTY_ID'] = gdf_counties.index
  gdf_merged = gdf_counties.merge(df, left_on='COUNTY_ID', right_on='county', how = 'left')
  if interpolate:
    gdf_merged = interp_nearest_neighbors(gdf_merged)

  return gdf_merged

def calculate_county_stats_xarr(xarray_data):
    """
    Most efficient version using xarray groupby directly
    """
    import geopandas as gpd
    import pandas as pd
    import xarray as xr

    all_results = []

    for rp in xarray_data.return_periods.values:
        print(f"\nProcessing {rp}-year return period...")

        # Select this return period
        rp_data = xarray_data.sel(return_periods=rp)


        dims = [d for d in rp_data.dims if d != 'county']
        # Calculate stats
        mean_vals = rp_data.mean(dim=dims)
        quantile_vals = rp_data.quantile(
            [0.1, 0.25, 0.5, 0.75, 0.9],
            dim=dims
        )


        mean_vals = mean_vals.compute()
        quantile_vals = quantile_vals.compute()

        # Extract individual quantiles
        p10 = quantile_vals.sel(quantile=0.1)
        p25 = quantile_vals.sel(quantile=0.25)
        p50 = quantile_vals.sel(quantile=0.5)
        p75 = quantile_vals.sel(quantile=0.75)
        p90 = quantile_vals.sel(quantile=0.9)

        # Convert to dataframe
        df = pd.DataFrame({
            'county': mean_vals.county.values,
            f'{rp}-yr mean': mean_vals.values,
            f'{rp}-yr 10th ptile': p10.values,
            f'{rp}-yr 25th ptile': p25.values,
            f'{rp}-yr 50th ptile': p50.values,
            f'{rp}-yr 75th ptile': p75.values,
            f'{rp}-yr 90th ptile': p90.values,
        })

        all_results.append(df.set_index('county'))

    # Combine all return periods
    result_df = pd.concat(all_results, axis=1).reset_index()
    result_df = result_df.rename(columns={'index': 'county_mask'})
    result_df['county'] = result_df['county'].astype('int')

    # Merge with geodataframe

    gdf_counties = data_io.load_counties()
    gdf_merged = gdf_counties.reset_index().rename(columns = {'index':'COUNTY_ID'}).merge(result_df, left_on='COUNTY_ID', right_on='county', how = 'left')

    gdf_merged = interp_nearest_neighbors(gdf_merged)

    return gdf_merged
