"""
Time series processing for MARISA IDF Analysis
Functions for extracting PDS, AMS, and annual statistics from daily precipitation data
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal
from typing import Optional, Union, Tuple, List
from modules import config, data_io


def process_LOCA_timeseries(DIRECTORY, scenario, subset_time, ver = 'LOCA', model_set = None, out_string = None):
    if model_set is None:
        if ver == 'LOCA':
            model_set = config.MODELS_LOCA
            variable = 'precip'

        elif ver == 'LOCA2':
            model_set = config.MODELS_LOCA2
            variable = 'pr'
            pat = None
            
    
    for model in model_set:
        print(model)
        print(subset_time)
        try:

            ds = data_io.load_model(
                directory=DIRECTORY,
                model=model,
                scenario = scenario,
                subset_time = subset_time,
                variable = variable,
                pattern = f"{DIRECTORY}/LOCA_{model}_{scenario}.nc"
            )
        except:
            print("model data not found")
            continue

        result = process_precipitation_timeseries(
            ds[variable],
            min_separation_days=7,       
            compute_pds=True            
        ).compute()


        out_string = f'{DIRECTORY}/{model}.{scenario}.{str(subset_time[0])}-{str(subset_time[1])}_processed.zarr'

        print("\nProcessing complete!")
        result.to_zarr(out_string, compute = True, zarr_format=2, consolidated=False, mode = 'w')
        print(f"{out_string} saved!")

    return
  
def extract_annual_maxima(
    precip_data: xr.DataArray,
    time_dim: str = 'time'
) -> xr.DataArray:
    """
    Extract Annual Maximum Series (AMS) from daily precipitation data
    
    Parameters
    ----------
    precip_data : xr.DataArray
        Daily precipitation data with time dimension
    time_dim : str
        Name of time dimension (default: 'time')
        
    Returns
    -------
    xr.DataArray
        Annual maximum values with 'year' dimension

    """
    # Group by year and take maximum
    ams = precip_data.groupby(f'{time_dim}.year').max(dim=time_dim)
    
    # Rename dimension to 'year' if it's not already
    if 'year' not in ams.dims:
        ams = ams.rename({f'{time_dim}.year': 'year'})
    
    return ams


def extract_peaks_over_threshold(
    precip_data: xr.DataArray,
    n_peaks: int,
    min_separation_days: int = 7,
    time_dim: str = 'time',
    spatial_dims: [str] = ['x', 'y']
) -> xr.Dataset:
    """Extract n highest peaks per grid cell using xr.apply_ufunc"""
    
    def find_n_highest_peaks(data_1d, n_peaks, min_sep):
        """Find n highest peaks in 1D time series"""
        if len(data_1d) == 0 or np.all(np.isnan(data_1d)):
            return np.full(n_peaks, np.nan)
        
        # Find all local maxima
        peaks, _ = signal.find_peaks(data_1d, distance=min_sep)
        
        if len(peaks) == 0:
            # No peaks found, take n highest values
            sorted_indices = np.argsort(data_1d)[::-1][:n_peaks]
            return data_1d[sorted_indices]
        
        # Sort peaks by value and take top n
        peak_values = data_1d[peaks]
        sorted_peak_indices = np.argsort(peak_values)[::-1]
        
        if len(sorted_peak_indices) >= n_peaks:
            top_n_indices = sorted_peak_indices[:n_peaks]
        else:
            top_n_indices = sorted_peak_indices
        
        selected_peak_values = data_1d[peaks[top_n_indices]]
        
        # Pad with NaN if needed
        if len(selected_peak_values) < n_peaks:
            pad_length = n_peaks - len(selected_peak_values)
            selected_peak_values = np.concatenate([
                selected_peak_values,
                np.full(pad_length, np.nan)
            ])
        
        return selected_peak_values
    
    print(f"Extracting {n_peaks} highest peaks per grid cell...")
    
    # Apply function across all grid cells, preserving ensemble if present
    peak_values_da = xr.apply_ufunc(
        find_n_highest_peaks,
        precip_data.chunk(dict(time=-1)),
        n_peaks,
        min_separation_days,
        input_core_dims=[[time_dim], [], []],
        output_core_dims=[['peak']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        dask_gufunc_kwargs={'output_sizes': {'peak': n_peaks}}
    )
    
    # Add coordinates for peak dimension
    peak_values_da = peak_values_da.assign_coords(peak=range(n_peaks))
    
    # Calculate minimum peak value
    min_peak_value = peak_values_da.min(dim='peak')
    
    # Create result dataset
    result = xr.Dataset({
        'peak_values': peak_values_da,
        'min_peak_value': min_peak_value
    })
    
    result.attrs['n_peaks'] = n_peaks
    result.attrs['min_separation_days'] = min_separation_days
    result.attrs['description'] = f'PDS - {n_peaks} highest peaks per grid cell'
    
    print("Processing complete!")
    return result



def extract_annual_totals(
    precip_data: xr.DataArray,
    time_dim: str = 'time'
) -> xr.DataArray:
    """
    Calculate annual precipitation totals
    
    Parameters
    ----------
    precip_data : xr.DataArray
        Daily precipitation data
    time_dim : str
        Name of time dimension
        
    Returns
    -------
    xr.DataArray
        Annual precipitation totals with 'year' dimension
        
    Examples
    --------
    >>> annual_totals = extract_annual_totals(daily_precip)
    """
    # Group by year and sum
    annual_totals = precip_data.groupby(f'{time_dim}.year').sum(dim=time_dim)
    
    # Rename dimension
    if 'year' not in annual_totals.dims:
        annual_totals = annual_totals.rename({f'{time_dim}.year': 'year'})
    
    return annual_totals


def calculate_threshold_percentile(
    precip_data: xr.DataArray,
    percentile: float = 95.0,
    min_value: float = 0.1,
    time_dim: str = 'time'
) -> xr.DataArray:
    """
    Calculate spatial threshold map based on percentile of precipitation data
    
    Parameters
    ----------
    precip_data : xr.DataArray
        Daily precipitation data
    percentile : float
        Percentile for threshold (default: 95)
    min_value : float
        Minimum threshold value (default: 0.1 inches)
    time_dim : str
        Name of time dimension
        
    Returns
    -------
    xr.DataArray
        Spatial map of threshold values
        
    Examples
    --------
    >>> threshold = calculate_threshold_percentile(daily_precip, percentile=95)
    """
    # Calculate percentile
    threshold = precip_data.quantile(percentile / 100.0, dim=time_dim)
    
    # Apply minimum threshold
    threshold = threshold.where(threshold >= min_value, min_value)
    
    threshold.attrs['percentile'] = percentile
    threshold.attrs['min_value'] = min_value
    threshold.attrs['description'] = f'{percentile}th percentile precipitation threshold'
    
    return threshold


def process_precipitation_timeseries(
    precip_data: xr.DataArray,
    n_pds_peaks: Optional[int] = None,
    min_separation_days: int = 7,
    time_dim: str = 'time',
    compute_pds: bool = True
) -> xr.Dataset:
    """
    Process daily precipitation into AMS, PDS, and annual totals
    Main function that combines all extractions into one dataset
    
    Parameters
    ----------
    precip_data : xr.DataArray
        Daily precipitation data
    n_pds_peaks : int, optional
        Number of highest peaks to extract for PDS per grid cell
        If None, uses number of years in dataset (default behavior)
    min_separation_days : int
        Minimum days between PDS peaks (default: 3)
    time_dim : str
        Name of time dimension
    compute_pds : bool
        Whether to compute PDS (can be slow for large datasets)
        
    Returns
    -------
    xr.Dataset
        Dataset containing:
        - 'ams': Annual Maximum Series
        - 'annual_total': Annual precipitation totals
        - 'pds_peak_values': Peak precipitation values (if compute_pds=True)
        - 'pds_min_peak': Minimum peak value at each grid cell (if compute_pds=True)
        
    Examples
    --------
    >>> # Process full dataset (n_peaks = n_years automatically)
    >>> result = process_precipitation_timeseries(daily_precip)
    >>> print(result)
    >>> 
    >>> # Access results
    >>> ams = result['ams']
    >>> totals = result['annual_total']
    >>> pds_peaks = result['pds_peak_values']  # Shape: (n_peaks, y, x)
    """
    print("Processing precipitation time series...")
    
    # Extract AMS
    print("  1/3 Extracting Annual Maximum Series (AMS)...")
    ams = extract_annual_maxima(precip_data, time_dim=time_dim)
    ams.attrs['description'] = 'Annual Maximum Series - maximum daily precipitation per year'
    
    # Extract annual totals
    print("  2/3 Calculating annual totals...")
    annual_totals = extract_annual_totals(precip_data, time_dim=time_dim)
    annual_totals.attrs['description'] = 'Annual precipitation totals'
    
    # Create dataset with AMS and totals
    result = xr.Dataset({
        'ams': ams,
        'annual_total': annual_totals
    })
    
    # Optionally compute PDS
    if compute_pds:
        print("  3/3 Extracting Partial Duration Series (PDS)...")
        
        # Determine number of peaks if not specified
        if n_pds_peaks is None:
            n_years = len(ams.year)
            n_pds_peaks = n_years
            print(f"      Using n_peaks = {n_pds_peaks} (number of years)")
        else:
            print(f"      Using n_peaks = {n_pds_peaks}")
        
        print(f"      Extracting {n_pds_peaks} highest peaks per grid cell...")
        pds = extract_peaks_over_threshold(
            precip_data,
            n_peaks=n_pds_peaks,
            min_separation_days=min_separation_days,
            time_dim=time_dim
        )
        
        result['pds_peak_values'] = pds['peak_values']
        result['pds_min_peak'] = pds['min_peak_value']
        result['pds_peak_values'].attrs['description'] = f'Top {n_pds_peaks} precipitation peaks at each grid cell'
        result['pds_min_peak'].attrs['description'] = 'Minimum peak value selected (effective threshold)'
    else:
        print("  3/3 Skipping PDS extraction (compute_pds=False)")
    
    # Add metadata
    result.attrs['time_range'] = f"{precip_data[time_dim].min().values} to {precip_data[time_dim].max().values}"
    result.attrs['n_years'] = len(ams.year)
    if compute_pds:
        result.attrs['n_pds_peaks'] = n_pds_peaks
    result.attrs['min_separation_days'] = min_separation_days
    
    print("Processing complete!")
    return result

#### Inverse Risk Processing ###
import numpy as np
from scipy.stats import genextreme
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import numpy as np
import xarray as xr
import zarr
import rasterio
import rioxarray
from scipy.stats import bernoulli, gamma
from rasterio import features
import matplotlib.pyplot as plt

import pandas as pd
import geopandas as gpd

import xarray as xr
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator

def fit_gev_atlas14(T, x, verbose = False):
    """
    Fit a GEV to (T, x) pairs where x is intensity at return period T.
    T: array-like of return periods (years). Must be > 1 typically.
    x: array-like of intensities (same length).
    weights: optional array-like of same length to weight squared errors.
    Returns a dict with fitted params and helper functions.
    """
    T = np.asarray(T, dtype=float)
    x = np.asarray(x, dtype=float)
    if T.shape != x.shape:
        raise ValueError("T and x must have same shape")
    # Non-exceedance probability
    p = 1.0-1.0/T
    # Avoid exactly 0 or 1 (numerical)
    eps = 1e-10
    p = np.clip(p, eps, 1 - eps)

    # objective: sum of weighted squared differences between model CDF(x) and empirical p
    def obj(params):
        c, loc, scale = params
        try:
            model_p = genextreme.cdf(x, c, loc=loc, scale=scale)
            model_pred = genextreme.ppf(p, c, loc=loc, scale=scale)
        except Exception:
            return 1e8
        #resid = np.log(model_p) - np.log(p)
        resid = model_pred - x

        return np.sum(resid**2)

    # initial guesses
    loc0 = np.median(x)
    scale0 = np.std(x, ddof=1) if np.std(x, ddof=1) > 0 else max(1.0, 0.01*abs(loc0))
    c0 = -0.2
    x0 = np.array([c0, loc0, scale0])

    bounds = [(-2.0, 2.0), (None, None), (0, None)]  # keep shape within reasonable range, scale>0
    res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds)

    if not res.success:
        if verbose:
            print("Optimization warning:", res.message)

    c_hat, loc_hat, scale_hat = res.x

    return c_hat, loc_hat, scale_hat

def process_inversion(zarr_adjFact, a14, design_adjFactors, step_up=False):

  ### HELPER FUNCTIONS ###
    def invert_adjFact(rp, adjFactor, new_intensity):

        inv_rp = cdf_interp((1/adjFactor)*new_intensity)

        return inv_rp

    def interp_atlas14(a14_local):
        from scipy.interpolate import interp1d

        a14_const = xr.DataArray(
                a14_local,
                dims="return_periods",
                coords={"return_periods": [2, 5, 10, 25, 50, 100, 200, 500]}
                )
        #Fit GEV
        T = np.array(a14_const.return_periods.values.astype(int))
        x = np.array(a14_local)  # intensities

        # Create interpolation function
        cdf_interp = interp1d(x, 1-1/T,
                                kind='cubic',  # or 'cubic' for smoother
                                bounds_error=False,
                                fill_value=(0, 1))
        return x, cdf_interp

    ### Interpolate Atlas 14 Data ###
    a14_intensity, cdf_interp = interp_atlas14(a14)

    ### Adjusted Intensity ###
    zarr_intensity = zarr_adjFact*a14_intensity[:zarr_adjFact.return_periods.shape[0]]
    zarr_intensity = xr.Dataset({"intensity": zarr_intensity})
    zarr_intensity = zarr_intensity.chunk(dict(return_periods=-1))

    zarr_adjFact = xr.Dataset({"adjFact": zarr_adjFact})
    zarr_adjFact = zarr_adjFact.chunk(dict(return_periods=-1))

    ### Design Intensity ###
    new_intensity = design_adjFactors*a14_intensity[:design_adjFactors.shape[0]]
    new_intensity = np.sort(new_intensity)
    if step_up:
        new_intensity = a14_intensity[1:]
        new_intensity = new_intensity[:6]

    rp_vals = zarr_adjFact["return_periods"].values
    inverted = xr.apply_ufunc(
        invert_adjFact,
        zarr_adjFact["return_periods"],
        zarr_adjFact["adjFact"],
        new_intensity,
        input_core_dims=[["return_periods"], ["return_periods"], ["new_intensity"]],
        output_core_dims=[["new_intensity"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float]
    )

    inverted_ds = inverted.to_dataset(name='exceedance_probability')
    inverted_ds = inverted_ds.assign(new_intensity=new_intensity)

    inverted = inverted_ds['exceedance_probability']
    frequencies = inverted.new_intensity.values
    #Annual exceedance probability
    data = {f: 1-(((inverted.sel(new_intensity = f, method = 'nearest').values.flatten()))) for f in frequencies}

    ### Clean Data ###
    depths = []
    values = []
    for k, v in data.items():
        v = np.asarray(v)
        v_clean = v[np.isfinite(v)]        # drop NaN, inf, -inf
        if len(v_clean) > 0:               # only keep if not empty
            depths.append(k)
            values.append(v_clean)

    return values