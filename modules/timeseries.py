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
        Minimum days between PDS peaks (default: 7)
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
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import xarray as xr
import zarr
import rasterio
import rioxarray
from scipy.stats import bernoulli, gamma
from rasterio import features
import pandas as pd
import geopandas as gpd
 
 
# ---------------------------------------------------------------------------
# Atlas-14 Return Periods (standard)
# ---------------------------------------------------------------------------
ATLAS14_RETURN_PERIODS = np.array([2, 5, 10, 25, 50, 100, 200, 500])
 
 
# ---------------------------------------------------------------------------
# GEV Fitting
# ---------------------------------------------------------------------------
 
def fit_gev_atlas14(T, x, verbose=False):
    """
    Fit a GEV to (T, x) pairs where x is intensity at return period T.
 
    Parameters
    ----------
    T : array-like
        Return periods (years). Must be > 1.
    x : array-like
        Intensities (same length as T).
    verbose : bool
        If True, print optimization warnings.
 
    Returns
    -------
    c_hat, loc_hat, scale_hat : float
        Fitted GEV shape, location, and scale parameters.
    """
    T = np.asarray(T, dtype=float)
    x = np.asarray(x, dtype=float)
    if T.shape != x.shape:
        raise ValueError("T and x must have the same shape")
 
    p = np.clip(1.0 - 1.0 / T, 1e-10, 1 - 1e-10)
 
    def obj(params):
        c, loc, scale = params
        try:
            model_pred = genextreme.ppf(p, c, loc=loc, scale=scale)
        except Exception:
            return 1e8
        return np.sum((model_pred - x) ** 2)
 
    loc0 = np.median(x)
    scale0 = np.std(x, ddof=1) if np.std(x, ddof=1) > 0 else max(1.0, 0.01 * abs(loc0))
    x0 = np.array([-0.2, loc0, scale0])
    bounds = [(-2.0, 2.0), (None, None), (0, None)]
 
    res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds)
    if not res.success and verbose:
        print("Optimization warning:", res.message)
 
    return tuple(res.x)
 
 
# ---------------------------------------------------------------------------
# Atlas-14 CDF Interpolator (shared helper)
# ---------------------------------------------------------------------------
 
def _build_atlas14_cdf_interp(a14):
    """
    Build a cubic interpolator mapping intensity → non-exceedance probability
    for a single Atlas-14 intensity vector.
 
    Parameters
    ----------
    a14 : array-like, shape (n_rp,)
        Atlas-14 intensities at ATLAS14_RETURN_PERIODS[:n_rp].
 
    Returns
    -------
    a14_intensity : np.ndarray
        Intensities corresponding to each standard return period.
    cdf_interp : callable
        Interpolator: intensity → non-exceedance probability ∈ [0, 1].
    """
    a14_intensity = np.asarray(a14, dtype=float)
    n = len(a14_intensity)
    T = ATLAS14_RETURN_PERIODS[:n]
    p = 1.0 - 1.0 / T
 
    cdf_interp = interp1d(
        a14_intensity, p,
        kind="cubic",
        bounds_error=False,
        fill_value=(0.0, 1.0),
    )
    return a14_intensity, cdf_interp
 
 
# ---------------------------------------------------------------------------
# Design Depth Builder
# ---------------------------------------------------------------------------
 
def build_design_depths(
    a14,
    mode="change_factor",
    change_factors=None,
    ci_upper=None,
    n_rp=None,
):
    """
    Construct a design-depth array for use in inversion.
 
    One design depth is produced per return period so that the resulting array
    can be passed directly to ``process_inversion``.
 
    Parameters
    ----------
    a14 : array-like, shape (n_rp,)
        Atlas-14 mean intensities at the standard return periods.
    mode : {"change_factor", "step_up", "upper_ci"}
        Strategy for deriving design depths:
 
        ``"change_factor"``
            Scale each Atlas-14 intensity by the corresponding entry in
            *change_factors*.  ``change_factors`` must be provided.
 
        ``"step_up"``
            Shift each return-period position one step to the right on the
            Atlas-14 curve (i.e., the design depth for the 2-yr event becomes
            the current 5-yr intensity, etc.).  The last return period is
            extrapolated by applying the ratio of the last two Atlas-14 values.
 
        ``"upper_ci"``
            Use the Atlas-14 upper confidence-interval values supplied in
            *ci_upper* as the design depths.  ``ci_upper`` must be provided and
            must have the same length as *a14*.
 
    change_factors : array-like, shape (n_rp,), optional
        Required when ``mode="change_factor"``.  Multiplicative adjustment
        factors applied element-wise to *a14*.
    ci_upper : array-like, shape (n_rp,), optional
        Required when ``mode="upper_ci"``.  Atlas-14 upper CI intensities.
    n_rp : int, optional
        Truncate output to the first *n_rp* return periods.  Defaults to
        ``len(a14)``.
 
    Returns
    -------
    design_depths : np.ndarray, shape (n_rp,)
        One design depth per return period, sorted ascending.
 
    Examples
    --------
    >>> a14 = [0.5, 0.8, 1.0, 1.3, 1.5, 1.8, 2.1, 2.5]
    >>> cf  = [1.0, 1.05, 1.1, 1.1, 1.15, 1.2, 1.2, 1.25]
    >>> build_design_depths(a14, mode="change_factor", change_factors=cf)
    """
    a14 = np.asarray(a14, dtype=float)
    n = n_rp if n_rp is not None else len(a14)
 
    if mode == "change_factor":
        if change_factors is None:
            raise ValueError("change_factors must be provided when mode='change_factor'")
        cf = np.asarray(change_factors, dtype=float)
        if cf.shape[0] < n:
            raise ValueError(
                f"change_factors has {cf.shape[0]} entries but n_rp={n} was requested"
            )
        design_depths = cf[:n] * a14[:n]
 
    elif mode == "step_up":
        # Shift one position to the right; extrapolate the last entry.
        stepped = np.empty(n, dtype=float)
        # For positions 0 … n-2, take the next Atlas-14 value
        available = min(n, len(a14) - 1)
        stepped[:available] = a14[1 : available + 1]
        # Extrapolate beyond the available Atlas-14 range using the last ratio
        if available < n:
            ratio = a14[-1] / a14[-2] if a14[-2] != 0 else 1.0
            for i in range(available, n):
                stepped[i] = stepped[i - 1] * ratio
        design_depths = stepped
 
    elif mode == "upper_ci":
        if ci_upper is None:
            raise ValueError("ci_upper must be provided when mode='upper_ci'")
        ci = np.asarray(ci_upper, dtype=float)
        if len(ci) < n:
            raise ValueError(
                f"ci_upper has {len(ci)} entries but n_rp={n} was requested"
            )
        design_depths = ci[:n]
 
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: change_factor, step_up, upper_ci")
 
    return np.sort(design_depths)
 
 
# ---------------------------------------------------------------------------
# Core Inversion
# ---------------------------------------------------------------------------
 
def process_inversion(zarr_adjFact, a14, design_depths):
    """
    Invert adjustment factors to compute annual exceedance probabilities for
    each design depth at each county grid cell.
 
    For a given climate-model adjustment factor *f* at return period *T*, the
    future intensity at *T* is ``f * I_atlas14(T)``.  Given a target design
    depth *d*, the inverted return period is found by asking: "at what
    non-exceedance probability does the *current* Atlas-14 CDF equal
    ``d / f``?"  This is the probability that the *future* distribution
    exceeds *d*.
 
    Parameters
    ----------
    zarr_adjFact : xr.DataArray
        Adjustment factors with a ``return_periods`` dimension, shape
        (n_rp, [spatial dims…]).  Values are dimensionless multipliers
        (e.g. 1.15 = 15 % increase).
    a14 : array-like, shape (n_rp,)
        Atlas-14 mean intensities at the standard return periods, ordered to
        match ``zarr_adjFact.return_periods``.
    design_depths : array-like, shape (n_design,)
        Target design depths for which to compute exceedance probabilities.
        Typically produced by :func:`build_design_depths`.
 
    Returns
    -------
    values : list of np.ndarray
        One array per design depth (in the same order as *design_depths*),
        each containing the annual exceedance probabilities across all spatial
        units in *zarr_adjFact*.
    """
    design_depths = np.asarray(design_depths, dtype=float)
    a14_intensity, cdf_interp = _build_atlas14_cdf_interp(a14)
 
    # Align Atlas-14 intensities to the return periods present in zarr_adjFact
    n_rp = zarr_adjFact.return_periods.shape[0]
    a14_aligned = a14_intensity[:n_rp]
 
    # Adjusted (future) intensities: shape (n_rp, [spatial dims…])
    zarr_intensity = zarr_adjFact * a14_aligned
 
    zarr_adjFact_ds = xr.Dataset({"adjFact": zarr_adjFact}).chunk(dict(return_periods=-1))
 
    def invert_adjFact(rp, adjFactor, new_intensity):
        """Scalar ufunc: given adjFactor vector and design depth, return exceedance prob."""
        inv_rp = cdf_interp((1.0 / adjFactor) * new_intensity)
        return inv_rp
 
    inverted = xr.apply_ufunc(
        invert_adjFact,
        zarr_adjFact_ds["return_periods"],
        zarr_adjFact_ds["adjFact"],
        design_depths,
        input_core_dims=[["return_periods"], ["return_periods"], ["new_intensity"]],
        output_core_dims=[["new_intensity"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
 
    inverted_ds = inverted.to_dataset(name="exceedance_probability")
    inverted_ds = inverted_ds.assign(new_intensity=design_depths)
    inverted = inverted_ds["exceedance_probability"]
 
    # Convert non-exceedance → annual exceedance probability, drop non-finite
    data = {
        f: 1.0 - inverted.sel(new_intensity=f, method="nearest").values.flatten()
        for f in design_depths
    }
 
    depths, values = [], []
    for depth, v in data.items():
        v = np.asarray(v)
        v_clean = v[np.isfinite(v)]
        if len(v_clean) > 0:
            depths.append(depth)
            values.append(v_clean)
 
    return pd.Series([v[0] for v in values], index=depths)
 
 
# ---------------------------------------------------------------------------
# Convenience wrapper (county-level entry point)
# ---------------------------------------------------------------------------
 
def compute_county_exceedance(
    zarr_adjFact,
    a14,
    mode="change_factor",
    change_factors=None,
    ci_upper=None,
    n_rp=None,
):
    """
    Parameters
    ----------
    zarr_adjFact : xr.DataArray
        Adjustment factors with a ``return_periods`` dimension.
    a14 : array-like
        Atlas-14 mean depths
    mode : str
        Passed to :func:`build_design_depths`.
    change_factors : array-like, optional
        Passed to :func:`build_design_depths` when ``mode="change_factor"``.
    ci_upper : array-like, optional
        Passed to :func:`build_design_depths` when ``mode="upper_ci"``.
    n_rp : int, optional
        Number of return periods to use.  Defaults to ``len(a14)``.
 
    Returns
    -------
    design_depths : np.ndarray
        The design depths used.
    values : list of np.ndarray
        Exceedance probabilities per design depth (see :func:`process_inversion`).
    """
    design_depths = build_design_depths(
        a14,
        mode=mode,
        change_factors=change_factors,
        ci_upper=ci_upper,
        n_rp=n_rp,
    )
    values = process_inversion(zarr_adjFact, a14, design_depths)
    return design_depths, values