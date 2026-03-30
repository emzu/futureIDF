"""
Statistical functions for MARISA IDF Analysis
IDF curve inversion, GEV fitting, and statistical tests
"""

import numpy as np
import xarray as xr
from scipy import stats
from scipy.stats import genextreme, gamma, ks_2samp, anderson_ksamp, median_test
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.optimize import minimize
from typing import Optional, Literal


def fit_gev(data: np.ndarray, method: str = 'mle') -> tuple:
    """
    Fit Generalized Extreme Value (GEV) distribution to data
    
    Parameters
    ----------
    data : np.ndarray
        Data to fit
    method : str
        Fitting method ('mle' for maximum likelihood)
        
    Returns
    -------
    tuple
        GEV parameters (shape, loc, scale)
    """
    return genextreme.fit(data, method=method)


def gev_return_period(params: tuple, return_period: float) -> float:
    """
    Calculate intensity for a given return period using GEV distribution
    
    Parameters
    ----------
    params : tuple
        GEV parameters (shape, loc, scale)
    return_period : float
        Return period in years
        
    Returns
    -------
    float
        Intensity value
    """
    exceedance_prob = 1.0 / return_period
    return genextreme.ppf(1 - exceedance_prob, *params)


def invert_curve(
    rp: np.ndarray,
    intensities: np.ndarray,
    new_intensity: float,
    interp_method: Literal['linear', 'pchip'] = 'linear'
) -> float:
    """
    Invert IDF curve to get return period for a given intensity
    
    Parameters
    ----------
    rp : np.ndarray
        Array of return periods
    intensities : np.ndarray
        Array of intensities corresponding to return periods
    new_intensity : float
        Intensity value to invert
    interp_method : str
        Interpolation method ('linear' or 'pchip')
        
    Returns
    -------
    float
        Return period (as exceedance probability)
    """
    # Ensure increasing intensity for interpolation
    if intensities[0] > intensities[-1]:
        rp = rp[::-1]
        intensities = intensities[::-1]
    
    # Convert return period to exceedance probability
    exceedance_prob = 1.0 / rp
    
    # Interpolate
    if interp_method == 'pchip':
        interpolator = PchipInterpolator(intensities, exceedance_prob)
    else:
        interpolator = interp1d(intensities, exceedance_prob, 
                               bounds_error=False, fill_value='extrapolate')
    
    return interpolator(new_intensity)


def invert_xarr(
    stack_zarr: xr.Dataset,
    new_intensity: np.ndarray,
    rp_dim: str = 'return_periods',
    intensity_var: str = 'intensity'
) -> xr.DataArray:
    """
    Invert IDF curves in a stacked xarray dataset
    
    Parameters
    ----------
    stack_zarr : xr.Dataset
        Dataset with return periods and intensities
    new_intensity : np.ndarray
        Array of intensities to invert
    rp_dim : str
        Name of return period dimension
    intensity_var : str
        Name of intensity variable
        
    Returns
    -------
    xr.DataArray
        Inverted return periods (exceedance probabilities)
    """
    rp_vals = stack_zarr[rp_dim].values
    
    inverted = xr.apply_ufunc(
        invert_curve,
        stack_zarr[rp_dim],
        stack_zarr[intensity_var],
        new_intensity,
        input_core_dims=[[rp_dim], [rp_dim], []],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]
    )
    
    return inverted


def fit_gev_atlas14(
    atlas14_intensities: np.ndarray,
    return_periods: np.ndarray
) -> tuple:
    """
    Fit GEV parameters to Atlas 14 IDF curve
    
    Parameters
    ----------
    atlas14_intensities : np.ndarray
        Atlas 14 intensity values
    return_periods : np.ndarray
        Corresponding return periods
        
    Returns
    -------
    tuple
        Fitted GEV parameters
    """
    def obj(params):
        """Objective function for optimization"""
        shape, loc, scale = params
        predicted = genextreme.ppf(1 - 1/return_periods, shape, loc, scale)
        return np.sum((predicted - atlas14_intensities)**2)
    
    # Initial guess
    initial_params = genextreme.fit(atlas14_intensities)
    
    # Optimize
    result = minimize(obj, initial_params, method='Nelder-Mead')
    
    return tuple(result.x)


def calculate_adjustment_factors(
    model_intensities: np.ndarray,
    atlas14_intensities: np.ndarray
) -> np.ndarray:
    """
    Calculate adjustment factors relative to Atlas 14
    
    Parameters
    ----------
    model_intensities : np.ndarray
        Model-derived intensities
    atlas14_intensities : np.ndarray
        Atlas 14 reference intensities
        
    Returns
    -------
    np.ndarray
        Adjustment factors
    """
    return model_intensities / atlas14_intensities


def ks_test_two_sample(data1: np.ndarray, data2: np.ndarray) -> tuple:
    """
    Perform two-sample Kolmogorov-Smirnov test
    
    Parameters
    ----------
    data1, data2 : np.ndarray
        Two samples to compare
        
    Returns
    -------
    tuple
        (statistic, p-value)
    """
    return ks_2samp(data1, data2)


def anderson_darling_k_sample(*samples) -> tuple:
    """
    Perform Anderson-Darling k-sample test
    
    Parameters
    ----------
    *samples : variable number of np.ndarray
        Multiple samples to compare
        
    Returns
    -------
    tuple
        Test results including statistic and critical values
    """
    return anderson_ksamp(samples)


def mood_median_test(*samples) -> tuple:
    """
    Perform Mood's median test
    
    Parameters
    ----------
    *samples : variable number of np.ndarray
        Multiple samples to compare
        
    Returns
    -------
    tuple
        (statistic, p-value, median, contingency_table)
    """
    return median_test(*samples)


def fit_gamma_distribution(data: np.ndarray) -> tuple:
    """
    Fit gamma distribution to data
    
    Parameters
    ----------
    data : np.ndarray
        Data to fit
        
    Returns
    -------
    tuple
        Gamma distribution parameters (shape, loc, scale)
    """
    return gamma.fit(data, floc=0)


def calculate_percentiles(
    data: np.ndarray,
    percentiles: list = [10, 25, 50, 75, 90]
) -> dict:
    """
    Calculate multiple percentiles of data
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    percentiles : list
        List of percentile values to calculate
        
    Returns
    -------
    dict
        Dictionary mapping percentile to value
    """
    return {p: np.percentile(data, p) for p in percentiles}


def df_variance(df: np.ndarray) -> float:
    """
    Calculate variance accounting for degrees of freedom
    
    Parameters
    ----------
    df : np.ndarray
        Input data
        
    Returns
    -------
    float
        Variance estimate
    """
    n = len(df)
    if n <= 1:
        return 0.0
    return np.var(df, ddof=1)
