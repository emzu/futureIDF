"""
Plotting functions for MARISA IDF Analysis
Maps, ridgeline plots, IDF curves, and statistical visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.colorbar import ColorbarBase
import seaborn as sns
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from typing import Optional, Union, Tuple, List
from modules import config
from .config import C_LOCA, C_LOCA2, PLT_STYLE


########## Future IDF Curve ##########

def future_idf_curve(LOCA, LOCA2, atlas14_counties, county_name, scenario = ['rcp45','ssp245'], time_period = '2050-2100', duration=None, ax=None, label=None, **kwargs):
    from scipy.stats import gaussian_kde

    frequencies = config.RETURN_PERIODS
    county_idx = atlas14_counties[atlas14_counties['county_name']==county_name].index.values[0]
    a14_county = atlas14_counties.iloc[county_idx][['2', '5', '10', '25', '50', '100']]

    ds_zarr_LOCA = LOCA.sel(scenario = scenario[0], time_period = time_period).sel(county = str(county_idx)).mean('centroid_cell')['adj_factor']
    ds_zarr_LOCA2 = LOCA2.sel(scenario = scenario[1], time_period = time_period).sel(county = str(county_idx)).mean('centroid_cell')['adj_factor']

    data_loca = {f: a14_county[str(f)].values*ds_zarr_LOCA.sel(return_periods=f).values.flatten() for f in frequencies}
    data_loca2 = {f: a14_county[str(f)].values*ds_zarr_LOCA2.sel(return_periods=f).values.flatten() for f in frequencies}

    # Compute percentiles
    p10a = [np.nanpercentile(data_loca[f], 10) for f in frequencies]
    p50a = [np.nanpercentile(data_loca[f], 50) for f in frequencies]
    p90a = [np.nanpercentile(data_loca[f], 90) for f in frequencies]

    p10b = [np.nanpercentile(data_loca2[f], 10) for f in frequencies]
    p50b = [np.nanpercentile(data_loca2[f], 50) for f in frequencies]
    p90b = [np.nanpercentile(data_loca2[f], 90) for f in frequencies]

    a14 = [a14_county[str(f)].values for f in frequencies]

    plt.rcParams.update(config.PLT_STYLE)

    # Main axis for frequency vs value
    fig, ax = plt.subplots(figsize=(6.5,3))
    hist_width = .8  # fixed width for all histograms
    bin_width = 0.5
    bins = np.arange(0, 20 + bin_width, bin_width)

    #ax = axs[0]
    x_vals = np.array([2, 5, 10, 25, 50, 100])
    positions = np.arange(len(x_vals))

    for f in frequencies:
        x_pos = positions[frequencies.tolist().index(f)]
        vals = data_loca2[f]
        vals = vals[~np.isnan(vals)]

        # KDE for violin shape

        kde = gaussian_kde(vals, bw_method='scott')
        y_range = np.linspace(vals.min(), vals.max(), 200)
        kde_vals = kde(y_range)

        # Normalize and scale by width
        kde_scaled = kde_vals / kde_vals.max() * hist_width

        ax.fill_betweenx(y_range, x_pos, x_pos + kde_scaled / 2,
                        facecolor=(config.C_LOCA2, 0.4), edgecolor=(0, 0, 0, 1.0), linewidth=1)

        p10, p50, p90 = np.percentile(vals, [10, 50, 90])

        for p, ls in zip([p10, p50, p90], ['--', '-', '--']):
            hw = np.interp(p, y_range, kde_scaled) / 2
            ax.hlines(p, x_pos, x_pos + hw, color=config.C_LOCA2, linewidth=1.5, linestyle=ls)

    for f in frequencies:
        x_pos = positions[frequencies.tolist().index(f)]
        vals = data_loca[f]
        vals = vals[~np.isnan(vals)]

        # KDE for violin shape
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(vals, bw_method='scott')
        y_range = np.linspace(vals.min(), vals.max(), 200)
        kde_vals = kde(y_range)

        # Normalize and scale by width
        kde_scaled = kde_vals / kde_vals.max() * hist_width

        # Draw filled violin (right half only, mirroring your original style)
        ax.fill_betweenx(y_range, x_pos - kde_scaled / 2, x_pos,
                        facecolor=(C_LOCA, 0.4), edgecolor=(0, 0, 0, 1.0), linewidth=1)
        p10, p50, p90 = np.percentile(vals, [10, 50, 90])

        for p, ls in zip([p10, p50, p90], ['--', '-', '--']):
            hw = np.interp(p, y_range, kde_scaled) / 2
            ax.hlines(p, x_pos, x_pos - hw, color=C_LOCA, linewidth=1.5, linestyle=ls)

    # Plot percentile curves on frequency axis
    ax.plot(positions, p50a, 'o', color=C_LOCA, label="LOCA (CMIP5)")
    #ax.plot(frequencies, p10, 'o', color='orange', label="_nolegend_")
    #ax.plot(frequencies, p90, 'o', color='orange', label="_nolegend_")

    ax.plot(positions, p50b, 'o', color=C_LOCA2,  label="LOCA2 (CMIP6)")
    #ax.plot(frequencies, p10b, 'o--', color='purple',  label="_nolegend_")
    #ax.plot(frequencies, p90b, 'o--', color='purple',  label="_nolegend_")

    ax.plot(positions, a14, 'o-', color='black', label="Atlas 14")

    # Label axes
    ax.set_xlabel("Return Period (yrs)")
    ax.set_ylabel("24-hr Depth (in)")
    ax.legend(loc="upper left", frameon = False)
    ax.set_ylim([0,10])
    plt.xlim([-1,6])
    plt.xticks(positions, x_vals)

    return fig, ax

def quick_plot(
    gdf: gpd.GeoDataFrame,
    p10_col: str = 'p10',
    p50_col: str = 'p50',
    p90_col: str = 'p90',
    cmap: str = 'RdYlBu_r',
    show_nan: bool = True,
    nan_color: str = 'lightgray',
    figsize: Tuple[int, int] = (15, 5),
    title: Optional[str] = None
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a three-panel plot showing p10, p50, and p90 values
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with percentile columns
    p10_col, p50_col, p90_col : str
        Column names for percentiles
    cmap : str
        Colormap name
    show_nan : bool
        If True, plot NaN regions in gray
    nan_color : str
        Color for NaN regions
    figsize : tuple
        Figure size
    title : str, optional
        Overall title
        
    Returns
    -------
    tuple
        (figure, axes array)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Get global vmin/vmax across all percentiles
    all_values = pd.concat([gdf[p10_col], gdf[p50_col], gdf[p90_col]]).dropna()
    vmin, vmax = all_values.min(), all_values.max()
    
    # Plot each percentile
    for ax, col, label in zip(axes, [p10_col, p50_col, p90_col], 
                              ['10th Percentile', '50th Percentile', '90th Percentile']):
        
        # Handle NaN values
        if show_nan:
            gdf_nan = gdf[gdf[col].isna()]
            if not gdf_nan.empty:
                gdf_nan.plot(ax=ax, color=nan_color, edgecolor='black', linewidth=0.3)
        
        # Plot data
        gdf_valid = gdf[gdf[col].notna()]
        gdf_valid.plot(column=col, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                      legend=True, edgecolor='black', linewidth=0.3)
        
        ax.set_title(label)
        ax.axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16, y=1.02)
    
    plt.tight_layout()
    return fig, axes


def create_bivariate_palette(
    n_classes: int = 3,
    primary_color: str = 'Blues',
    secondary_color: str = 'Reds'
) -> np.ndarray:
    """
    Create a bivariate color palette
    
    Parameters
    ----------
    n_classes : int
        Number of classes per dimension
    primary_color : str
        Colormap for first variable
    secondary_color : str
        Colormap for second variable
        
    Returns
    -------
    np.ndarray
        2D array of colors
    """
    cmap1 = plt.cm.get_cmap(primary_color)
    cmap2 = plt.cm.get_cmap(secondary_color)
    
    palette = np.zeros((n_classes, n_classes, 4))
    
    for i in range(n_classes):
        for j in range(n_classes):
            color1 = np.array(cmap1(i / (n_classes - 1)))
            color2 = np.array(cmap2(j / (n_classes - 1)))
            # Blend colors
            palette[i, j] = (color1 + color2) / 2
    
    return palette


def bivariate_plot(
    gdf: gpd.GeoDataFrame,
    var1: str,
    var2: str,
    n_classes: int = 3,
    cmap1: str = 'Blues',
    cmap2: str = 'Reds',
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create bivariate choropleth map
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input geodataframe
    var1, var2 : str
        Column names for two variables
    n_classes : int
        Number of classes for each variable
    cmap1, cmap2 : str
        Colormaps for each variable
    figsize : tuple
        Figure size
    title : str, optional
        Plot title
        
    Returns
    -------
    tuple
        (figure, axis)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create palette
    palette = create_bivariate_palette(n_classes, cmap1, cmap2)
    
    # Classify variables
    gdf = gdf.copy()
    gdf['class1'] = pd.qcut(gdf[var1], n_classes, labels=False, duplicates='drop')
    gdf['class2'] = pd.qcut(gdf[var2], n_classes, labels=False, duplicates='drop')
    
    # Assign colors
    colors = []
    for idx, row in gdf.iterrows():
        if pd.notna(row['class1']) and pd.notna(row['class2']):
            colors.append(palette[int(row['class1']), int(row['class2'])])
        else:
            colors.append([0.9, 0.9, 0.9, 1.0])  # Light gray for NaN
    
    gdf.plot(ax=ax, color=colors, edgecolor='black', linewidth=0.3)
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    return fig, ax


def dual_map_plot(
    gdf: gpd.GeoDataFrame,
    var1: str,
    var2: str,
    cmap: str = 'RdYlBu_r',
    figsize: Tuple[int, int] = (15, 6),
    titles: Optional[List[str]] = None
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create side-by-side maps for two variables
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input geodataframe
    var1, var2 : str
        Variables to plot
    cmap : str
        Colormap
    figsize : tuple
        Figure size
    titles : list of str, optional
        Titles for each subplot
        
    Returns
    -------
    tuple
        (figure, axes array)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    variables = [var1, var2]
    if titles is None:
        titles = [var1, var2]
    
    for ax, var, title in zip(axes, variables, titles):
        gdf.plot(column=var, ax=ax, cmap=cmap, legend=True,
                edgecolor='black', linewidth=0.3)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    return fig, axes


def ridgeline_plot(
    data_dict: dict,
    figsize: Tuple[int, int] = (10, 8),
    overlap: float = 0.5,
    alpha: float = 0.7,
    color_palette: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create ridgeline (joyplot) visualization
    
    Parameters
    ----------
    data_dict : dict
        Dictionary with labels as keys and data arrays as values
    figsize : tuple
        Figure size
    overlap : float
        Overlap between distributions (0-1)
    alpha : float
        Transparency of fills
    color_palette : str, optional
        Seaborn color palette name
        
    Returns
    -------
    tuple
        (figure, axis)
    """
    n_distributions = len(data_dict)
    
    if color_palette is None:
        colors = sns.color_palette("husl", n_distributions)
    else:
        colors = sns.color_palette(color_palette, n_distributions)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, (label, data) in enumerate(data_dict.items()):
        # Calculate KDE
        kde = sns.kdeplot(data, ax=ax, color=colors[i], alpha=0)
        
        # Get the line
        line = kde.get_lines()[-1]
        x, y = line.get_data()
        
        # Offset y values
        y_offset = i * overlap
        y_scaled = y + y_offset
        
        # Plot
        ax.fill_between(x, y_offset, y_scaled, alpha=alpha, color=colors[i], label=label)
        ax.plot(x, y_scaled, color=colors[i], linewidth=1.5)
    
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    
    return fig, ax


def plot_idf_curve(
    return_periods: np.ndarray,
    intensities: np.ndarray,
    duration: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot IDF curve
    
    Parameters
    ----------
    return_periods : np.ndarray
        Return periods
    intensities : np.ndarray
        Intensity values
    duration : float, optional
        Duration (for labeling)
    ax : plt.Axes, optional
        Axis to plot on
    label : str, optional
        Label for the curve
    **kwargs
        Additional plotting arguments
        
    Returns
    -------
    plt.Axes
        Plotting axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    if label is None and duration is not None:
        label = f'{duration}-hr'
    
    ax.plot(return_periods, intensities, marker='o', label=label, **kwargs)
    ax.set_xlabel('Return Period (years)')
    ax.set_ylabel('Intensity (in/hr)')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    if label is not None:
        ax.legend()
    
    return ax


def plot_multiple_idf_curves(
    curves_dict: dict,
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot multiple IDF curves on the same axis
    
    Parameters
    ----------
    curves_dict : dict
        Dictionary with curve labels as keys and (return_periods, intensities) tuples as values
    figsize : tuple
        Figure size
    title : str, optional
        Plot title
        
    Returns
    -------
    tuple
        (figure, axis)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for label, (rp, intensities) in curves_dict.items():
        ax.plot(rp, intensities, marker='o', label=label, linewidth=2)
    
    ax.set_xlabel('Return Period (years)', fontsize=12)
    ax.set_ylabel('Intensity (in/hr)', fontsize=12)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if title:
        ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    return fig, ax


def create_map_with_basemap(
    gdf: gpd.GeoDataFrame,
    column: Optional[str] = None,
    cmap: str = 'viridis',
    figsize: Tuple[int, int] = (12, 8),
    add_states: bool = True,
    add_coastline: bool = True,
    title: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create map with cartopy basemap features
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Data to plot
    column : str, optional
        Column to visualize
    cmap : str
        Colormap
    figsize : tuple
        Figure size
    add_states : bool
        Add state boundaries
    add_coastline : bool
        Add coastlines
    title : str, optional
        Plot title
        
    Returns
    -------
    tuple
        (figure, axis)
    """
    fig, ax = plt.subplots(figsize=figsize, 
                           subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Add basemap features
    if add_coastline:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    if add_states:
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
    
    # Plot data
    if column is not None:
        gdf.plot(column=column, ax=ax, cmap=cmap, legend=True,
                transform=ccrs.PlateCarree(), edgecolor='black', linewidth=0.3)
    else:
        gdf.plot(ax=ax, transform=ccrs.PlateCarree(), 
                edgecolor='black', linewidth=0.5)
    
    ax.set_extent([gdf.total_bounds[0], gdf.total_bounds[2],
                   gdf.total_bounds[1], gdf.total_bounds[3]])
    
    if title:
        ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    return fig, ax


def label(x: float, unit: str = '') -> str:
    """
    Format label for plots
    
    Parameters
    ----------
    x : float
        Value to format
    unit : str
        Unit string
        
    Returns
    -------
    str
        Formatted label
    """
    if unit:
        return f"{x:.2f} {unit}"
    return f"{x:.2f}"

# Side-by-side maps: one for values, one for confidence
def dual_map_plot(gdf, p10_col='p10', p50_col='p50', p90_col='p90',
                  value_cmap='viridis', ci_cmap='Greys',
                  show_nan=True, nan_color='lightgray',
                  vmin=None, vmax=None, figsize=(16, 6)):
    """
    Create two side-by-side maps: median values (left) and CI width (right)

    Parameters:
    -----------
    value_cmap : str
        Colormap for the median values (default: PuOr - purple to orange)
    ci_cmap : str
        Colormap for CI width (default: Greys_r - dark=narrow, light=wide)
    vmin, vmax : float, optional
        Manually set value range. If None, uses 2nd and 98th percentiles
    figsize : tuple
        Figure size (width, height)
    """

    # Create a copy
    gdf = gdf.copy()

    # Identify valid rows
    valid_mask = (
        gdf[p10_col].notna() &
        gdf[p50_col].notna() &
        gdf[p90_col].notna()
    )

    gdf_valid = gdf[valid_mask].copy()
    gdf_nan = gdf[~valid_mask].copy() if show_nan else None

    if len(gdf_valid) == 0:
        print("No valid data to plot!")
        return None, None, None

    # Calculate CI width
    ci_width = gdf_valid[p90_col] - gdf_valid[p10_col]
    gdf_valid['ci_width'] = ci_width
    # Calculate value range
    if vmin is None:
        vmin = np.percentile(gdf_valid[p50_col], 2)
    if vmax is None:
        vmax = np.percentile(gdf_valid[p50_col], 98)

    # Create figure with two maps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- LEFT MAP: Median Values ---
    median_norm = (gdf_valid[p50_col] - vmin) / (vmax - vmin)
    median_norm = np.clip(median_norm, 0, 1)

    value_colormap = plt.cm.get_cmap(value_cmap)

    gdf_valid.plot(column=p50_col, ax=ax1, cmap=value_cmap,
                    edgecolor='white', linewidth=0.3,
                   vmin = vmin, vmax = vmax)

    # Plot NaN regions
    if show_nan and gdf_nan is not None and len(gdf_nan) > 0:
        gdf_nan.plot(ax=ax1, color=nan_color, alpha=0.5,
                     edgecolor='white', linewidth=0.3)

    ax1.set_axis_off()
    ax1.set_title('Adjustment Factor (Median)', fontsize=14, fontweight='bold', pad=10)

    # Add colorbar for values
    sm1 = plt.cm.ScalarMappable(cmap=value_cmap,
                                 norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar1 = plt.colorbar(sm1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Adjustment Factor', fontsize=11)

    # --- RIGHT MAP: CI Width ---
    ci_min, ci_max = ci_width.min(), ci_width.max()
    ci_norm = (ci_width - ci_min) / (ci_max - ci_min) if ci_max > ci_min else np.zeros(len(ci_width))

    ci_colormap = plt.cm.get_cmap(ci_cmap)

    gdf_valid.plot(column='ci_width', ax=ax2, cmap=ci_cmap,
                               edgecolor='white', linewidth=0.3, 
                               vmin = 0.3, vmax = 1)

    # Plot NaN regions
    if show_nan and gdf_nan is not None and len(gdf_nan) > 0:
        gdf_nan.plot(ax=ax2, color=nan_color, alpha=0.5,
                     edgecolor='white', linewidth=0.3)

    ax2.set_axis_off()
    ax2.set_title('Confidence Interval Width', fontsize=14, fontweight='bold', pad=10)

    # Add colorbar for CI
    sm2 = plt.cm.ScalarMappable(cmap=ci_cmap,
                                 norm=plt.Normalize(vmin=0.3, vmax=1))
    cbar2 = plt.colorbar(sm2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('CI Width (90th - 10th percentile)', fontsize=11)


    plt.tight_layout()

    return fig, ax1, ax2

def plot_extreme_value_series(result):
    # Annual Maximum Series (AMS)
    ams = result['ams']
    print("Annual Maximum Series (AMS):")
    print(f"  Shape: {ams.shape}")
    print(f"  Years: {len(ams.year)}")
    print(f"  Range: {ams.min().values:.2f} to {ams.max().values:.2f} inches/day")

    # Annual Totals
    annual_totals = result['annual_total']
    print("\nAnnual Totals:")
    print(f"  Shape: {annual_totals.shape}")
    print(f"  Range: {annual_totals.min().values:.2f} to {annual_totals.max().values:.2f} inches/year")

    # PDS (if computed)
    if 'pds_n_peaks' in result:
        n_peaks = result['pds_n_peaks']
        threshold = result['pds_threshold']
        print("\nPartial Duration Series (PDS):")
        print(f"  Mean peaks per grid cell: {n_peaks.mean().values:.1f}")
        print(f"  Threshold range: {threshold.min().values:.2f} to {threshold.max().values:.2f} inches/day")

    # Plot mean AMS across all years
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Mean AMS
    ams.mean(dim='year').plot(ax=axes[0], cmap='YlGnBu')
    axes[0].set_title('Mean Annual Maximum\nDaily Precipitation (in/day)')

    # Mean Annual Total
    annual_totals.mean(dim='year').plot(ax=axes[1], cmap='YlGnBu')
    axes[1].set_title('Mean Annual Total\nPrecipitation (in/year)')

    # PDS threshold (if available)
    if 'pds_min_peak' in result:
        result['pds_min_peak'].plot(ax=axes[2], cmap='YlOrRd')
        axes[2].set_title('PDS Minimum Peak Value\n(in/day)')
    else:
        # Plot number of peaks instead
        axes[2].text(0.5, 0.5, 'PDS not computed', ha='center', va='center')
        axes[2].axis('off')
    return fig, axes