"""
Configuration file for MARISA IDF Analysis
Contains model lists, file paths, and constants
"""

# LOCA2 Climate Models
MODELS_LOCA2 = [
    "ACCESS-CM2", "ACCESS-ESM1-5", "BCC-CSM2-MR", "CESM2", "CESM2-WACCM",
    "CMCC-CM2-SR5", "CMCC-ESM2", "CNRM-CM6-1", "CNRM-ESM2-1", "CanESM5",
    "EC-Earth3", "EC-Earth3-Veg", "FGOALS-g3", "GFDL-ESM4", "GISS-E2-1-G",
    "HadGEM3-GC31-LL", "INM-CM4-8", "INM-CM5-0", "IPSL-CM6A-LR", "KACE-1-0-G",
    "KIOST-ESM", "MIROC-ES2L", "MIROC6", "MPI-ESM1-2-HR", "MPI-ESM1-2-LR",
    "MRI-ESM2-0", "NESM3", "NorESM2-LM", "NorESM2-MM", "TaiESM1", "UKESM1-0-LL"
]

# Original LOCA Models
MODELS_LOCA = [
    "ACCESS1-0", "ACCESS1-3", "CCSM4", "CESM1-BGC", "CESM1-CAM5", "CMCC-CM",
    "CMCC-CMS", "CNRM-CM5", "CSIRO-Mk3-6-0", "CanESM2", "EC-EARTH", "FGOALS-g2",
    "GFDL-CM3", "GFDL-ESM2G", "GFDL-ESM2M", "GISS-E2-H", "GISS-E2-R", "HadGEM2-AO",
    "HadGEM2-CC", "HadGEM2-ES", "IPSL-CM5A-LR", "IPSL-CM5A-MR", "MIROC-ESM",
    "MIROC-ESM-CHEM", "MIROC5", "MPI-ESM-LR", "MPI-ESM-MR", "MRI-CGCM3",
    "NorESM1-M", "bcc-csm1-1", "bcc-csm1-1-m"
]

# Scenarios
SCENARIOS = {
    'LOCA': ['rcp45', 'rcp85'],
    'LOCA2': ['ssp245', 'ssp370', 'ssp585']
}

# Unit conversions
MM_TO_INCHES = 0.0393701
SECONDS_PER_DAY = 86400

# Return periods for IDF analysis
RETURN_PERIODS = [2, 5, 10, 25, 50, 100]

# Durations (hours)
DURATIONS = [1, 2, 3, 6, 12, 24]

# Percentiles for analysis
PERCENTILES = [10, 25, 50, 75, 90]

# Default figure settings
FIGURE_DPI = 600
FIGURE_FORMAT = 'png'

# Color palettes
CMAP_DIVERGING = 'RdYlBu_r'
CMAP_SEQUENTIAL = 'viridis'
