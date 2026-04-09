# futureIDF
Code framework to rapidly generate future IDF curves from downscaled GCM precipitation

An open-source Python framework to rabidly generate future IDF curves from downscaled climate model outputs to support climate-resilient stormwater infrastructure design. Future IDF curves quantify projected changes in extreme precipitation through multiplicative change factors derived from climate projections applied to observed precipitation data through quantile delta mapping. This framework supports ensemble analysis of both LOCA-downscaled CMIP5 and LOCA2-downscaled CMIP6 projections, enabling direct comparison of climate model generations and derived change factors. It can be easily updated with additional downscaled data (STAR-ESDM, NA-Cordex) or future CMIP generations (CMIP7)
 
The framework was developed for the Mid-Atlantic region as part of updates to the [MARISA IDF Curve Tool](https://midatlantic-idf.rcc-acis.org/), but is designed to be adaptable to other regions with access to gridded downscaled precipitation data.
 
Key capabilities:
- Download and process daily precipitation from LOCA (CMIP5) and LOCA2 (CMIP6) archives
- Extract Annual Maximum Series (AMS) and fit Generalized Extreme Value (GEV) distributions using regional L-moments (Hosking & Wallis 1997) or MLE (Atlas 15)
- Compute multiplicative change factors for design return periods (e.g., 10-, 25-, 50-year events) relative to a fixed historical baseline (1950–2000)
- Assess infrastructure risk via effective return period analysis
- Visualize gridded and county-level spatial outputs

## Data
 
### Input Data
 
This framework is designed to work with:
 
- **LOCA (CMIP5):** Daily precipitation at 6-km statistically downscaled from CMIP5. Available from [LLNL LOCA archive]([http://loca.ucsd.edu/](https://gdo-dcp.llnl.gov/downscaled_cmip_projections/dcp/archive/cmip5/loca_hydro/LOCA_VIC_dpierce_2017-02-28).
- **LOCA2 (CMIP6):**  Daily precipitation at 6-km statistically downscaled from CMIP6. Available from [the LOCA2 data portal](https://cirrus.ucsd.edu/~pierce/LOCA2/CONUS_regions_split).
- **NOAA CO-OP Station Network:** [NCEI COOP Hourly Precip](https://www.ncei.noaa.gov/data/coop-hourly-precipitation/v2/access/)
- **NOAA Atlas 14:** Historical observed precipitation frequency estimates used as the baseline for change factor normalization. Available from [NOAA's Precipitation Frequency Data Server](https://hdsc.nws.noaa.gov/pfds/).
- **County shapefiles:** Used as spatial pooling regions. Available from the [U.S. Census Bureau TIGER/Line files](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html).
 
`0)Download_Data.ipynb` provides utilities for downloading and organizing LOCA/LOCA2, Atlas 14, and Station data programmatically.
 
### Intermediate and Output Data
 
The `data/` directory contains examples of intermediate outputs (gridded AMS/PDS) and final outputs (county-level change factor arrays, fitted parameters) and is structured for use with `xarray` and `zarr`/`netCDF4`. Full output data is available in the [Zenodo Repository].

## Usage
 
Notebooks provide utilities and example workflows to download and process data and generate change factors:
 
1. **`0)Download_Data.ipynb`** — Configure paths and download LOCA/LOCA2 precipitation data for your region and model ensemble.
2. **`01)Process_Timeseries.ipynb`** — Extract Annual Maximum Series, fit GEV distributions using regional L-moments, and compute per-model change factors for target return periods.
3. **`02)Viz_Data.ipynb`** — Visualize ensemble change factor distributions, model spread, and spatial patterns at the county level.
4. **`8)Effective_Return_Periods.ipynb`** — Compute effective return periods and quantify infrastructure underdesign risk under projected climate conditions.
 
Configuration (paths, model lists, return periods, future windows) is managed in `modules/config.py`.
