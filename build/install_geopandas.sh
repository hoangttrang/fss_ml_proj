#!/bin/bash

# Install compatible versions in correct order
pip install --upgrade pip
pip install numpy==1.24.0
pip install pandas==2.0.0
pip install pyproj>=3.5.0 shapely>=2.0.0
pip install matplotlib>=3.7

# Install GDAL first (important!)
pip install GDAL==3.5.0

# Now install geopandas and rasterio
pip install geopandas
pip install rasterio
