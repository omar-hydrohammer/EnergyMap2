# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:19:14 2025

@author: oalma
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time
import psutil
import os
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import gc
import matplotlib.ticker as ticker

# File paths
elevation_file = r"C:\Users\oalma\Downloads\ETOPO_2022_v1_30s_N90W180_surface.nc"  # ✅ Correct file
precip_files = [
    r"C:\Users\oalma\Downloads\FLDAS_NOAH01_CP_GL_M.A202401.001.nc",
    r"C:\Users\oalma\Downloads\FLDAS_NOAH01_CP_GL_M.A202402.001.nc",
    r"C:\Users\oalma\Downloads\FLDAS_NOAH01_CP_GL_M.A202403.001.nc",
    r"C:\Users\oalma\Downloads\FLDAS_NOAH01_CP_GL_M.A202404.001.nc",
    r"C:\Users\oalma\Downloads\FLDAS_NOAH01_CP_GL_M.A202405.001.nc",
    r"C:\Users\oalma\Downloads\FLDAS_NOAH01_CP_GL_M.A202406.001.nc",
    r"C:\Users\oalma\Downloads\FLDAS_NOAH01_CP_GL_M.A202407.001.nc",
    r"C:\Users\oalma\Downloads\FLDAS_NOAH01_CP_GL_M.A202408.001.nc",
    r"C:\Users\oalma\Downloads\FLDAS_NOAH01_CP_GL_M.A202409.001.nc",
    r"C:\Users\oalma\Downloads\FLDAS_NOAH01_CP_GL_M.A202410.001.nc",
    r"C:\Users\oalma\Downloads\FLDAS_NOAH01_CP_GL_M.A202411.001.nc",
    r"C:\Users\oalma\Downloads\FLDAS_NOAH01_CP_GL_M.A202412.001.nc",
]
output_file = r"C:\Users\oalma\Downloads\energy_map_high_quality.png"

# Constants
GRAVITY = 9.81
EXPONENT = 2.718  # Base of the natural logarithm

# Define color palette with smooth transitions
colors = [
    (128 / 255, 128 / 255, 128 / 255),  # grey
    (192 / 255, 192 / 255, 192 / 255),  # light grey
    (220 / 255, 220 / 255, 220 / 255),  # very light grey
    (98 / 255, 175 / 255, 222 / 255),   # blue
    (3 / 255, 106 / 255, 172 / 255),    # deep blue
]
custom_cmap = LinearSegmentedColormap.from_list("custom_energy_cmap", colors, N=256)

def check_memory():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    print(f"Current memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def load_filtered_elevation(elevation_file):
    """ Load and filter surface elevation from NetCDF (set ocean to 0). """
    ds = xr.open_dataset(elevation_file)
    elevation_data = ds["z"].squeeze()
    elevation_data = elevation_data.where(elevation_data > 0, 0)  # Remove ocean depths
    ds.close()
    return elevation_data

def generate_energy_map(elevation_file, precip_files, output_file):
    """
    Generate an energy potential map based on elevation and precipitation data.
    """
    start_time = time.time()
    start_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    print("\nLoading elevation data...")
    elevation_data = load_filtered_elevation(elevation_file)

    print("\nLoading precipitation data...")
    precip_var = "Rainf_f_tavg"
    precip_data = []

    for file in tqdm(precip_files, desc="Loading precipitation files", position=1, leave=False):
        ds = xr.open_dataset(file)
        precip_data.append(ds[precip_var].squeeze())
        ds.close()

    # Compute annual mean precipitation
    annual_precip = xr.concat(precip_data, dim="time").mean(dim="time")
    annual_precip = annual_precip.where(annual_precip > 0)

    check_memory()

    print("\nRegridding elevation data...")
    elevation_resampled = elevation_data.interp(lat=annual_precip.Y, lon=annual_precip.X, method="linear")

    check_memory()

    print("\nCalculating energy potential...")
    precip_meters = annual_precip * 365 * 24 * 60 * 60 / 1000

    lat_res = (annual_precip.Y[1] - annual_precip.Y[0]).values.item()
    lon_res = (annual_precip.X[1] - annual_precip.X[0]).values.item()
    pixel_area_m2 = (111_139 * lat_res) * (111_139 * lon_res)

    mass_of_water = precip_meters * pixel_area_m2 * 1000
    energy_potential_joules = EXPONENT * mass_of_water * GRAVITY * elevation_resampled
    energy_potential_TWh = energy_potential_joules / (3.6e+15)

    check_memory()

    energy_potential_TWh = energy_potential_TWh.where(energy_potential_TWh > 0)

    vmin = energy_potential_TWh.quantile(0.01).compute().values.item()
    vmax = energy_potential_TWh.quantile(0.99).compute().values.item()

    vmin = max(vmin, 1e-3)
    vmax = max(vmax, 1e0)

    print(f"Using vmin: {vmin}, vmax: {vmax}")

    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    print("\nCreating high-quality energy map...")
    fig, ax = plt.subplots(figsize=(16, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_facecolor('white')

    energy_potential_TWh.plot(
        ax=ax,
        cmap=custom_cmap,
        transform=ccrs.PlateCarree(),
        norm=norm,
        add_colorbar=True,
        cbar_kwargs={
            'label': 'Energy Potential (TWh/year)',
            'orientation': 'horizontal',
            'pad': 0.1,
            'fraction': 0.046,
            'aspect': 50
        },
        rasterized=True
    )

    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m',
                    edgecolor='black', facecolor='none'), linewidth=0.3, alpha=0.7)

    plt.title("Global Energy Potential from Precipitation and Elevation (TWh/year)", pad=20, size=16, weight='bold')
    plt.figtext(0.99, 0.01, f"Generated: {time.strftime('%Y-%m-%d')}", ha='right', va='bottom', size=8, style='italic')

    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    plt.close()

    print(f"\n✅ High-quality energy map saved at: {output_file}")

if __name__ == "__main__":
    generate_energy_map(elevation_file, precip_files, output_file)
