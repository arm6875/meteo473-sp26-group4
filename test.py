# Milestone 2: Threat Index Development

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import metpy.calc as mpcalc
from metpy.units import units
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from metpy.calc import lat_lon_grid_deltas
import os


# ---- Configuration ----
DATA_FILE = "/courses/meteo473/sp26/473_sp26_group4/data/MergedIncident"
OUTPUT_DIR  = "/courses/meteo473/sp26/473_sp26_group4/website/images"
# -----------------------


# Load Data for Incident Day
print("Loading data")
ds1 = xr.open_dataset(DATA_FILE)
ds1 = ds1.sortby('valid_time')
print(ds1)

# Choose one time index (you can change this later)
fcst1 = ds1.isel(valid_time=48)

# Extract initialization and valid times
init_time = pd.to_datetime(ds1.time.values)
valid_time = pd.to_datetime(fcst1.valid_time.values)

# Format times for plot titles
init_str = init_time.strftime('%HZ %b %d %Y')
valid_str = valid_time.strftime('%HZ %b %d %Y')

# Print out initial and valid times
print("Init Time:", init_str)
print("Valid Time:", valid_str)

#---- Index Creation:

# Convert temperatures to Fahrenheit
temp_f = (fcst1['t2m'] - 273.15) * 9/5 + 32
dewpt_f = (fcst1['d2m'] - 273.15) * 9/5 + 32

cape = fcst1['cape']
cin = fcst1['cin']
cloud = fcst1['tcc'] * 100

# Compute dx and dy from lat/lon
dx, dy = lat_lon_grid_deltas(fcst1['longitude'].values, fcst1['latitude'].values)

# Compute convergence (negative divergence)
convergence = -mpcalc.divergence(fcst1['u10'], fcst1['v10'], dx=dx, dy=dy)

# Define function for Threat Index
def threat_index(temp_f, dewpt_f, convergence, cape, cin, cloud):

    # Temperature (best 60–90°F)
    temp = np.clip(100 * np.exp(-((temp_f - 75)**2) / 200), 0, 100)

    # Dewpoint (logistic growth)
    dew_point = 100 / (1 + np.exp(-0.2 * (dewpt_f - 60)))

    # CAPE (exponential growth)
    cape_index = 100 * (1 - np.exp(-cape / 1000))

    # CIN (inverse exponential decay)
    cin_index = 100 * np.exp(-np.abs(cin) / 100)

    # Cloud cover (logistic)
    cloud_cover = 100 / (1 + np.exp(-0.1 * (cloud - 50)))

    # Convergence (scaled)
    convergent_winds = np.clip(convergence * 1e5, 0, 100)

    # Weighted Index Calculation
    index1 = (0.15 * temp + 0.25 * dew_point + 0.05 * convergent_winds +
             0.30 * cape_index + 0.20 * cin_index + 0.05 * cloud_cover)

    return index1

# Print index maximum and minimum values
index1 = threat_index(temp_f, dewpt_f, convergence, cape, cin, cloud)
index1

# Extract 10-meter wind components and compute wind speed magnitude
u10 = fcst1['u10']
v10 = fcst1['v10']
wind_speed = np.sqrt(u10**2 + v10**2)

##--- Timestep map generated:

# Ensure output directory exists
outdir = OUTPUT_DIR
os.makedirs(outdir, exist_ok=True)

# Conus Map Function
def conus_map():
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-125, -65, 24, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)
    return fig, ax

# Loop over all times in the dataset
for i in range(len(ds1.valid_time)):

    fcst1 = ds1.isel(valid_time=i)

    # Convert times
    init_time = pd.to_datetime(ds1.time.values)
    valid_time = pd.to_datetime(fcst1.valid_time.values)
    init_str = init_time.strftime('%HZ %b %d %Y')
    valid_str = valid_time.strftime('%HZ %b %d %Y')

    # Compute fields
    temp_f = (fcst1['t2m'] - 273.15) * 9/5 + 32
    dewpt_f = (fcst1['d2m'] - 273.15) * 9/5 + 32
    cape = fcst1['cape']
    cin = fcst1['cin']
    cloud = fcst1['tcc'] * 100

    dx, dy = lat_lon_grid_deltas(fcst1['longitude'].values, fcst1['latitude'].values)
    convergence = -mpcalc.divergence(fcst1['u10'], fcst1['v10'], dx=dx, dy=dy)

    # Compute threat index
    index1 = threat_index(temp_f, dewpt_f, convergence, cape, cin, cloud)

    # Plotted conus map
    fig, ax = conus_map()

    bounds = np.linspace(0, 100, 101)
    norm = mcolors.BoundaryNorm(bounds, ncolors=256)

    c = ax.contourf(
        fcst1['longitude'],
        fcst1['latitude'],
        index1,
        cmap='RdYlGn_r',
        levels=bounds,
        norm=norm,
        shading='auto'
    )
    
    cb = plt.colorbar(
        c,
        ax=ax,
        orientation='horizontal',
        pad=0.02,
        ticks=[0, 20, 40, 60, 80, 100]
    )

    plt.title("Thunderstorm Development Probability Index (0–100)", loc='center', fontweight='bold')
    plt.title(f"Init: {init_str}", loc='left')
    plt.title(f"Valid: {valid_str}", loc='right')

    plt.text(
        0.01, 0.01,
        "Parameters: Temp, Dewpoint, CAPE, CIN, Convergence, Cloud Cover",
        transform=ax.transAxes,
        fontsize=9,
        color='white',
        bbox=dict(facecolor='black', alpha=0.4, pad=3)
    )

    # Save files as pngs and save them in a folder of all 48 hours for incident day
    fname = f"threat_{i:03d}.png"

    plt.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved {fname}")



