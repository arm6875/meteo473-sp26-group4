# Milestone 2: Threat Index Development

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import metpy.calc as mpcalc
from metpy.calc import lat_lon_grid_deltas
import matplotlib.colors as mcolors
import os

# ---- Configuration ----
DATA_FILE = "/courses/meteo473/sp26/473_sp26_group4/MergedIncident.nc"
OUTPUT_DIR = "/courses/meteo473/sp26/473_sp26_group4/website/images"
# -----------------------

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# LOAD DATA
# ---------------------------
print("Loading data...")

if not os.path.exists(DATA_FILE):
    print("ERROR: File not found!")
    print(f"Path given: {DATA_FILE}")
    exit()

ds1 = xr.open_dataset(DATA_FILE)
ds1 = ds1.sortby('valid_time')

print("Dataset loaded successfully")
print(ds1)
print(f"Number of timesteps: {len(ds1.valid_time)}")

# ---------------------------
# THREAT INDEX FUNCTION
# ---------------------------
def threat_index(temp_f, dewpt_f, convergence, cape, cin, cloud):

    temp = np.clip(100 * np.exp(-((temp_f - 75)**2) / 200), 0, 100)
    dew_point = 100 / (1 + np.exp(-0.2 * (dewpt_f - 60)))
    cape_index = 100 * (1 - np.exp(-cape / 1000))
    cin_index = 100 * np.exp(-np.abs(cin) / 100)
    cloud_cover = 100 / (1 + np.exp(-0.1 * (cloud - 50)))
    convergent_winds = np.clip(convergence * 1e5, 0, 100)

    index = (0.15 * temp + 0.25 * dew_point + 0.05 * convergent_winds +
             0.30 * cape_index + 0.20 * cin_index + 0.05 * cloud_cover)

    return index

# ---------------------------
# MAP FUNCTION
# ---------------------------
def conus_map():
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([-125, -65, 24, 50])
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)

    return fig, ax

# ---------------------------
# LOOP THROUGH TIME
# ---------------------------
print("Starting loop through timesteps...")

# TEMPORARY: only first 3 timesteps for debugging
for i, time_step in enumerate(ds1.valid_time):

    print(f"\nProcessing timestep {i}")

    try:
        fcst = ds1.sel(valid_time=time_step)

        # Time strings
        init_time = pd.to_datetime(ds1.time.values)
        valid_time = pd.to_datetime(fcst.valid_time.values)

        init_str = init_time.strftime('%HZ %b %d %Y')
        valid_str = valid_time.strftime('%HZ %b %d %Y')

        print(f"Valid time: {valid_str}")

        # Variables
        temp_f = (fcst['t2m'] - 273.15) * 9/5 + 32
        dewpt_f = (fcst['d2m'] - 273.15) * 9/5 + 32
        cape = fcst['cape']
        cin = fcst['cin']
        cloud = fcst['tcc'] * 100

        # Wind + convergence
        dx, dy = lat_lon_grid_deltas(fcst['longitude'].values,
                                    fcst['latitude'].values)

        convergence = -mpcalc.divergence(fcst['u10'], fcst['v10'],
                                        dx=dx, dy=dy)

        # Compute index
        index = threat_index(temp_f, dewpt_f, convergence, cape, cin, cloud)

        print("Index calculated successfully")

        # Plot
        fig, ax = conus_map()

        bounds = np.linspace(0, 100, 101)
        norm = mcolors.BoundaryNorm(bounds, ncolors=256)

        c = ax.contourf(
            fcst['longitude'],
            fcst['latitude'],
            index,
            cmap='RdYlGn_r',
            levels=bounds,
            norm=norm
        )

        plt.colorbar(c, ax=ax, orientation='horizontal', pad=0.02)

        plt.title("Thunderstorm Development Probability Index (0–100)", loc='center', fontweight='bold')
        plt.title(f"Init: {init_str}", loc='left')
        plt.title(f"Valid: {valid_str}", loc='right')

        # Save
        fname = f"threat_{i:03d}.png"
        outpath = os.path.join(OUTPUT_DIR, fname)

        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f" Saved: {outpath}")

    except Exception as e:
        print(f"Error at timestep {i}")
        print(e)
        break

print("Done.")