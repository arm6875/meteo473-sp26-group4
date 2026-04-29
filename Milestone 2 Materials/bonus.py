# Milestone 2 + Bonus: Automated Threat Index (FINAL CLEAN VERSION)

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
from datetime import datetime, timedelta, timezone

# -----------------------
# CONFIGURATION
# -----------------------
BASE_DIR = "/courses/meteo473/sp26/473_sp26_group4/website/images_bonus"
DATA_FILE = "/courses/meteo473/sp26/473_sp26_group4/HRRR.nc"

os.makedirs(BASE_DIR, exist_ok=True)

# -----------------------
# LOAD DATA (NO HRRR)
# -----------------------
def load_data():
    print("Loading dataset...")
    ds = xr.open_dataset(DATA_FILE)
    ds = ds.sortby("valid_time")
    return ds

# -----------------------
# TIME (FIXED DEPRECATION)
# -----------------------
def get_run_time():
    now = datetime.now(timezone.utc) - timedelta(hours=1)
    date_str = now.strftime("%Y%m%d")
    run_str = f"{now.hour:02d}"
    return date_str, run_str

# -----------------------
# THREAT INDEX
# -----------------------
def threat_index(temp_f, dewpt_f, convergence, cape, cin, cloud):

    temp = np.clip(100 * np.exp(-((temp_f - 75)**2) / 200), 0, 100)
    dew_point = 100 / (1 + np.exp(-0.2 * (dewpt_f - 60)))
    cape_index = 100 * (1 - np.exp(-cape / 1000))
    cin_index = 100 * np.exp(-np.abs(cin) / 100)
    cloud_cover = 100 / (1 + np.exp(-0.1 * (cloud - 50)))
    convergent_winds = np.clip(convergence * 1e5, 0, 100)

    return (
        0.15 * temp +
        0.25 * dew_point +
        0.05 * convergent_winds +
        0.30 * cape_index +
        0.20 * cin_index +
        0.05 * cloud_cover
    )

# -----------------------
# MAP
# -----------------------
def conus_map():
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([-125, -65, 24, 50])
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)

    return fig, ax

# -----------------------
# MAIN
# -----------------------
print("Loading data...")

try:
    ds = load_data()
except Exception as e:
    print("Failed to load dataset:", e)
    exit()

print("Dataset loaded successfully")

date_str, run_str = get_run_time()
run_folder = os.path.join(BASE_DIR, f"{date_str}_{run_str}")
os.makedirs(run_folder, exist_ok=True)

print("Processing timesteps...")

for i, time_step in enumerate(ds.valid_time):

    print(f"Processing timestep {i}")

    try:
        fcst = ds.sel(valid_time=time_step)

        # FIXED TIME HANDLING
        init_time = pd.to_datetime(fcst.time.values)
        valid_time = pd.to_datetime(fcst.valid_time.values)

        init_str = init_time.strftime('%HZ %b %d %Y')
        valid_str = valid_time.strftime('%HZ %b %d %Y')

        # VARIABLES
        temp_f = (fcst['t2m'] - 273.15) * 9/5 + 32
        dewpt_f = (fcst['d2m'] - 273.15) * 9/5 + 32
        cape = fcst['cape']
        cin = fcst['cin']
        cloud = fcst['tcc'] * 100

        # WIND CONVERGENCE
        dx, dy = lat_lon_grid_deltas(
            fcst['longitude'].values,
            fcst['latitude'].values
        )

        convergence = -mpcalc.divergence(
            fcst['u10'],
            fcst['v10'],
            dx=dx,
            dy=dy
        )

        # THREAT INDEX
        index = threat_index(
            temp_f,
            dewpt_f,
            convergence,
            cape,
            cin,
            cloud
        )

        # PLOT
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

        plt.title("Thunderstorm Development Probability Index", fontweight='bold')
        plt.title(f"Init: {init_str}", loc='left')
        plt.title(f"Valid: {valid_str}", loc='right')

        # SAVE (NO OVERLAP)
        fname = f"threat_{i:03d}.png"
        outpath = os.path.join(run_folder, fname)

        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved: {outpath}")

    except Exception as e:
        print(f"Error at timestep {i}: {e}")
        continue

print("Done.")
