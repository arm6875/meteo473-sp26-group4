# Milestone 2 + Bonus: Automated Threat Index (FINAL CLEAN VERSION)
# Milestone 2 + Bonus: Automated Threat Index — CRON-READY VERSION

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
from herbie import FastHerbie

# -----------------------
# CONFIGURATION
# -----------------------
BASE_DIR   = "/courses/meteo473/sp26/473_sp26_group4/website/images_cron_test"
HERBIE_DIR = "/courses/meteo473/sp26/473_sp26_group4/herbie_cache"
os.makedirs(BASE_DIR,   exist_ok=True)
os.makedirs(HERBIE_DIR, exist_ok=True)

# -----------------------
# LOAD DATA
# -----------------------
def load_todays_data():
    # Lag by 2 hours for HRRR availability; floor to nearest hour
    now = datetime.now(timezone.utc) - timedelta(hours=2)
    run = now.replace(minute=0, second=0, microsecond=0)

    print(f"Fetching HRRR run: {run.strftime('%Y-%m-%d %H:%M UTC')}")

    H = FastHerbie(
        [run.strftime("%Y-%m-%d %H:%M")],
        model="hrrr",
        fxx=list(range(0, 49)),
        save_dir=HERBIE_DIR,
        overwrite=False       # don't re-download if already cached
    )

    # ---- FIXED: same search string pattern that worked in your original code ----
    ss = "(:TMP:2 m|:CAPE:surface:|:TCDC:entire|:CIN:surface:|:UGRD:10 m above|:VGRD:10 m above|:DPT:2 m above)"
    H.download(ss)

    # Load each variable separately then merge — avoids grid mismatch errors
    ds_sfc  = H.xarray(":(?:TMP|DPT):2 m above ground:")
    ds_wind = H.xarray(":(?:UGRD|VGRD):10 m above ground:")
    ds_cape = H.xarray(":CAPE:surface:")
    ds_cin  = H.xarray(":CIN:surface:")
    ds_cloud= H.xarray(":TCDC:entire atmosphere:")

    ds = xr.merge([ds_sfc, ds_wind, ds_cape, ds_cin, ds_cloud])
    return ds, run

# -----------------------
# THREAT INDEX
# -----------------------
def threat_index(temp_f, dewpt_f, convergence, cape, cin, cloud):
    temp             = np.clip(100 * np.exp(-((temp_f - 75)**2) / 200), 0, 100)
    dew_point        = 100 / (1 + np.exp(-0.2 * (dewpt_f - 60)))
    cape_index       = 100 * (1 - np.exp(-cape / 1000))
    cin_index        = 100 * np.exp(-np.abs(cin) / 100)
    cloud_cover      = 100 / (1 + np.exp(-0.1 * (cloud - 50)))
    # FIX: multiply by 1e5 brings s^-1 to ~0-1 range; *100 scales to 0-100
    convergent_winds = np.clip(convergence * 1e5 * 100, 0, 100)

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
    ax  = plt.axes(projection=ccrs.PlateCarree())
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
    ds, run_time = load_todays_data()
except Exception as e:
    print(f"FATAL: Failed to load dataset: {e}")
    raise SystemExit(1)          # exit code 1 so cron can detect failure

print("Dataset loaded successfully")

# Derive folder name from ACTUAL data run time — not a separate clock call
date_str   = run_time.strftime("%Y%m%d")
run_str    = run_time.strftime("%H")
run_folder = os.path.join(BASE_DIR, f"{date_str}_{run_str}")
os.makedirs(run_folder, exist_ok=True)

print(f"Output folder: {run_folder}")
print("Processing timesteps...")

for i, time_step in enumerate(ds.valid_time):
    print(f"Processing timestep {i}")
    try:
        fcst = ds.sel(valid_time=time_step)

        init_time  = pd.to_datetime(fcst.time.values)
        valid_time = pd.to_datetime(fcst.valid_time.values)
        init_str   = init_time.strftime('%HZ %b %d %Y')
        valid_str  = valid_time.strftime('%HZ %b %d %Y')

        temp_f  = (fcst['t2m']  - 273.15) * 9/5 + 32
        dewpt_f = (fcst['d2m']  - 273.15) * 9/5 + 32
        cape    = fcst['cape']
        cin     = fcst['cin']
        cloud   = fcst['tcc'] * 100

        dx, dy = lat_lon_grid_deltas(
            fcst['longitude'].values,
            fcst['latitude'].values
        )

        # FIX: .metpy.dequantify() strips Pint units so numpy ops work cleanly
        convergence = (-mpcalc.divergence(
            fcst['u10'], fcst['v10'], dx=dx, dy=dy
        )).metpy.dequantify()

        index = threat_index(temp_f, dewpt_f, convergence, cape, cin, cloud)

        fig, ax = conus_map()
        bounds  = np.linspace(0, 100, 101)
        norm    = mcolors.BoundaryNorm(bounds, ncolors=256)

        c = ax.contourf(
            fcst['longitude'], fcst['latitude'], index,
            cmap='RdYlGn_r', levels=bounds, norm=norm
        )

        plt.colorbar(c, ax=ax, orientation='horizontal', pad=0.02)
        plt.title("Thunderstorm Development Probability Index", fontweight='bold')
        plt.title(f"Init: {init_str}",  loc='left')
        plt.title(f"Valid: {valid_str}", loc='right')

        fname   = f"threat_{i:03d}.png"
        outpath = os.path.join(run_folder, fname)
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {outpath}")

    except Exception as e:
        print(f"Error at timestep {i}: {e}")
        continue

print("Done.")