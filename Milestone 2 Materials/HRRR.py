# Automated Thunderstorm Development Index — CRON READY VERSION

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
# CONFIG
# -----------------------

BASE_DIR = "/courses/meteo473/sp26/473_sp26_group4/website/images"

HERBIE_DIR = "/courses/meteo473/sp26/473_sp26_group4/herbie_cache"

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(HERBIE_DIR, exist_ok=True)

LATEST_DIR = os.path.join(BASE_DIR, "latest")

# Always overwrite latest so website always reads same folder
os.makedirs(LATEST_DIR, exist_ok=True)

# -----------------------
# LOAD DATA
# -----------------------

def load_hrrr():
    now = datetime.now(timezone.utc) - timedelta(hours=2)
    run = now.replace(minute=0, second=0, microsecond=0)

    print(f"Downloading HRRR run: {run}")

    H = FastHerbie(
        [run.strftime("%Y-%m-%d %H:%M")],
        model="hrrr",
        fxx=list(range(0, 49)),
        save_dir=HERBIE_DIR,
        overwrite=False
    )

    search = "(:TMP:2 m|:DPT:2 m|:CAPE:surface:|:CIN:surface:|:TCDC:entire|:UGRD:10 m|:VGRD:10 m)"

    H.download(search)

    ds = xr.merge([
        H.xarray(":(?:TMP|DPT):2 m"),
        H.xarray(":(?:UGRD|VGRD):10 m"),
        H.xarray(":CAPE:surface:"),
        H.xarray(":CIN:surface:"),
        H.xarray(":TCDC:entire"),
    ])

    return ds, run


# -----------------------
# INDEX FUNCTION
# -----------------------

def threat_index(temp_f, dew_f, convergence, cape, cin, cloud):

    temp = np.clip(100 * np.exp(-((temp_f - 75)**2) / 200), 0, 100)

    dew = 100 / (1 + np.exp(-0.2 * (dew_f - 60)))

    cape_i = 100 * (1 - np.exp(-cape / 1000))

    cin_i = 100 * np.exp(-np.abs(cin) / 100)

    cloud_i = 100 / (1 + np.exp(-0.1 * (cloud - 50)))

    conv_i = np.clip(convergence * 1e5 * 100, 0, 100)

    return (
        0.15 * temp +
        0.25 * dew +
        0.05 * conv_i +
        0.30 * cape_i +
        0.20 * cin_i +
        0.05 * cloud_i
    )


# -----------------------
# MAP SETUP
# -----------------------

def make_map():
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-125, -65, 24, 50])
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)
    return fig, ax


# -----------------------
# MAIN SCRIPT
# -----------------------

print("Starting run...")

ds, run_time = load_hrrr()

run_str = run_time.strftime("%Y%m%d_%H")

print("Processing forecast hours...")

for i, vt in enumerate(ds.valid_time):

    try:
        fcst = ds.sel(valid_time=vt)

        temp = (fcst.t2m - 273.15) * 9/5 + 32
        dew  = (fcst.d2m - 273.15) * 9/5 + 32
        cape = fcst.cape
        cin  = fcst.cin
        cloud = fcst.tcc * 100

        dx, dy = lat_lon_grid_deltas(fcst.longitude, fcst.latitude)

        convergence = (-mpcalc.divergence(
            fcst.u10, fcst.v10, dx=dx, dy=dy
        )).metpy.dequantify()

        index = threat_index(temp, dew, convergence, cape, cin, cloud)

        fig, ax = make_map()

        levels = np.linspace(0, 100, 101)
        norm = mcolors.BoundaryNorm(levels, 256)

        c = ax.contourf(
            fcst.longitude,
            fcst.latitude,
            index,
            levels=levels,
            cmap="RdYlGn_r",
            norm=norm
        )

        plt.colorbar(c, ax=ax, orientation="horizontal", pad=0.02)

        plt.title("Thunderstorm Development Index")

        # IMPORTANT: simple naming for JS compatibility
        fname = f"threat_{i:03d}.png"
        out = os.path.join(LATEST_DIR, fname)

        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved {out}")

    except Exception as e:
        print(f"Failed timestep {i}: {e}")

print("DONE")
