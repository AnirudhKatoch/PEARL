import os
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update({"font.size":15, "font.family": "Times New Roman", "axes.labelsize":15, "axes.titlesize":15,"xtick.labelsize":15, "ytick.labelsize":15, "legend.fontsize": 15})


# directory holding the .nc files
nc_dir = r"E:\PEARL\LCL Filter\z\Paper_related\Temperature cities\TEMP_profiles_nc"   # <-- folder with the .nc files

# city -> (filename, exact lat, lon)
SITES = {
    "Munich":     ("Munich_2025.nc",     48.149, 11.568),
    "Koudougou":  ("Koudougou_2025.nc",  12.250, -2.363),
    "Bahir_Dar":  ("Bahir_Dar_2025.nc",  11.594, 37.388),
    "Kumasi":     ("Kumasi_2025.nc",      6.674, -1.572),
    "Mumbai":     ("Mumbai_2025.nc",     19.133, 72.915),
    "Bandung":    ("Bandung_2025.nc",    -6.890, 107.610),
    "Juja ": ("Juja_2025.nc",                 -1.094, 37.014),
    "Windhoek": ("Windhoek_2025.nc",         -22.565, 17.075),
    "Lima": ("Lima_2025.nc",                 -12.069, -77.080),
    "Stellenbosch": ("Stellenbosch_2025.nc", -33.928, 18.866),
    "Kampala": ("Kampala_2025.nc",             0.330, 32.570),
}

def print_site_coordinates(nc_dir, sites):
    """
    For each site, open its ERA5 .nc, select the grid cell nearest the city
    coordinates, and print the requested vs. resolved grid coordinates along
    with a temperature summary (°C).
    """
    print(f"{'City':12s} {'req lat':>8s} {'req lon':>8s} "
          f"{'grid lat':>9s} {'grid lon':>9s} "
          f"{'min':>6s} {'mean':>6s} {'max':>6s}  [°C]")
    print("-" * 78)

    for name, (fname, lat, lon) in sites.items():
        ds = xr.open_dataset(os.path.join(nc_dir, fname))

        da = ds["t2m"]
        for extra in ("number", "expver"):
            if extra in da.dims:
                da = da.isel({extra: 0}, drop=True)

        cell = da.sel(latitude=lat, longitude=lon, method="nearest")
        glat = float(cell["latitude"])
        glon = float(cell["longitude"])

        s = cell.to_series().astype(float) - 273.15   # K -> °C
        print(f"{name:12s} {lat:8.3f} {lon:8.3f} "
              f"{glat:9.3f} {glon:9.3f} "
              f"{s.min():6.1f} {s.mean():6.1f} {s.max():6.1f}")

        ds.close()
def get_temp_dataframe(nc_dir, fname, lat, lon, in_celsius=True):
    """
    Load one ERA5 .nc file and return a DataFrame with two columns:
    'time' and 'T_amb' for the grid cell nearest (lat, lon).

    in_celsius=True -> T_amb in °C; False -> kelvin.
    """
    ds = xr.open_dataset(os.path.join(nc_dir, fname))

    da = ds["t2m"]
    for extra in ("number", "expver"):
        if extra in da.dims:
            da = da.isel({extra: 0}, drop=True)

    cell = da.sel(latitude=lat, longitude=lon, method="nearest")

    df = pd.DataFrame({
        "time":  pd.to_datetime(cell["valid_time"].values),
        "T_amb": cell.to_series().astype(float).values,
    })
    if in_celsius:
        df["T_amb"] = df["T_amb"] - 273.15

    ds.close()
    return df

def build_all_cities_dataframe(nc_dir, sites, in_celsius=True):
    """
    Build a single DataFrame indexed by time, with one column per city
    holding its T_amb series (°C if in_celsius, else K).
    """
    series = {}
    for name, (fname, lat, lon) in sites.items():
        df = get_temp_dataframe(nc_dir, fname, lat, lon, in_celsius=in_celsius)
        series[name] = df.set_index("time")["T_amb"]

    out = pd.DataFrame(series)
    out.index.name = "time"
    return out

df_all = build_all_cities_dataframe(nc_dir, SITES)

def plot_all_cities_daily(df_all, figures_dir=None, fname="Fig_cities_temperature"):
    """
    Resample each city's hourly series to daily mean (365 values) and plot
    all cities overlaid on a single axis.

    df_all : DataFrame indexed by time, one column per city (°C).
    """
    daily = df_all.resample("1D").mean()   # 8760 hourly -> 365 daily


    fig, ax = plt.subplots(figsize=(6.4*2, 4.8))
    for city in daily.columns:
        ax.plot(daily.index, daily[city], linewidth=1.2, label=city)

    ax.set_xlabel("Date")
    ax.set_ylabel("Ambient temperature [°C]")
    ax.set_title("Daily mean 2 m air temperature, 2025 (ERA5)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=11)
    fig.tight_layout()
    ax.set_xlim(min(daily.index),max(daily.index))

    if figures_dir:
        fig.savefig(f"{figures_dir}/{fname}.png", dpi=300)
        plt.close(fig)
    else:
        plt.show()

    return daily

daily = plot_all_cities_daily(df_all,figures_dir="Figures")



# hourly master (8760 rows) — full-resolution source of truth
df_all.to_parquet("E:/PEARL/LCL Filter/z/Paper_related/Temperature cities/Temperature_2025_hourly.parquet")

# daily average (365 rows) — one value per day
daily = df_all.resample("1D").mean()
daily.to_parquet(os.path.join("E:/PEARL/LCL Filter/z/Paper_related/Temperature cities/Temperature_2025_daily.parquet"))


df_daily  = pd.read_parquet("Temperature_2025_daily.parquet")
Munich_T = df_daily["Munich"]
Munich_T = Munich_T.to_numpy() +273.15
print(Munich_T)