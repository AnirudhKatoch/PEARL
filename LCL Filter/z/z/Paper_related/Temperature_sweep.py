import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size":16, "font.family": "Times New Roman", "axes.labelsize":16, "axes.titlesize":16,"xtick.labelsize":16, "ytick.labelsize":16, "legend.fontsize": 15})


base_dir = "TEMP_profiles_nc"
figures_dir = "Figures"


def _sim_folders(base_dir):
    """Sorted list of Simulation_* folders by numeric index."""
    folders = [d for d in os.listdir(base_dir)
               if d.startswith("Simulation_")
               and os.path.isdir(os.path.join(base_dir, d))]
    folders.sort(key=lambda d: int(d.split("_")[1]))
    return folders


def _read_Tamb_key(df_1):
    """
    Ambient-temperature key for one simulation, in °C.

    T_amb is stored in kelvin in df_1_power_flow_RMS; the first row is taken
    (it is constant across the run) and converted to °C for plotting.
    """
    return float(df_1["T_amb"].iloc[0]) - 273


def _collect(base_dir, specs):
    """
    Generic collector for the temperature sweep.

    `specs` maps an output name -> (parquet_filename, column, reducer),
    where reducer turns the column into a scalar.

    Returns {name: {T_amb_key: value}}.
    """
    out = {name: {} for name in specs}

    for sim in _sim_folders(base_dir):
        df_dir = os.path.join(base_dir, sim, "Dataframes")
        df_1 = pd.read_parquet(os.path.join(df_dir, "df_1_power_flow_RMS.parquet"))
        T_key = _read_Tamb_key(df_1)

        cache = {}  # read each parquet once per simulation
        for name, (fname, col, reducer) in specs.items():
            if fname not in cache:
                cache[fname] = pd.read_parquet(os.path.join(df_dir, fname))
            out[name][T_key] = reducer(cache[fname][col])

    return out


# reducers
def _mean(s):       return float(np.nanmean(s))
def _mean_degC(s):  return float(np.nanmean(s)) - 273.15
def _last_valid(s): return float(s.dropna().iloc[-1])


# ----------------------------------------------------------------------
# Lifetime: the three dictionaries
# ----------------------------------------------------------------------
life = _collect(base_dir, {
    "C":  ("df_3_C.parquet",  "Lifetime_C",  _last_valid),
    "L1": ("df_4_L1.parquet", "Lifetime_L1", _last_valid),
    "L2": ("df_5_L2.parquet", "Lifetime_L2", _last_valid),
})

life_min = {k: min(life["C"][k], life["L1"][k], life["L2"][k])
    for k in life["C"]}

def plot_quantity_vs_Tamb_with_min(C_dict, L1_dict, L2_dict, min_dict, figures_dir,ylabel, fname, titles=None, y_margin=0.05):
    """
    4-row (C, L1, L2, module-min) plot vs ambient temperature.

    Each dict maps T_amb [K] -> value. A single curve per component is drawn
    against ambient temperature in °C. A single shared legend is placed at the
    top of the figure.
    """
    if titles is None:
        titles = ["Capacitor C", "Inverter-side inductor L1",
                  "Grid-side inductor L2", "LCL filter"]

    def xy(d):
        # keys are kelvin; sort and convert to °C for the x-axis
        xs = sorted(d)
        return [x - 0 for x in xs], [d[x] for x in xs]

    def ylim_from(y):
        allv = np.asarray(y, float)
        if allv.size == 0:
            return 0.0, 1.0
        lo, hi = np.nanmin(allv), np.nanmax(allv)
        pad = (hi - lo) * y_margin if hi > lo else (abs(lo) * y_margin or 1.0)
        return lo - pad, hi + pad

    rows = [(titles[0], C_dict), (titles[1], L1_dict),
            (titles[2], L2_dict), (titles[3], min_dict)]
    fig, axes = plt.subplots(4, 1, figsize=(6.4, 4.8 * 4 * 0.33), sharex=True)

    line = None
    for ax, (title, d) in zip(axes, rows):
        x, y = xy(d)
        ln, = ax.plot(x, y, color="blue", marker="o", label="Lifetime")
        line = ln
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(ylim_from(y))
        ax.set_ylim(0,50)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Ambient temperature [°C]")
    axes[-1].set_xlim(0, 100)

    fig.tight_layout()
    fig.savefig(f"{figures_dir}/{fname}.pdf", dpi=300)
    plt.close(fig)


plot_quantity_vs_Tamb_with_min(life["C"], life["L1"], life["L2"], life_min, figures_dir,"Lifetime [years]", "Fig_lifetime_vs_Tamb_with_min")