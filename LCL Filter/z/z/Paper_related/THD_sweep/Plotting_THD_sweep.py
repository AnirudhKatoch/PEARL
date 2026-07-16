import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

base_dir = "THD_Sweep_results"
figures_dir = "Figures"


plt.rcParams.update({"font.size":16, "font.family": "Times New Roman", "axes.labelsize":16, "axes.titlesize":16,"xtick.labelsize":16, "ytick.labelsize":16, "legend.fontsize": 15})

# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------

def _sim_folders(base_dir):
    """Sorted list of Simulation_* folders by numeric index."""
    folders = [d for d in os.listdir(base_dir)
               if d.startswith("Simulation_")
               and os.path.isdir(os.path.join(base_dir, d))]
    folders.sort(key=lambda d: int(d.split("_")[1]))
    return folders

def _read_pf_key(df_1, sim_name):
    """
    Power-factor key for one simulation.

    pf reads back as a plain float (the sign of zero is lost through parquet),
    so the inductive zero cannot be told apart from the capacitive zero by value
    alone. It is therefore identified by its folder number and stored as -0.0,
    which keeps it distinct from the capacitive +0.0 key.
    """
    pf0 = float(df_1["pf"].iloc[0])
    if pf0 == 0.0:
        idx = int(sim_name.split("_")[1])
        return -0.0 if idx == INDUCTIVE_ZERO_SIM else 0.0
    return pf0

schedule = [
    (0.0,  "Simulation_1"),
    (0.5,  "Simulation_2"),
    (1.0,  "Simulation_3"),
    (1.5,  "Simulation_4"),
    (2.0,  "Simulation_5"),
    (2.5,  "Simulation_6"),
    (3.0,  "Simulation_7"),
    (3.5,  "Simulation_8"),
    (4.0,  "Simulation_9"),
]


# distortion scale per simulation, matching your run schedule
DISTORTION_BY_SIM = {name: scale for scale, name in schedule}

def _collect(base_dir, specs):
    """Collect scalars keyed by distortion scale."""
    out = {name: {} for name in specs}

    for sim in _sim_folders(base_dir):
        df_dir = os.path.join(base_dir, sim, "Dataframes")
        key = DISTORTION_BY_SIM[sim]

        cache = {}
        for name, (fname, col, reducer) in specs.items():
            if fname not in cache:
                cache[fname] = pd.read_parquet(os.path.join(df_dir, fname))
            out[name][key] = reducer(cache[fname][col])

    return out

# reducers
def _mean(s):       return float(np.nanmean(s))
def _mean_degC(s):  return float(np.nanmean(s)) - 273.15
def _last_valid(s): return float(s.dropna().iloc[-1])
def _rms(s):  return float(np.sqrt(np.nanmean(np.asarray(s, float) ** 2)))

########################################################################################################################

def plot_thd_vs_distortion(THD_dict, figures_dir, fname="Fig_thd_vs_distortion",
                           ylabel="THD [%]", xlabel="Distortion scale [-]",
                           y_margin=0.05):
    """
    Single-panel plot of grid-side current THD vs the injected distortion scale
    at fixed power factor.
    """
    x = sorted(THD_dict)
    y = [THD_dict[k] for k in x]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(x, y, color="blue", marker="o", linestyle="-")

    allv = np.asarray(y, float)
    lo, hi = np.nanmin(allv), np.nanmax(allv)
    pad = (hi - lo) * y_margin if hi > lo else (abs(lo) * y_margin or 1.0)
    ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(min(x), max(x))

    fig.tight_layout()
    fig.savefig(f"{figures_dir}/{fname}.png", dpi=300)
    plt.close(fig)


thd = _collect(base_dir, {
    "THD": ("df_2_power_flow_inst.parquet", "THD_percent_I_L2", _last_valid),
})

plot_thd_vs_distortion(thd["THD"], figures_dir, "Fig_thd_vs_distortion")



########################################################################################################################

def plot_lifetime_vs_thd(THD_dict, C_dict, L1_dict, L2_dict, min_dict, figures_dir,
                         ylabel="Lifetime [years]", fname="Fig_lifetime_vs_thd",
                         titles=None, power=None, y_margin=0.05):
    """
    4-row (L2, C, L1, LCL) lifetime plot versus the grid-side current THD at
    fixed power factor. THD_dict and the component dicts must share the same
    keys, namely the distortion scale of each simulation.

    `power`, if given, is a list of four (lo, hi) tuples applied per panel in
    order [L2, C, L1, LCL]. Use None for any panel to auto-scale it.
    """
    if titles is None:
        titles = ["Grid-side inductor L2", "Capacitor C",
                  "Inverter-side inductor L1", "LCL filter"]

    def xy(d):
        # order points by increasing THD, not by distortion scale
        keys = sorted(d, key=lambda k: THD_dict[k])
        return [THD_dict[k] for k in keys], [d[k] for k in keys]

    def ylim_from(y):
        allv = np.asarray(y, float)
        lo, hi = np.nanmin(allv), np.nanmax(allv)
        pad = (hi - lo) * y_margin if hi > lo else (abs(lo) * y_margin or 1.0)
        return lo - pad, hi + pad

    rows = [(titles[0], L2_dict), (titles[1], C_dict),
            (titles[2], L1_dict), (titles[3], min_dict)]

    fig, axes = plt.subplots(4, 1, figsize=(6.4, 4.8 * 4 * 0.5), sharex=True)

    for i, (ax, (title, d)) in enumerate(zip(axes, rows)):
        x, y = xy(d)
        ax.plot(x, y, color="blue", marker="o", linestyle="-")
        ax.set_title(title)
        if power is not None and power[i] is not None:
            ax.set_ylim(power[i])
        else:
            ax.set_ylim(ylim_from(y))

    axes[-1].set_xlabel("THD [%]")
    fig.supylabel(ylabel)

    fig.tight_layout(rect=[-0.033, 0, 1, 0.98])
    fig.savefig(f"{figures_dir}/{fname}.png", dpi=300)
    plt.close(fig)

thd = _collect(base_dir, {
    "THD": ("df_2_power_flow_inst.parquet", "THD_percent_I_L2", _last_valid),
})

life = _collect(base_dir, {
    "C":  ("df_3_C.parquet",  "Lifetime_C",  _last_valid),
    "L1": ("df_4_L1.parquet", "Lifetime_L1", _last_valid),
    "L2": ("df_5_L2.parquet", "Lifetime_L2", _last_valid),
})
life_min = {k: min(life["C"][k], life["L1"][k], life["L2"][k]) for k in life["C"]}

plot_lifetime_vs_thd(thd["THD"], life["C"], life["L1"], life["L2"], life_min,
                     figures_dir, fname="Fig_lifetime_vs_thd")


########################################################################################################################


RES_PER_CYCLE = 100000   # actual resolution_per_cycle

def _nonfund_rms(s):
    """
    Non-fundamental (harmonic) RMS of a waveform.

    Fundamental = the single spectral bin at the fundamental, extracted with a
    leakage-free single-bin DFT. Harmonic content = everything else, obtained
    by Parseval as sqrt(total^2 - DC^2 - fundamental^2).
    """
    x = np.asarray(s, dtype=float)
    x = x[np.isfinite(x)]
    N = len(x)

    spc = int(round(RES_PER_CYCLE))
    k = int(round(N / spc))                    # cycles in window = fundamental bin
    n = np.arange(N)
    X = np.dot(x, np.exp(-1j * 2 * np.pi * k * n / N))
    fund_rms = (2.0 * np.abs(X) / N) / np.sqrt(2)

    total_rms = np.sqrt(np.mean(x ** 2))
    dc = np.mean(x)
    return float(np.sqrt(max(total_rms ** 2 - dc ** 2 - fund_rms ** 2, 0.0)))


def plot_harmonic_rms_vs_thd(THD_dict, C_dict, L1_dict, L2_dict, figures_dir,
                             ylabel="Harmonic RMS current [A]",
                             fname="Fig_harmonic_rms_vs_thd",
                             titles=None, power=None, y_margin=0.05):
    """
    3-row (L2, C, L1) plot of the non-fundamental RMS current of each component
    versus the grid-side current THD at fixed power factor.
    """
    if titles is None:
        titles = ["Grid-side inductor L2", "Capacitor C",
                  "Inverter-side inductor L1"]

    def xy(d):
        keys = sorted(d, key=lambda k: THD_dict[k])
        return [THD_dict[k] for k in keys], [d[k] for k in keys]

    def ylim_from(y):
        allv = np.asarray(y, float)
        lo, hi = np.nanmin(allv), np.nanmax(allv)
        pad = (hi - lo) * y_margin if hi > lo else (abs(lo) * y_margin or 1.0)
        return lo - pad, hi + pad

    rows = [(titles[0], L2_dict), (titles[1], C_dict), (titles[2], L1_dict)]
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 4.8 * 3 * 0.4), sharex=True)

    for i, (ax, (title, d)) in enumerate(zip(axes, rows)):
        x, y = xy(d)
        ax.plot(x, y, color="blue", marker="o", linestyle="-")
        ax.set_title(title)
        if power is not None and power[i] is not None:
            ax.set_ylim(power[i])
        else:
            ax.set_ylim(ylim_from(y))

    axes[-1].set_xlabel("THD [%]")
    fig.supylabel(ylabel)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(f"{figures_dir}/{fname}.png", dpi=300)
    plt.close(fig)

thd = _collect(base_dir, {
    "THD": ("df_2_power_flow_inst.parquet", "THD_percent_I_L2", _last_valid),
})

hrm = _collect(base_dir, {
    "C":  ("df_2_power_flow_inst.parquet", "I_C",  _nonfund_rms),
    "L1": ("df_2_power_flow_inst.parquet", "I_L1", _nonfund_rms),
    "L2": ("df_2_power_flow_inst.parquet", "I_L2", _nonfund_rms),
})

plot_harmonic_rms_vs_thd(thd["THD"], hrm["C"], hrm["L1"], hrm["L2"], figures_dir)


###########################################################################################################################



def plot_harmonic_vrms_vs_thd(THD_dict, C_dict, L1_dict, L2_dict, figures_dir,
                              ylabel="Harmonic RMS voltage [V]",
                              fname="Fig_harmonic_vrms_vs_thd",
                              titles=None, power=None, y_margin=0.05):
    """
    3-row (L2, C, L1) plot of the non-fundamental RMS voltage of each component
    versus the grid-side current THD at fixed power factor.
    """
    if titles is None:
        titles = ["Grid-side inductor L2", "Capacitor C",
                  "Inverter-side inductor L1"]

    def xy(d):
        keys = sorted(d, key=lambda k: THD_dict[k])
        return [THD_dict[k] for k in keys], [d[k] for k in keys]

    def ylim_from(y):
        allv = np.asarray(y, float)
        lo, hi = np.nanmin(allv), np.nanmax(allv)
        pad = (hi - lo) * y_margin if hi > lo else (abs(lo) * y_margin or 1.0)
        return lo - pad, hi + pad

    rows = [(titles[0], L2_dict), (titles[1], C_dict), (titles[2], L1_dict)]
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 4.8 * 3 * 0.4), sharex=True)

    for i, (ax, (title, d)) in enumerate(zip(axes, rows)):
        x, y = xy(d)
        ax.plot(x, y, color="blue", marker="o", linestyle="-")
        ax.set_title(title)
        if power is not None and power[i] is not None:
            ax.set_ylim(power[i])
        else:
            ax.set_ylim(ylim_from(y))

    axes[-1].set_xlabel("THD [%]")
    fig.supylabel(ylabel)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(f"{figures_dir}/{fname}.png", dpi=300)
    plt.close(fig)

vhrm = _collect(base_dir, {
    "C":  ("df_2_power_flow_inst.parquet", "V_C",  _nonfund_rms),
    "L1": ("df_2_power_flow_inst.parquet", "V_L1", _nonfund_rms),
    "L2": ("df_2_power_flow_inst.parquet", "V_L2", _nonfund_rms),
})

plot_harmonic_vrms_vs_thd(thd["THD"], vhrm["C"], vhrm["L1"], vhrm["L2"], figures_dir)