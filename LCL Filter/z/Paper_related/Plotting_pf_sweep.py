import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base_dir = "pf_sweep_results"
figures_dir = "pf_sweep_results/Figures"

# Folder index that holds the INDUCTIVE pf = 0 run (pf reads back as 0.0,
# so we tag it by folder number instead of by the float sign).
INDUCTIVE_ZERO_SIM = 21

plt.rcParams.update({"font.size":15, "font.family": "Times New Roman", "axes.labelsize":15, "axes.titlesize":15,"xtick.labelsize":15, "ytick.labelsize":15, "legend.fontsize": 15})

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


def _collect(base_dir, specs):
    """
    Generic collector.

    `specs` maps an output name -> (parquet_filename, column, reducer),
    where reducer turns the column into a scalar.

    Returns {name: {pf_key: value}}.
    """
    out = {name: {} for name in specs}

    for sim in _sim_folders(base_dir):
        df_dir = os.path.join(base_dir, sim, "Dataframes")
        df_1 = pd.read_parquet(os.path.join(df_dir, "df_1_power_flow_RMS.parquet"))
        pf_key = _read_pf_key(df_1, sim)

        cache = {}  # read each parquet once per simulation
        for name, (fname, col, reducer) in specs.items():
            if fname not in cache:
                cache[fname] = pd.read_parquet(os.path.join(df_dir, fname))
            out[name][pf_key] = reducer(cache[fname][col])

    return out


# reducers
def _mean(s):       return float(np.nanmean(s))
def _mean_degC(s):  return float(np.nanmean(s)) - 273.15
def _last_valid(s): return float(s.dropna().iloc[-1])


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_quantity_vs_pf(C_dict, L1_dict, L2_dict, figures_dir, ylabel, fname, titles=None, y_margin=0.05):
    """
    3-row (C, L1, L2) plot vs power factor. Capacitive (+pf, incl. +0.0) and
    inductive (-pf, incl. -0.0) branches are overlaid against |pf|.

    The capacitive and inductive zeros are GENUINE separate runs (keyed +0.0
    and -0.0), so each branch has its own real endpoint at pf = 0. Only the
    unity point (pf = 1, one run) is shared between the branches.

    A single shared legend is placed at the top of the figure.
    """
    if titles is None:
        titles = ["Capacitor C", "Inverter-side inductor L1", "Grid-side inductor L2"]

    def split_branches(d):
        # signbit: -0.0 -> inductive, +0.0 -> capacitive
        cap = {abs(pf): v for pf, v in d.items() if not np.signbit(pf)}
        ind = {abs(pf): v for pf, v in d.items() if np.signbit(pf)}

        # unity (pf = 1) is a single run; share it with the inductive branch
        if 1.0 in cap and 1.0 not in ind:
            ind[1.0] = cap[1.0]
        # fallback: if no inductive zero was simulated, copy the capacitive one
        if 0.0 in cap and 0.0 not in ind:
            ind[0.0] = cap[0.0]

        cap_x = sorted(cap); ind_x = sorted(ind)
        return cap_x, [cap[x] for x in cap_x], ind_x, [ind[x] for x in ind_x]

    def ylim_from(*ys):
        allv = np.concatenate([np.asarray(y, float) for y in ys if len(y)])
        if allv.size == 0:
            return 0.0, 1.0
        lo, hi = np.nanmin(allv), np.nanmax(allv)
        pad = (hi - lo) * y_margin if hi > lo else (abs(lo) * y_margin or 1.0)
        return lo - pad, hi + pad

    rows = [(titles[0], C_dict), (titles[1], L1_dict), (titles[2], L2_dict)]
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 4.8 * 3 * 0.40), sharex=True)

    cap_line = ind_line = None
    for ax, (title, d) in zip(axes, rows):
        cap_x, cap_y, ind_x, ind_y = split_branches(d)
        cl, = ax.plot(cap_x, cap_y, color="blue", marker="o", label="Capacitive")
        il, = ax.plot(ind_x, ind_y, color="red", marker="s", linestyle="--", label="Inductive")
        cap_line, ind_line = cl, il
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(ylim_from(cap_y, ind_y))
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Power factor [-]")
    axes[-1].set_xlim(0, 1)

    # single shared legend at the top
    fig.legend(handles=[cap_line, ind_line], labels=["Capacitive", "Inductive"],
               loc="upper center", ncol=2, frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 0.955])  # leave room for the legend
    fig.savefig(f"{figures_dir}/{fname}.pdf", dpi=300)
    plt.close(fig)

def plot_quantity_vs_pf_with_min(C_dict, L1_dict, L2_dict, min_dict, figures_dir,
                                 ylabel, fname, titles=None, y_margin=0.05):
    """
    4-row (C, L1, L2, module-min) plot vs power factor. Capacitive (+pf, incl.
    +0.0) and inductive (-pf, incl. -0.0) branches are overlaid against |pf|.

    A single shared legend is placed at the top of the figure.
    """
    if titles is None:
        titles = ["Capacitor C", "Inverter-side inductor L1",
                  "Grid-side inductor L2", "LCL filter"]

    def split_branches(d):
        cap = {abs(pf): v for pf, v in d.items() if not np.signbit(pf)}
        ind = {abs(pf): v for pf, v in d.items() if np.signbit(pf)}
        if 1.0 in cap and 1.0 not in ind:
            ind[1.0] = cap[1.0]
        if 0.0 in cap and 0.0 not in ind:
            ind[0.0] = cap[0.0]
        cap_x = sorted(cap); ind_x = sorted(ind)
        return cap_x, [cap[x] for x in cap_x], ind_x, [ind[x] for x in ind_x]

    def ylim_from(*ys):
        allv = np.concatenate([np.asarray(y, float) for y in ys if len(y)])
        if allv.size == 0:
            return 0.0, 1.0
        lo, hi = np.nanmin(allv), np.nanmax(allv)
        pad = (hi - lo) * y_margin if hi > lo else (abs(lo) * y_margin or 1.0)
        return lo - pad, hi + pad

    rows = [(titles[0], C_dict), (titles[1], L1_dict),
            (titles[2], L2_dict), (titles[3], min_dict)]
    fig, axes = plt.subplots(4, 1, figsize=(6.4, 4.8 * 4 * 0.40), sharex=True)

    cap_line = ind_line = None
    for ax, (title, d) in zip(axes, rows):
        cap_x, cap_y, ind_x, ind_y = split_branches(d)
        cl, = ax.plot(cap_x, cap_y, color="blue", marker="o", label="Capacitive")
        il, = ax.plot(ind_x, ind_y, color="red", marker="s", linestyle="--", label="Inductive")
        cap_line, ind_line = cl, il
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(ylim_from(cap_y, ind_y))
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Power factor [-]")
    axes[-1].set_xlim(0, 1)

    # single shared legend at the top
    fig.legend(handles=[cap_line, ind_line], labels=["Capacitive", "Inductive"],
               loc="upper center", ncol=2, frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 0.965])  # leave room for the legend
    fig.savefig(f"{figures_dir}/{fname}.pdf", dpi=300)
    plt.close(fig)
# ----------------------------------------------------------------------
# Collect + plot each quantity
# ----------------------------------------------------------------------

# Lifetime  (scalar stored on last valid row)
life = _collect(base_dir, {
    "C":  ("df_3_C.parquet",  "Lifetime_C",  _last_valid),
    "L1": ("df_4_L1.parquet", "Lifetime_L1", _last_valid),
    "L2": ("df_5_L2.parquet", "Lifetime_L2", _last_valid),
})
life_min = {k: min(life["C"][k], life["L1"][k], life["L2"][k])
    for k in life["C"]}


plot_quantity_vs_pf_with_min(
    life["C"], life["L1"], life["L2"], life_min, figures_dir,
    "Lifetime [years]", "Fig_lifetime_vs_pf_with_min"
)

# Total losses  (array -> mean)
loss = _collect(base_dir, {
    "C":  ("df_3_C.parquet",  "P_total_C",  _mean),
    "L1": ("df_4_L1.parquet", "P_total_L1", _mean),
    "L2": ("df_5_L2.parquet", "P_total_L2", _mean),
})
plot_quantity_vs_pf(loss["C"], loss["L1"], loss["L2"], figures_dir,
                    "Power loss [W]", "Fig_loss_vs_pf")

# Temperature  (array -> mean, K -> degC)
temp = _collect(base_dir, {
    "C":  ("df_3_C.parquet",  "T_C",            _mean_degC),
    "L1": ("df_4_L1.parquet", "T_inductor_L1",  _mean_degC),
    "L2": ("df_5_L2.parquet", "T_inductor_L2",  _mean_degC),
})
plot_quantity_vs_pf(temp["C"], temp["L1"], temp["L2"], figures_dir,
                    "Temperature [°C]", "Fig_temp_vs_pf")

# RMS currents  (array -> mean)
irms = _collect(base_dir, {
    "C":  ("df_3_C.parquet",  "I_C_RMS",  _mean),
    "L1": ("df_4_L1.parquet", "I_L1_RMS", _mean),
    "L2": ("df_5_L2.parquet", "I_L2_RMS", _mean),
})
plot_quantity_vs_pf(irms["C"], irms["L1"], irms["L2"], figures_dir,"RMS current [A]", "Fig_irms_vs_pf")