import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

base_dir = "pf_sweep_results/Final_final"
figures_dir = "pf_sweep_results/Figures"


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
def _rms(s):  return float(np.sqrt(np.nanmean(np.asarray(s, float) ** 2)))

# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_quantity_vs_pf(C_dict, L1_dict, L2_dict, figures_dir, ylabel, fname, titles=None, y_margin=0.05):
    """
    3-row (L1, C, L2) plot vs power factor. Capacitive (+pf, incl. +0.0) and
    inductive (-pf, incl. -0.0) branches are overlaid against |pf|.

    The capacitive and inductive zeros are GENUINE separate runs (keyed +0.0
    and -0.0), so each branch has its own real endpoint at pf = 0. Only the
    unity point (pf = 1, one run) is shared between the branches.

    A single shared legend is placed at the top of the figure.
    """
    if titles is None:
        titles = ["Inverter-side inductor L1", "Capacitor C", "Grid-side inductor L2"]

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

    rows = [(titles[0], L1_dict), (titles[1], C_dict), (titles[2], L2_dict)]
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 4.8 * 3 * 0.415), sharex=True)

    cap_line = ind_line = None
    for ax, (title, d) in zip(axes, rows):
        cap_x, cap_y, ind_x, ind_y = split_branches(d)
        cl, = ax.plot(cap_x, cap_y, color="blue", marker="o", label="Capacitive")
        il, = ax.plot(ind_x, ind_y, color="red", marker="s", linestyle="--", label="Inductive")
        cap_line, ind_line = cl, il
        ax.set_title(title)
        ax.set_ylim(ylim_from(cap_y, ind_y))
        #ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Power factor [-]")
    axes[-1].set_xlim(0, 1)

    # single shared y-axis label
    fig.supylabel(ylabel)

    # single shared legend at the top
    fig.legend(handles=[cap_line, ind_line], labels=["Capacitive", "Inductive"],
               loc="upper center", ncol=2, frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 0.9525])  # leave room for the legend
    fig.savefig(f"{figures_dir}/{fname}.pdf", dpi=300)
    plt.close(fig)

def plot_irms_vs_pf(C_dict, L1_dict, L2_dict, figures_dir, ylabel="Current [A]",fname="Fig_irms_vs_pf", titles=None, power=None, y_margin=0.05):
    """
    3-row (L1, C, L2) RMS-current plot vs power factor. Capacitive (+pf, incl.
    +0.0) and inductive (-pf, incl. -0.0) branches are overlaid against |pf|.

    `power`, if given, is a list of three (lo, hi) tuples applied per panel in
    order [L1, C, L2]. Use None for any panel to auto-scale it.
    A single shared legend is placed at the top of the figure.
    """
    if titles is None:
        titles = ["Grid-side inductor L2", "Capacitor C", "Inverter-side inductor L1"]

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

    rows = [(titles[0], L2_dict), (titles[1], C_dict), (titles[2], L1_dict)]
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 4.8 * 3 * 0.425), sharex=True)

    cap_line = ind_line = None
    for i, (ax, (title, d)) in enumerate(zip(axes, rows)):
        cap_x, cap_y, ind_x, ind_y = split_branches(d)
        cl, = ax.plot(cap_x, cap_y, color="blue", marker="o", label="Capacitive")
        il, = ax.plot(ind_x, ind_y, color="red", marker="s", linestyle="--", label="Inductive")
        cap_line, ind_line = cl, il
        ax.set_title(title)
        if power is not None and power[i] is not None:
            ax.set_ylim(power[i])
        else:
            ax.set_ylim(ylim_from(cap_y, ind_y))

    axes[-1].set_xlabel("Power factor [-]")
    axes[-1].set_xlim(0, 1)

    fig.supylabel(ylabel)
    fig.legend(handles=[cap_line, ind_line], labels=["Capacitive", "Inductive"],
               loc="upper center", ncol=2, frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{figures_dir}/{fname}.pdf", dpi=300)
    plt.close(fig)

def plot_inductor_irms_vs_pf(C_dict, L1_dict, L2_dict, figures_dir, ylabel="Current [A]",fname="2_Fig_inductor_irms_vs_pf", titles=None,power=None, y_margin=0.05):
    """
    2-row (L2, L1) RMS-current plot vs power factor. Capacitive (+pf, incl.
    +0.0) and inductive (-pf, incl. -0.0) branches are overlaid against |pf|.

    C_dict is accepted for signature compatibility but not plotted, since the
    inductor lifetimes are driven by current alone while the capacitor is
    treated separately with its current and voltage.

    `power`, if given, is a list of two (lo, hi) tuples applied per panel in
    order [L2, L1]. Use None for any panel to auto-scale it.
    A single shared legend is placed at the top of the figure.
    """
    if titles is None:
        titles = ["Grid-side inductor L2", "Inverter-side inductor L1"]

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

    rows = [(titles[0], L2_dict), (titles[1], L1_dict)]
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.8 * 2 * 0.45), sharex=True,
                             layout="constrained")

    cap_line = ind_line = None
    for i, (ax, (title, d)) in enumerate(zip(axes, rows)):
        cap_x, cap_y, ind_x, ind_y = split_branches(d)
        cl, = ax.plot(cap_x, cap_y, color="blue", marker="o", label="Capacitive")
        il, = ax.plot(ind_x, ind_y, color="red", marker="s", linestyle="--", label="Inductive")
        cap_line, ind_line = cl, il
        ax.set_title(title)
        if power is not None and power[i] is not None:
            ax.set_ylim(power[i])
        else:
            ax.set_ylim(ylim_from(cap_y, ind_y))

    axes[-1].set_xlabel("Power factor [-]")
    axes[-1].set_xlim(0, 1)

    fig.supylabel(ylabel)
    fig.legend(handles=[cap_line, ind_line], labels=["Capacitive", "Inductive"],
               loc="outside upper center", ncol=2, frameon=True)

    #fig.tight_layout(rect=[0, 0, 1, 0.925])
    fig.savefig(f"{figures_dir}/{fname}.pdf", dpi=300)
    plt.close(fig)


def plot_capacitor_vrms_irms_vs_pf(C_v_dict, C_i_dict, figures_dir, fname="3_Fig_capacitor_vrms_irms_vs_pf", titles=None, power=None, y_margin=0.05):
    """
    2-row capacitor plot vs power factor, RMS voltage on top and RMS current
    below. Capacitive (+pf, incl. +0.0) and inductive (-pf, incl. -0.0)
    branches are overlaid against |pf|.

    Both quantities are shown because the capacitor lifetime depends on the
    RMS voltage through the voltage-acceleration term and on the current
    through the loss and hence the hotspot temperature.

    `power`, if given, is a list of two (lo, hi) tuples applied per panel in
    order [voltage, current]. Use None for any panel to auto-scale it.
    A single shared legend is placed at the top of the figure.
    """
    if titles is None:
        titles = ["Capacitor C (Voltage)", "Capacitor C (Current)"]

    ylabels = ["Voltage [V]", "Current [A]"]

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

    rows = [(titles[0], C_v_dict), (titles[1], C_i_dict)]
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.8 * 2 * 0.45), sharex=True,
                             layout="constrained")

    cap_line = ind_line = None
    for i, (ax, (title, d)) in enumerate(zip(axes, rows)):
        cap_x, cap_y, ind_x, ind_y = split_branches(d)
        cl, = ax.plot(cap_x, cap_y, color="blue", marker="o",
                      label="Capacitive")
        il, = ax.plot(ind_x, ind_y, color="red", marker="s", linestyle="--",
                      label="Inductive")
        cap_line, ind_line = cl, il
        ax.set_title(title)
        ax.set_ylabel(ylabels[i])
        if power is not None and power[i] is not None:
            ax.set_ylim(power[i])
        else:
            ax.set_ylim(ylim_from(cap_y, ind_y))

    axes[-1].set_xlabel("Power factor [-]")
    axes[-1].set_xlim(0, 1)

    fig.legend(handles=[cap_line, ind_line], labels=["Capacitive", "Inductive"],
               loc="outside upper center", ncol=2, frameon=True)

    #fig.tight_layout(rect=[0, 0, 1, 0.925])
    fig.savefig(f"{figures_dir}/{fname}.pdf", dpi=300)
    plt.close(fig)

def plot_quantity_vs_pf_with_min(C_dict, L1_dict, L2_dict, min_dict, figures_dir,ylabel,fname, titles=None, y_margin=0.05,target=None, target_label="Design target", ylims=None):
    """
    4-row (C, L1, L2, module-min) plot vs power factor. Capacitive (+pf, incl.
    +0.0) and inductive (-pf, incl. -0.0) branches are overlaid against |pf|.

    If `target` is given, a black dotted horizontal line is drawn on every panel
    and the y-limits are widened so it stays visible.

    A single shared legend is placed at the top of the figure.
    """
    if titles is None:
        titles = ["Grid-side inductor L2", "Capacitor C",
                  "Inverter-side inductor L1", "LCL filter"]


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
        if target is not None:
            lo, hi = min(lo, target), max(hi, target)
        pad = (hi - lo) * y_margin if hi > lo else (abs(lo) * y_margin or 1.0)
        return lo - pad, hi + pad

    rows = [(titles[0], L2_dict), (titles[1], C_dict),
            (titles[2], L1_dict), (titles[3], min_dict)]

    fig, axes = plt.subplots(4, 1, figsize=(6.4, 4.8 * 4 * 0.40), sharex=True,
                             layout="constrained")

    cap_line = ind_line = target_line = None
    for i, (ax, (title, d)) in enumerate(zip(axes, rows)):
        cap_x, cap_y, ind_x, ind_y = split_branches(d)
        cap_line, = ax.plot(cap_x, cap_y, color="blue", marker="o", linestyle="-")
        ind_line, = ax.plot(ind_x, ind_y, color="red", marker="s", linestyle="--")

        #if target is not None:
        #    target_line = ax.axhline(target, color="black", linestyle="--", linewidth=1.5)

        ax.set_title(title)
        if ylims is not None and ylims[i] is not None:
            ax.set_ylim(ylims[i])
        else:
            ax.set_ylim(ylim_from(cap_y, ind_y))

    axes[-1].set_xlabel("Power factor [-]")
    axes[-1].set_xlim(0, 1)

    handles = [cap_line, ind_line]
    labels = ["Capacitive", "Inductive"]
    #if target_line is not None:
    #    handles.append(target_line)
    #    labels.append(target_label)

    fig.legend(handles=handles, labels=labels, loc="outside upper center",
               ncol=len(handles), frameon=True,
               handlelength=2, markerscale=1.0, borderpad=0.5,
               columnspacing=1, labelspacing=0.5)

    fig.supylabel(ylabel)

    #fig.tight_layout(rect=[-0.033, 0, 1, 0.955])
    fig.savefig(f"{figures_dir}/{fname}.pdf", dpi=300)
    plt.close(fig)

def plot_quantity_vs_pf_4panel(C_dict, L1_dict, L2_dict, min_dict, figures_dir,ylabel, fname, titles=None, y_margin=0.05):
    """
    4-row (C, L1, L2, LCL) plot vs power factor. Capacitive (+pf, incl. +0.0)
    and inductive (-pf, incl. -0.0) branches are overlaid against |pf|.

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

    fig, axes = plt.subplots(4, 1, figsize=(6.4, 4.8 * 4 * 0.5), sharex=True)

    cap_line = ind_line = None
    for ax, (title, d) in zip(axes, rows):
        cap_x, cap_y, ind_x, ind_y = split_branches(d)
        cap_line, = ax.plot(cap_x, cap_y, color="blue", marker="o", linestyle="-")
        ind_line, = ax.plot(ind_x, ind_y, color="red", marker="s", linestyle="--")

        ax.set_title(title)
        ax.set_ylim(ylim_from(cap_y, ind_y))

    axes[-1].set_xlabel("Power factor [-]")
    axes[-1].set_xlim(0, 1)

    fig.legend(handles=[cap_line, ind_line], labels=["Capacitive", "Inductive"],
               loc="upper center", ncol=2, frameon=True,
               handlelength=2, markerscale=1.0, borderpad=0.5,
               columnspacing=1, labelspacing=0.5)
    fig.supylabel(ylabel)

    fig.tight_layout(rect=[-0.033, 0, 1, 0.95])
    fig.savefig(f"{figures_dir}/{fname}.pdf", dpi=300)
    plt.close(fig)
# ----------------------------------------------------------------------
# Collect + plot each quantity
# ----------------------------------------------------------------------

b10 = _collect(base_dir, {
    "C":   ("df_6_MC.parquet", "B10_C",   _last_valid),
    "L1":  ("df_6_MC.parquet", "B10_L1",  _last_valid),
    "L2":  ("df_6_MC.parquet", "B10_L2",  _last_valid),
    "LCL": ("df_6_MC.parquet", "B10_LCL", _last_valid),
})

plot_quantity_vs_pf_4panel(
    b10["C"], b10["L1"], b10["L2"], b10["LCL"], figures_dir,
    "$B_{10}$ lifetime [years]", "Fig_B10_vs_pf")


# Lifetime  (scalar stored on last valid row)
life = _collect(base_dir, {
    "C":  ("df_3_C.parquet",  "Lifetime_C",  _last_valid),
    "L1": ("df_4_L1.parquet", "Lifetime_L1", _last_valid),
    "L2": ("df_5_L2.parquet", "Lifetime_L2", _last_valid),
})

life["C"] = {k: v*(15/life["C"][1.0]) for k, v in life["C"].items()}
life["L1"] = {k: v*(15/life["L1"][1.0]) for k, v in life["L1"].items()}
life["L2"] = {k: v*(15/life["L2"][1.0]) for k, v in life["L2"].items()}

life_min = {k: min(life["C"][k], life["L1"][k], life["L2"][k]) for k in life["C"]}

# per-component padding [years]
pads = {
    "C":   10,    # capacitor spans ~15-60, needs more headroom
    "L1":  2,
    "L2":  2,    # L2 spans ~15.3-15.6, a small range
    "min": 1,
}

lims = {}
for comp, d in (("C", life["C"]), ("L1", life["L1"]), ("L2", life["L2"]), ("min", life_min)):
    vals = d.values()
    p = pads[comp]
    lims[comp] = (min(vals) - p, max(vals) + p)

plot_quantity_vs_pf_with_min(life["C"], life["L1"], life["L2"], life_min, figures_dir,
                             "Lifetime [years]", "1_Fig_lifetime_vs_pf_with_min",
                             target=15.0, target_label="Design target (15 years)",
                             ylims=[lims["L2"], lims["C"], lims["L1"], lims["min"]])


# Total losses  (array -> mean)
loss = _collect(base_dir, {
    "C":  ("df_3_C.parquet",  "P_total_C",  _mean),
    "L1": ("df_4_L1.parquet", "P_total_L1", _mean),
    "L2": ("df_5_L2.parquet", "P_total_L2", _mean),
})
plot_quantity_vs_pf(loss["C"], loss["L1"], loss["L2"], figures_dir,"Power loss [W]", "Fig_loss_vs_pf")

# Temperature  (array -> mean, K -> degC)
temp = _collect(base_dir, {
    "C":  ("df_3_C.parquet",  "T_C",            _mean_degC),
    "L1": ("df_4_L1.parquet", "T_inductor_L1",  _mean_degC),
    "L2": ("df_5_L2.parquet", "T_inductor_L2",  _mean_degC),
})
plot_quantity_vs_pf(temp["C"], temp["L1"], temp["L2"], figures_dir,"Temperature [°C]", "Fig_temp_vs_pf")

# RMS currents  (array -> mean)
irms = _collect(base_dir, {
    "C":  ("df_3_C.parquet",  "I_C_RMS",  _mean),
    "L1": ("df_4_L1.parquet", "I_L1_RMS", _mean),
    "L2": ("df_5_L2.parquet", "I_L2_RMS", _mean),
})

# per-component current padding [A]
pads = {
    "C":  1.0,    # spans ~38-44, small range
    "L1": 10,    # spans ~815-855
    "L2": 10,    # spans ~832-833.5, very small range
}

lims = {}
for comp in ("C", "L1", "L2"):
    vals = irms[comp].values()
    lo, hi = min(vals), max(vals)
    p = pads[comp]
    lims[comp] = (lo - p, hi + p)

# 3-panel: order in the function is L2, C, L1
plot_irms_vs_pf(irms["C"], irms["L1"], irms["L2"], figures_dir,
                power=[lims["L2"], lims["C"], lims["L1"]])

# 2-panel inductors: order in the function is L2, L1
plot_inductor_irms_vs_pf(irms["C"], irms["L1"], irms["L2"], figures_dir,
                         power=[lims["L2"], lims["L1"]])



# RMS currents  (array -> mean)
vrms = _collect(base_dir, {
    "C":  ("df_3_C.parquet",  "V_C_RMS",  _mean),
    "L1": ("df_4_L1.parquet", "V_L1_RMS", _mean),
    "L2": ("df_5_L2.parquet", "V_L2_RMS", _mean),
})
plot_quantity_vs_pf(vrms["C"], vrms["L1"], vrms["L2"], figures_dir,"Voltage [V]", "Fig_vrms_vs_pf")


pad_i = 5      # current padding [A]
pad_v = 10      # voltage padding [V]

# current limits (existing)
lims = {}
for comp in ("C", "L1", "L2"):
    vals = irms[comp].values()
    lo, hi = min(vals), max(vals)
    lims[comp] = (lo - pad_i, hi + pad_i)

# capacitor voltage limits
v_vals = vrms["C"].values()
v_lo, v_hi = min(v_vals), max(v_vals)
lims_v_C = (v_lo - pad_v, v_hi + pad_v)

# capacitor figure: power = [voltage panel, current panel]
plot_capacitor_vrms_irms_vs_pf(vrms["C"], irms["C"], figures_dir,
                               power=[lims_v_C, lims["C"]])

'''
def plot_two_series_vs_pf(dict_a, dict_b, figures_dir, fname,label_a, label_b, ylabel,y_margin=0.05):
    """
    Single-panel plot of two quantities vs |pf|, each split into
    capacitive (+pf, incl. +0.0) and inductive (-pf, incl. -0.0) branches.
    """
    def split_branches(d):
        cap = {abs(pf): v for pf, v in d.items() if not np.signbit(pf)}
        ind = {abs(pf): v for pf, v in d.items() if np.signbit(pf)}
        if 1.0 in cap and 1.0 not in ind:
            ind[1.0] = cap[1.0]
        if 0.0 in cap and 0.0 not in ind:
            ind[0.0] = cap[0.0]
        cap_x = sorted(cap); ind_x = sorted(ind)
        return cap_x, [cap[x] for x in cap_x], ind_x, [ind[x] for x in ind_x]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    # series A
    cx, cy, ix, iy = split_branches(dict_a)
    ax.plot(cx, cy, color="blue",  marker="o", linestyle="-",  label=f"{label_a} capacitive")
    ax.plot(ix, iy, color="red",   marker="s", linestyle="--", label=f"{label_a} inductive")

    # series B
    cx, cy, ix, iy = split_branches(dict_b)
    ax.plot(cx, cy, color="green",  marker="^", linestyle="-",  label=f"{label_b} capacitive")
    ax.plot(ix, iy, color="orange", marker="v", linestyle="--", label=f"{label_b} inductive")

    allv = np.concatenate([np.asarray(list(dict_a.values()), float),
                           np.asarray(list(dict_b.values()), float)])
    lo, hi = np.nanmin(allv), np.nanmax(allv)
    pad = (hi - lo) * y_margin if hi > lo else (abs(lo) * y_margin or 1.0)
    ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel("Power factor [-]")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(f"{figures_dir}/{fname}.pdf", dpi=300)
    plt.close(fig)


Current = _collect(base_dir, {
    "Ig_ref":  ("df_2_power_flow_inst.parquet",  "Ig_ref",  _rms),
    "I_L2": ("df_2_power_flow_inst.parquet", "I_L2", _rms),
})


plot_two_series_vs_pf(Current["Ig_ref"], Current["I_L2"], figures_dir, "Fig_Ig_ref_I_L2_vs_pf",
                      label_a=r"Ig_ref", label_b=r"$I_{L2}$",
                      ylabel="Current [A]")



RES_PER_CYCLE = 50000   # actual resolution_per_cycle
F = 50
def _split_fund_nonfund(s):
    """
    Return (fundamental_RMS, nonfundamental_RMS) for signal s.

    Fundamental = the single spectral bin at the fundamental (50 Hz).
    Non-fundamental = all remaining content, via Parseval:
        nonfund_RMS = sqrt(total_RMS^2 - fund_RMS^2)
    The integer-cycle window guarantees the fundamental lands exactly on a bin,
    so the single-bin extraction is leakage-free.
    """
    x = np.asarray(s, dtype=float)
    x = x[~np.isnan(x)]
    N = len(x)

    k = round(N / RES_PER_CYCLE)          # fundamental bin index (whole cycles in window)
    n = np.arange(N)

    # single-bin DFT at the fundamental
    basis = np.exp(-1j * 2 * np.pi * k * n / N)
    X = np.dot(x, basis)
    fund_rms = (2.0 * np.abs(X) / N) / np.sqrt(2)

    # total RMS and non-fundamental by Parseval
    total_rms = np.sqrt(np.mean(x ** 2))
    nonfund_rms = np.sqrt(max(total_rms ** 2 - fund_rms ** 2, 0.0))

    return fund_rms, nonfund_rms

# reducers that pick one of the two components
def _fund_rms(s):     return _split_fund_nonfund(s)[0]
def _nonfund_rms(s):  return _split_fund_nonfund(s)[1]

split = _collect(base_dir, {
    "Ig_ref_fund":    ("df_2_power_flow_inst.parquet", "Ig_ref", _fund_rms),
    "Ig_ref_nonfund": ("df_2_power_flow_inst.parquet", "Ig_ref", _nonfund_rms),
    "I_L2_fund":      ("df_2_power_flow_inst.parquet", "I_L2",   _fund_rms),
    "I_L2_nonfund":   ("df_2_power_flow_inst.parquet", "I_L2",   _nonfund_rms),
})

# print table: pf | Ig_ref fund | Ig_ref non-fund | I_L2 fund | I_L2 non-fund
#print(f"{'pf':>7} {'Igref_f1':>13} {'Igref_hrm':>13} {'IL2_f1':>13} {'IL2_hrm':>13}")
#for pf in sorted(split["Ig_ref_fund"], key=lambda p: (np.signbit(p), abs(p))):
#    igf = split["Ig_ref_fund"][pf]
#    igh = split["Ig_ref_nonfund"][pf]
#    ilf = split["I_L2_fund"][pf]
#    ilh = split["I_L2_nonfund"][pf]
#    print(f"{pf:>7.3f} {igf:>13.6f} {igh:>13.6f} {ilf:>13.6f} {ilh:>13.6f}")


def plot_IL2_hrm_vs_pf(IL2_hrm_dict, figures_dir, fname="Fig_IL2_hrm_vs_pf",ylabel=r"$I_{L2}$ harmonic RMS [A]", y_margin=0.05):
    """
    Single-panel plot of the non-fundamental (harmonic/ripple) RMS of the
    grid-side current I_L2 vs |pf|, split into capacitive (+pf, incl. +0.0)
    and inductive (-pf, incl. -0.0) branches.
    """
    def split_branches(d):
        cap = {abs(pf): v for pf, v in d.items() if not np.signbit(pf)}
        ind = {abs(pf): v for pf, v in d.items() if np.signbit(pf)}
        if 1.0 in cap and 1.0 not in ind:
            ind[1.0] = cap[1.0]
        if 0.0 in cap and 0.0 not in ind:
            ind[0.0] = cap[0.0]
        cap_x = sorted(cap); ind_x = sorted(ind)
        return cap_x, [cap[x] for x in cap_x], ind_x, [ind[x] for x in ind_x]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    cap_x, cap_y, ind_x, ind_y = split_branches(IL2_hrm_dict)
    ax.plot(cap_x, cap_y, color="blue", marker="o", linestyle="-",  label="Capacitive")
    ax.plot(ind_x, ind_y, color="red",  marker="s", linestyle="--", label="Inductive")

    allv = np.asarray(list(IL2_hrm_dict.values()), float)
    lo, hi = np.nanmin(allv), np.nanmax(allv)
    pad = (hi - lo) * y_margin if hi > lo else (abs(lo) * y_margin or 1.0)
    ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel("Power factor [-]")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(f"{figures_dir}/{fname}.pdf", dpi=300)
    plt.close(fig)

plot_IL2_hrm_vs_pf(split["I_L2_nonfund"], figures_dir, "Fig_IL2_hrm_vs_pf")


def build_vs_correction(split):
    """
    Build the Vs_ref fundamental-magnitude correction factor for each
    operating point.

        correction(pf) = Ig_ref_fund(pf) / IL2_fund(pf)

    Scaling Vs_ref by this factor pulls the delivered I_L2 fundamental onto
    the commanded grid-current fundamental Ig_ref. correction > 1 means the
    delivered fundamental fell short and Vs_ref must be increased; < 1 means
    it overshot.

    Keys are the signed pf used throughout the sweep (positive = capacitive,
    negative = inductive; +0.0 / -0.0 distinguish the two zeros).

    Parameters
    ----------
    split : dict
        Output of _collect containing "Ig_ref_fund" and "I_L2_fund",
        each mapping signed-pf -> fundamental RMS.

    Returns
    -------
    dict
        {pf_signed: correction_factor}.
    """
    ig_fund = split["Ig_ref_fund"]
    il_fund = split["I_L2_fund"]

    correction = {}
    for pf in ig_fund:
        delivered = il_fund[pf]
        target = ig_fund[pf]
        correction[pf] = target / delivered if delivered != 0 else 1.0

    return correction

VS_CORRECTION = build_vs_correction(split)


# RMS of instantaneous waveforms from df_2 (one cycle -> RMS)
THD = _collect(base_dir, {
    "THD": ("df_2_power_flow_inst.parquet", "THD_percent_I_L2", _last_valid),
})

def plot_thd_vs_pf(THD_dict, figures_dir, fname="Fig_thd_vs_pf",ylabel="THD [%]", y_margin=0.05):
    """
    Single-panel plot of grid-side current THD vs |pf|, split into
    capacitive (+pf, incl. +0.0) and inductive (-pf, incl. -0.0) branches.
    """
    def split_branches(d):
        cap = {abs(pf): v for pf, v in d.items() if not np.signbit(pf)}
        ind = {abs(pf): v for pf, v in d.items() if np.signbit(pf)}
        if 1.0 in cap and 1.0 not in ind:
            ind[1.0] = cap[1.0]
        if 0.0 in cap and 0.0 not in ind:
            ind[0.0] = cap[0.0]
        cap_x = sorted(cap); ind_x = sorted(ind)
        return cap_x, [cap[x] for x in cap_x], ind_x, [ind[x] for x in ind_x]

    fig, ax = plt.subplots(figsize=(6.4, 4.8 * 0.75))

    cap_x, cap_y, ind_x, ind_y = split_branches(THD_dict)
    ax.plot(cap_x, cap_y, color="blue", marker="o", linestyle="-",  label="Capacitive")
    ax.plot(ind_x, ind_y, color="red",  marker="s", linestyle="--", label="Inductive")

    allv = np.asarray(list(THD_dict.values()), float)
    lo, hi = np.nanmin(allv), np.nanmax(allv)
    pad = (hi - lo) * y_margin if hi > lo else (abs(lo) * y_margin or 1.0)
    ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel("Power factor [-]")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(f"{figures_dir}/{fname}.pdf", dpi=300)
    plt.close(fig)

plot_thd_vs_pf(THD["THD"], figures_dir, "3_Fig_thd_vs_pf")


chk = _collect(base_dir, {
    "IL2_f1": ("df_2_power_flow_inst.parquet", "I_L2",   _fund_rms),
    "Ig_f1":  ("df_2_power_flow_inst.parquet", "Ig_ref", _fund_rms),
})
#for pf in sorted(chk["IL2_f1"], key=lambda p:(np.signbit(p),abs(p))):
    #print(f"{pf:7.3f}  IL2_f1={chk['IL2_f1'][pf]:10.4f}  Ig_f1={chk['Ig_f1'][pf]:10.4f}  err={chk['IL2_f1'][pf]-chk['Ig_f1'][pf]:+.4f}")
'''