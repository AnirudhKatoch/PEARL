import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from scipy.interpolate import PchipInterpolator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # project/
sys.path.insert(0, str(ROOT))

from Input_parameters import Input_parameters_class
from Calculation_functions import Calculation_functions_class

params = Input_parameters_class()
Calculation_functions = Calculation_functions_class()

Vdc_rated = params.Vdc_rated; Vo_rated = params.Vo_rated; inverter_phases = params.inverter_phases; M_rated = params.M_rated; single_phase_inverter_topology = params.single_phase_inverter_topology; waveform_voltage_definition = params.waveform_voltage_definition; modulation_scheme = params.modulation_scheme; f = params.f; fsw = params.fsw; T = params.T; Tsw = params.Tsw; omega = params.omega
Profile_size = params.Profile_size; Vdc_RMS = params.Vdc_RMS; M = params.M; Vo = params.Vo; Vg_RMS = params.Vg_RMS; S_RMS = params.S_RMS; pf = params.pf; P_RMS = params.P_RMS; Q_RMS = params.Q_RMS; Ig_RMS = params.Ig_RMS
T_amb = params.T_amb; heat_transfer_coefficient = params.heat_transfer_coefficient
resolution_per_cycle = params.resolution_per_cycle; dt = params.dt; samples_per_switching_period = params.samples_per_switching_period; Minimum_required_samples_per_switching_period = params.Minimum_required_samples_per_switching_period; seconds_per_sample = params.seconds_per_sample; h_max = params.h_max
L1_specs = params.L1_specs; C_specs = params.C_specs; L2_specs = params.L2_specs
Vg_ll_RMS = params.Vg_ll_RMS; S_rated = params.S_rated; I_rated_RMS = params.I_rated_RMS; I_rated_peak = params.I_rated_peak; current_ripple_limit = params.current_ripple_limit; delta = params.delta; omega_sw = params.omega_sw


plt.rcParams.update({"font.size": 15, "font.family": "Times New Roman", "axes.labelsize": 15, "axes.titlesize": 15,"xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15})

Without_drift_dir = ROOT / "OAPE_results" / "Results" / "Without_drift"
With_drift_dir = ROOT / "OAPE_results" / "Results" / "With_drift"
figures_dir = ROOT / "OAPE_results" / "Figures"

def consolidate(base_dir):

    n_years = len([d for d in glob.glob(os.path.join(base_dir, "Simulation_*"))
                   if os.path.isdir(d)])

    years = []
    V_C_RMS_list = []
    I_C_RMS_list = []
    P_total_C_list = []
    T_C_list = []
    D_cycle_C_list = []
    I_C_fundamental_list = []
    I_C_total_list = []
    V_L1_RMS_list = []
    I_L1_RMS_list = []
    P_total_L1_list = []
    T_L1_list = []
    D_cycle_L1_list = []
    I_L1_fundamental_list = []
    I_L1_total_list = []
    V_L2_RMS_list = []
    I_L2_RMS_list = []
    P_total_L2_list = []
    T_L2_list = []
    D_cycle_L2_list = []
    I_L2_fundamental_list = []
    I_L2_total_list = []
    THD_list = []
    C_new_list = []

    for year in range(1, n_years + 1):

        path = f"{base_dir}/Simulation_{year}/Dataframes"

        df_C = pd.read_parquet(f"{path}/df_3_C.parquet")
        df_L1 = pd.read_parquet(f"{path}/df_4_L1.parquet")
        df_L2 = pd.read_parquet(f"{path}/df_5_L2.parquet")
        df_H = pd.read_parquet(f"{path}/df_7_harmonics.parquet")
        df_2_power_flow_inst = pd.read_parquet(f"{path}/df_2_power_flow_inst.parquet")

        years.append(year)

        V_C_RMS_list.append(np.mean(df_C["V_C_RMS"]))
        I_C_RMS_list.append(np.mean(df_C["I_C_RMS"]))
        P_total_C_list.append(np.mean(df_C["P_total_C"]))
        T_C_list.append(np.mean(df_C["T_C"]))
        D_cycle_C_list.append(df_C["D_cycle_C"].dropna().iloc[-1])
        C_new_list.append(df_C["C_new"].dropna().iloc[-1])

        V_L1_RMS_list.append(np.mean(df_L1["V_L1_RMS"]))
        I_L1_RMS_list.append(np.mean(df_L1["I_L1_RMS"]))
        P_total_L1_list.append(np.mean(df_L1["P_total_L1"]))
        T_L1_list.append(np.mean(df_L1["T_inductor_L1"]))
        D_cycle_L1_list.append(df_L1["D_cycle_L1"].dropna().iloc[-1])

        V_L2_RMS_list.append(np.mean(df_L2["V_L2_RMS"]))
        I_L2_RMS_list.append(np.mean(df_L2["I_L2_RMS"]))
        P_total_L2_list.append(np.mean(df_L2["P_total_L2"]))
        T_L2_list.append(np.mean(df_L2["T_inductor_L2"]))
        D_cycle_L2_list.append(df_L2["D_cycle_L2"].dropna().iloc[-1])

        THD_list.append(df_2_power_flow_inst["THD_percent_I_L2"].dropna().iloc[-1])

        # --- harmonics: fundamental and total per component ---

        for component, fundamental_list, total_list in [("C", I_C_fundamental_list, I_C_total_list),
                                                        ("L1", I_L1_fundamental_list, I_L1_total_list),
                                                        ("L2", I_L2_fundamental_list, I_L2_total_list)]:

            df_component = df_H[df_H["component"] == component]

            fundamental = df_component[df_component["order"] == 1]["I_rms"].to_numpy()
            total = np.sqrt(df_component.groupby("second")["I_rms"].apply(lambda x: np.sum(x ** 2)).to_numpy())

            fundamental_list.append(np.mean(fundamental))
            total_list.append(np.mean(total))

    df = pd.DataFrame()
    df["year"] = years
    df["V_C_RMS"] = V_C_RMS_list
    df["I_C_RMS"] = I_C_RMS_list
    df["P_total_C"] = P_total_C_list
    df["T_C"] = T_C_list
    df["D_cycle_C"] = D_cycle_C_list
    df["D_cum_C"] = np.cumsum(D_cycle_C_list)
    df["I_C_fundamental"] = I_C_fundamental_list
    df["I_C_total"] = I_C_total_list
    df["V_L1_RMS"] = V_L1_RMS_list
    df["I_L1_RMS"] = I_L1_RMS_list
    df["P_total_L1"] = P_total_L1_list
    df["T_L1"] = T_L1_list
    df["D_cycle_L1"] = D_cycle_L1_list
    df["D_cum_L1"] = np.cumsum(D_cycle_L1_list)
    df["I_L1_fundamental"] = I_L1_fundamental_list
    df["I_L1_total"] = I_L1_total_list
    df["V_L2_RMS"] = V_L2_RMS_list
    df["I_L2_RMS"] = I_L2_RMS_list
    df["P_total_L2"] = P_total_L2_list
    df["T_L2"] = T_L2_list
    df["D_cycle_L2"] = D_cycle_L2_list
    df["D_cum_L2"] = np.cumsum(D_cycle_L2_list)
    df["I_L2_fundamental"] = I_L2_fundamental_list
    df["I_L2_total"] = I_L2_total_list
    df["THD"] = THD_list
    df["C_new"] = C_new_list

    return df

#df_without = consolidate(Without_drift_dir)
#df_with = consolidate(With_drift_dir)

#df_without.to_parquet("Results/summary_without_drift.parquet")
#df_with.to_parquet("Results/summary_with_drift.parquet")

df_without = pd.read_parquet("Results/summary_without_drift.parquet")
df_with = pd.read_parquet("Results/summary_with_drift.parquet")


def death_year(D_cum):
    D_cum = np.asarray(D_cum)
    i = np.argmax(D_cum >= 1.0)
    if D_cum[i] < 1.0:
        return np.nan
    D_prev = D_cum[i-1] if i > 0 else 0.0
    return i + (1.0 - D_prev) / (D_cum[i] - D_prev)

def normalise(D_cycle_list):
    D_cycle = np.asarray(D_cycle_list, dtype=float)
    D_end = np.cumsum(D_cycle)
    n = np.argmax(D_end >= 1.0)

    D_cycle = D_cycle[:n+1].copy()
    D_start = np.concatenate(([0.0], np.cumsum(D_cycle)[:-1]))
    D_cycle[n] = 1.0 - D_start[n]            # truncate the last year
    D_end = np.cumsum(D_cycle)               # now ends exactly at 1.0

    duration = np.ones(n+1)
    duration[n] = D_cycle[n] / np.asarray(D_cycle_list)[n]   # < 1 yr
    t_end = np.cumsum(duration)
    t_start = np.concatenate(([0.0], t_end[:-1]))

    return D_start, D_end, t_start, t_end


def auto_band(wo_s, w_s, band_variable=0.1):
    """
    Choose a single y-band covering both scenarios, padded by band_variable
    times the data range on each side. The lower edge is clamped at zero.
    """
    lo = min(np.min(wo_s), np.min(w_s))
    hi = max(np.max(wo_s), np.max(w_s))

    pad = (hi - lo) * band_variable

    lo = lo - pad
    hi = hi + pad

    if lo < 0:
        lo = 0.0

    return ((lo, hi),)


def auto_band_split(wo_a_s, w_a_s, wo_b_s, w_b_s, band_variable=0.1):
    """
    Two y-bands, one for each pair of series, ordered low band first.
    Each band is padded by band_variable times its own data range.
    """
    def one(wo_s, w_s):
        lo = min(np.min(wo_s), np.min(w_s))
        hi = max(np.max(wo_s), np.max(w_s))
        pad = (hi - lo) * band_variable
        lo = lo - pad
        hi = hi + pad
        if lo < 0:
            lo = 0.0
        return (lo, hi)

    band_a = one(wo_a_s, w_a_s)
    band_b = one(wo_b_s, w_b_s)

    if band_a[0] <= band_b[0]:
        return (band_a, band_b)
    return (band_b, band_a)

######################################
# Current Plot
######################################

def plot_currents_all_components(df_without, df_with, figures_dir, filename, bands_L1=None, bands_C=None, bands_L2=None, band_variable=0.1):

    # ----------------------------------------#
    # Common x-axis
    # ----------------------------------------#

    Ds_wo, De_wo, ts_wo, te_wo = normalise(df_without["D_cycle_C"].values)
    Ds_w, De_w, ts_w, te_w = normalise(df_with["D_cycle_C"].values)

    Dm_wo = 0.5 * (Ds_wo + De_wo)
    Dm_w = 0.5 * (Ds_w + De_w)

    D_smooth = np.linspace(0.0, 1.0, 500)
    C_smooth = 100.0 - 20.0 * D_smooth

    def smooth(D_mid, y):
        return PchipInterpolator(D_mid, y, extrapolate=True)(D_smooth)

    def align(wo_s, w_s):
        offset = wo_s[0] - w_s[0]
        return w_s + offset

    # ----------------------------------------#
    # L1
    # ----------------------------------------#

    fundamental_wo_L1 = df_without["I_L1_fundamental"].values[:len(Ds_wo)]
    total_wo_L1 = df_without["I_L1_total"].values[:len(Ds_wo)]
    fundamental_w_L1 = df_with["I_L1_fundamental"].values[:len(Ds_w)]
    total_w_L1 = df_with["I_L1_total"].values[:len(Ds_w)]

    harmonics_wo_L1 = np.sqrt(total_wo_L1 ** 2 - fundamental_wo_L1 ** 2)
    harmonics_w_L1 = np.sqrt(total_w_L1 ** 2 - fundamental_w_L1 ** 2)

    total_wo_L1_s = smooth(Dm_wo, total_wo_L1)
    total_w_L1_s = smooth(Dm_w, total_w_L1)
    harmonics_wo_L1_s = smooth(Dm_wo, harmonics_wo_L1)
    harmonics_w_L1_s = smooth(Dm_w, harmonics_w_L1)
    fundamental_wo_L1_s = smooth(Dm_wo, fundamental_wo_L1)
    fundamental_w_L1_s = smooth(Dm_w, fundamental_w_L1)

    total_w_L1_s = align(total_wo_L1_s, smooth(Dm_w, total_w_L1))
    harmonics_w_L1_s = align(harmonics_wo_L1_s, smooth(Dm_w, harmonics_w_L1))
    fundamental_w_L1_s = align(fundamental_wo_L1_s, smooth(Dm_w, fundamental_w_L1))


    # ----------------------------------------#
    # C
    # ----------------------------------------#

    fundamental_wo_C = df_without["I_C_fundamental"].values[:len(Ds_wo)]
    total_wo_C = df_without["I_C_total"].values[:len(Ds_wo)]
    fundamental_w_C = df_with["I_C_fundamental"].values[:len(Ds_w)]
    total_w_C = df_with["I_C_total"].values[:len(Ds_w)]

    harmonics_wo_C = np.sqrt(total_wo_C ** 2 - fundamental_wo_C ** 2)
    harmonics_w_C = np.sqrt(total_w_C ** 2 - fundamental_w_C ** 2)

    total_wo_C_s = smooth(Dm_wo, total_wo_C)
    total_w_C_s = smooth(Dm_w, total_w_C)
    harmonics_wo_C_s = smooth(Dm_wo, harmonics_wo_C)
    harmonics_w_C_s = smooth(Dm_w, harmonics_w_C)
    fundamental_wo_C_s = smooth(Dm_wo, fundamental_wo_C)
    fundamental_w_C_s = smooth(Dm_w, fundamental_w_C)

    total_w_C_s = align(total_wo_C_s, smooth(Dm_w, total_w_C))
    harmonics_w_C_s = align(harmonics_wo_C_s, smooth(Dm_w, harmonics_w_C))
    fundamental_w_C_s = align(fundamental_wo_C_s, smooth(Dm_w, fundamental_w_C))

    # ----------------------------------------#
    # L2
    # ----------------------------------------#

    fundamental_wo_L2 = df_without["I_L2_fundamental"].values[:len(Ds_wo)]
    total_wo_L2 = df_without["I_L2_total"].values[:len(Ds_wo)]
    fundamental_w_L2 = df_with["I_L2_fundamental"].values[:len(Ds_w)]
    total_w_L2 = df_with["I_L2_total"].values[:len(Ds_w)]

    harmonics_wo_L2 = np.sqrt(total_wo_L2 ** 2 - fundamental_wo_L2 ** 2)
    harmonics_w_L2 = np.sqrt(total_w_L2 ** 2 - fundamental_w_L2 ** 2)

    total_wo_L2_s = smooth(Dm_wo, total_wo_L2)
    total_w_L2_s = smooth(Dm_w, total_w_L2)
    harmonics_wo_L2_s = smooth(Dm_wo, harmonics_wo_L2)
    harmonics_w_L2_s = smooth(Dm_w, harmonics_w_L2)
    fundamental_wo_L2_s = smooth(Dm_wo, fundamental_wo_L2)
    fundamental_w_L2_s = smooth(Dm_w, fundamental_w_L2)

    total_w_L2_s = align(total_wo_L2_s, smooth(Dm_w, total_w_L2))
    harmonics_w_L2_s = align(harmonics_wo_L2_s, smooth(Dm_w, harmonics_w_L2))
    fundamental_w_L2_s = align(fundamental_wo_L2_s, smooth(Dm_w, fundamental_w_L2))

    if bands_L1 is None:
        bands_L1 = auto_band_split(total_wo_L1_s, total_w_L1_s,
                                   fundamental_wo_L1_s, fundamental_w_L1_s, band_variable)
    if bands_C is None:
        bands_C = auto_band_split(total_wo_C_s, total_w_C_s,
                                  fundamental_wo_C_s, fundamental_w_C_s, band_variable)
    if bands_L2 is None:
        bands_L2 = auto_band_split(total_wo_L2_s, total_w_L2_s,
                                   fundamental_wo_L2_s, fundamental_w_L2_s, band_variable)

    # ----------------------------------------#
    # Figure
    # ----------------------------------------#

    n_rows = len(bands_L1) + len(bands_C) + len(bands_L2)

    fig = plt.figure(figsize=(6.4, 4.8 * 3*0.75))
    gs = fig.add_gridspec(3, 1, hspace=0.125)

    def make_axes(gs_cell, bands, share_ax):
        heights = [b[1] - b[0] for b in bands][::-1]
        sub = gs_cell.subgridspec(len(bands), 1, height_ratios=heights, hspace=0.1)
        axes = []
        for row in range(len(bands)):
            ax = fig.add_subplot(sub[row], sharex=share_ax) if share_ax is not None else fig.add_subplot(sub[row])
            axes.append(ax)
            if share_ax is None:
                share_ax = ax
        return axes, share_ax

    axes_L1, share_ax = make_axes(gs[0], bands_L1, None)
    axes_C, share_ax = make_axes(gs[1], bands_C, share_ax)
    axes_L2, share_ax = make_axes(gs[2], bands_L2, share_ax)

    def style(axes, bands):
        for row in range(len(bands)):
            ax = axes[row]
            lo, hi = bands[len(bands) - 1 - row]
            ax.set_ylim(lo, hi)
            ax.ticklabel_format(axis="y", useOffset=False, style="plain")
            if row > 0:
                ax.spines["top"].set_visible(False)
            if row < len(bands) - 1:
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(bottom=False)

    # ----- L1 panel -----
    for ax_L1 in axes_L1:
        ax_L1.plot(C_smooth , total_wo_L1_s, color="red", linestyle="-", linewidth=4.0)
        ax_L1.plot(C_smooth , total_w_L1_s, color="blue", linestyle="-", linewidth=2.0)
        #ax_L1.plot(C_smooth , harmonics_wo_L1_s, color="red", linestyle="--", linewidth=4.0)
        #ax_L1.plot(C_smooth , harmonics_w_L1_s, color="blue", linestyle="--", linewidth=2.0)
        ax_L1.plot(C_smooth , fundamental_wo_L1_s, color="red", linestyle="--", linewidth=4.0)
        ax_L1.plot(C_smooth , fundamental_w_L1_s, color="blue", linestyle="--", linewidth=2.0)
    style(axes_L1, bands_L1)
    axes_L1[0].set_title("Inverter-side inductor L1")

    # ----- C panel -----
    for ax_C in axes_C:
        ax_C.plot(C_smooth , total_wo_C_s, color="red", linestyle="-", linewidth=4.0)
        ax_C.plot(C_smooth , total_w_C_s, color="blue", linestyle="-", linewidth=2.0)
        #ax_C.plot(C_smooth , harmonics_wo_C_s, color="red", linestyle="--", linewidth=4.0)
        #ax_C.plot(C_smooth , harmonics_w_C_s, color="blue", linestyle="--", linewidth=2.0)
        ax_C.plot(C_smooth , fundamental_wo_C_s, color="red", linestyle="--", linewidth=4.0)
        ax_C.plot(C_smooth , fundamental_w_C_s, color="blue", linestyle="--", linewidth=2.0)
    style(axes_C, bands_C)
    axes_C[0].set_title("Filter capacitor C")

    # ----- L2 panel -----
    for ax_L2 in axes_L2:
        ax_L2.plot(C_smooth , total_wo_L2_s, color="red", linestyle="-", linewidth=4.0)
        ax_L2.plot(C_smooth , total_w_L2_s, color="blue", linestyle="-", linewidth=2.0)
        #ax_L2.plot(C_smooth , harmonics_wo_L2_s, color="red", linestyle="--", linewidth=4.0)
        #ax_L2.plot(C_smooth , harmonics_w_L2_s, color="blue", linestyle="--", linewidth=2.0)
        ax_L2.plot(C_smooth , fundamental_wo_L2_s, color="red", linestyle="--", linewidth=4.0)
        ax_L2.plot(C_smooth , fundamental_w_L2_s, color="blue", linestyle="--", linewidth=2.0)
    style(axes_L2, bands_L2)
    axes_L2[0].set_title("Grid-side inductor L2")

    # hide x tick labels on every row except the very last
    all_axes = axes_L1 + axes_C + axes_L2
    for ax in all_axes[:-1]:
        ax.tick_params(labelbottom=False)

    all_axes[-1].set_xlim(100, 80)
    all_axes[-1].set_xlabel("Capacitance relative to initial value [%]")
    fig.supylabel("Current [A]")

    handles = [Patch(facecolor="red", label="Capacitance fixed"),
               Line2D([], [], color="black", linestyle="-", label="Total"),
               Patch(facecolor="blue", label="Capacitance degrading"),
               Line2D([], [], color="black", linestyle="--", label="Fundamental")]

    axes_L1[0].legend(handles=handles, ncol=2, loc="lower center",
                      bbox_to_anchor=(0.5, 1.15), frameon=True,
                      handlelength=2, markerscale=1.0, borderpad=0.5,
                      columnspacing=1, labelspacing=0.25)

    fig.savefig(f"{figures_dir}/{filename}", dpi=600, bbox_inches="tight")
    plt.close(fig)


######################################
# Harmonics Plot
######################################

def plot_df_7_harmonics_stacked(df_7_drift, df_7_nominal, figures_dir, filename, bands,second=-1):

    """
    Change in harmonic content between the nominal and drifted capacitance
    cases, plotted as three stacked rows, one per component.

    Each row has a broken x axis showing only the three bands that carry
    content. Red means the harmonic grew with drift, blue means it fell.

    Saved as Fig_29_harmonics_stacked.png
    """

    # ------------------------------------------------------------------
    # figure layout
    # ------------------------------------------------------------------
    widths = [b[1] - b[0] for b in bands]

    fig, axes = plt.subplots(3, len(bands), figsize=(6.4, 4.8 * 3*0.75), gridspec_kw={"width_ratios": widths, "wspace": 0.1, "hspace": 0.30})

    # ------------------------------------------------------------------
    # one row per component
    # ------------------------------------------------------------------
    components = ["L1", "C", "L2"]
    titles = ["Inverter-side inductor L1",
              "Filter capacitor C",
              "Grid-side inductor L2"]

    sec = df_7_drift["second"].max() if second == -1 else second

    for row in range(3):

        comp = components[row]

        # pull the two spectra for this component
        d = df_7_drift[(df_7_drift["second"] == sec) & (df_7_drift["component"] == comp)].sort_values("order")
        n = df_7_nominal[(df_7_nominal["second"] == sec) & (df_7_nominal["component"] == comp)].sort_values("order")

        orders = d["order"].to_numpy()
        diff = d["I_rms"].to_numpy() - n["I_rms"].to_numpy()
        colors = np.where(diff >= 0, "red", "blue")

        # draw the same data in each band, then zoom
        for col in range(len(bands)):

            ax = axes[row, col]
            lo, hi = bands[col]

            ax.bar(orders, diff, width=1.0, color=colors, edgecolor=colors)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlim(lo, hi)
            ax.grid(True, alpha=0.3)

            # share the y scale across the row
            if col > 0:
                #ax.sharey(axes[row, 0])
                ax.tick_params(labelleft=False, left=False)
                ax.spines["left"].set_visible(False)
            if col < len(bands) - 1:
                ax.spines["right"].set_visible(False)

        # row title, centred over the whole row
        axes[row, 1].set_title(titles[row])

    # ------------------------------------------------------------------
    # shared labels and legend
    # ------------------------------------------------------------------
    fig.supxlabel("Harmonic order [-]", y=0.055)
    fig.supylabel("Change in current with 20% capacitance loss [A]", x=0.0)

    handles = [Patch(color="red", label="Increased with 20% capacitance loss"),
               Patch(color="blue", label="Decreased with 20% capacitance loss")]

    fig.legend(handles=handles, ncol=1, loc="lower center",
                      bbox_to_anchor=(0.5, 0.9), frameon=True,
                      handlelength=2, markerscale=1.0, borderpad=0.5,
                      columnspacing=1, labelspacing=0.25)

    fig.savefig(f"{figures_dir}/{filename}.png",dpi=600, bbox_inches="tight")
    plt.close(fig)

df_7_drift   = pd.read_parquet("Results/Harmonics/C_0.8/Dataframes/df_7_harmonics.parquet")
df_7_nominal = pd.read_parquet("Results/Harmonics/C_1.0/Dataframes/df_7_harmonics.parquet")

######################################
# THD Plot
######################################

def plot_THD(df_without, df_with, figures_dir, filename, bands=None,band_variable=0.1):

    Ds_wo, De_wo, ts_wo, te_wo = normalise(df_without["D_cycle_C"].values)
    Ds_w, De_w, ts_w, te_w = normalise(df_with["D_cycle_C"].values)

    Dm_wo = 0.5 * (Ds_wo + De_wo)
    Dm_w = 0.5 * (Ds_w + De_w)

    D_smooth = np.linspace(0.0, 1.0, 500)
    C_smooth = 100 - 20 * D_smooth

    def smooth(D_mid, y):
        return PchipInterpolator(D_mid, y, extrapolate=True)(D_smooth)

    def align(wo_s, w_s):
        offset = wo_s[0] - w_s[0]
        return w_s + offset

    total_wo = df_without["THD"].values[:len(Ds_wo)]
    total_w = df_with["THD"].values[:len(Ds_w)]

    total_wo_s = smooth(Dm_wo, total_wo)
    total_w_s = align(total_wo_s, smooth(Dm_w, total_w))

    if bands is None:
        bands = auto_band(total_wo_s, total_w_s, band_variable)

    heights = [b[1] - b[0] for b in bands][::-1]

    fig, axes = plt.subplots(len(bands), 1, sharex=True, figsize=(6.4, 4.8*0.75),gridspec_kw={"height_ratios": heights, "hspace": 0.1},squeeze=False)
    axes = [a[0] for a in axes]

    for row in range(len(bands)):

        ax = axes[row]
        lo, hi = bands[len(bands) - 1 - row]

        ax.plot(C_smooth, total_wo_s, color="red", linestyle="-", linewidth=4.0)
        ax.plot(C_smooth, total_w_s, color="blue", linestyle="-", linewidth=2.0)
        ax.set_ylim(lo, hi)
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")

        if row > 0:
            ax.spines["top"].set_visible(False)
        if row < len(bands) - 1:
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(bottom=False)

    axes[-1].set_xlim(100, 80)
    axes[-1].set_xlabel("Capacitance relative to initial value [%]")
    fig.supylabel("THD [%]", x=0.02)

    handles = [Patch(facecolor="red", label="Capacitance fixed"),
               Patch(facecolor="blue", label="Capacitance degrading")]

    axes[0].legend(handles=handles, ncol=2, loc="lower center",
                   bbox_to_anchor=(0.5, 1.02), frameon=True,
                   handlelength=2, markerscale=1.0, borderpad=0.5,
                   columnspacing=1, labelspacing=0.25)

    fig.savefig(f"{figures_dir}/{filename}", dpi=600, bbox_inches="tight")
    plt.close(fig)


######################################
# Voltage Plot
######################################

def plot_voltage_all_components(df_without, df_with, figures_dir, filename, bands_L1=None, bands_C=None, bands_L2=None,band_variable=0.1):
    # ----------------------------------------#
    # Common x-axis
    # ----------------------------------------#

    Ds_wo, De_wo, ts_wo, te_wo = normalise(df_without["D_cycle_C"].values)
    Ds_w, De_w, ts_w, te_w = normalise(df_with["D_cycle_C"].values)

    Dm_wo = 0.5 * (Ds_wo + De_wo)
    Dm_w = 0.5 * (Ds_w + De_w)

    D_smooth = np.linspace(0.0, 1.0, 500)
    C_smooth = 100.0 - 20.0 * D_smooth

    def smooth(D_mid, y):
        return PchipInterpolator(D_mid, y, extrapolate=True)(D_smooth)

    def align(wo_s, w_s):
        offset = wo_s[0] - w_s[0]
        return w_s + offset

    # ----------------------------------------#
    # L1
    # ----------------------------------------#

    total_wo_L1 = df_without["V_L1_RMS"].values[:len(Ds_wo)]
    total_w_L1 = df_with["V_L1_RMS"].values[:len(Ds_w)]

    total_wo_L1_s = smooth(Dm_wo, total_wo_L1)
    total_w_L1_s = align(total_wo_L1_s, smooth(Dm_w, total_w_L1))

    # ----------------------------------------#
    # C
    # ----------------------------------------#

    total_wo_C = df_without["V_C_RMS"].values[:len(Ds_wo)]
    total_w_C = df_with["V_C_RMS"].values[:len(Ds_w)]

    total_wo_C_s = smooth(Dm_wo, total_wo_C)
    total_w_C_s = align(total_wo_C_s, smooth(Dm_w, total_w_C))

    # ----------------------------------------#
    # L2
    # ----------------------------------------#

    total_wo_L2 = df_without["V_L2_RMS"].values[:len(Ds_wo)]
    total_w_L2 = df_with["V_L2_RMS"].values[:len(Ds_w)]

    total_wo_L2_s = smooth(Dm_wo, total_wo_L2)
    total_w_L2_s = align(total_wo_L2_s, smooth(Dm_w, total_w_L2))


    if bands_L1 is None:
        bands_L1 = auto_band(total_wo_L1_s, total_w_L1_s, band_variable)
    if bands_C is None:
        bands_C = auto_band(total_wo_C_s, total_w_C_s, band_variable)
    if bands_L2 is None:
        bands_L2 = auto_band(total_wo_L2_s, total_w_L2_s, band_variable)

    # ----------------------------------------#
    # Figure
    # ----------------------------------------#

    fig = plt.figure(figsize=(6.4, 4.8 * 3 * 0.75))
    gs = fig.add_gridspec(3, 1, hspace=0.125)

    def make_axes(gs_cell, bands, share_ax):
        heights = [b[1] - b[0] for b in bands][::-1]
        sub = gs_cell.subgridspec(len(bands), 1, height_ratios=heights, hspace=0.1)
        axes = []
        for row in range(len(bands)):
            ax = fig.add_subplot(sub[row], sharex=share_ax) if share_ax is not None else fig.add_subplot(sub[row])
            axes.append(ax)
            if share_ax is None:
                share_ax = ax
        return axes, share_ax

    axes_L1, share_ax = make_axes(gs[0], bands_L1, None)
    axes_C, share_ax = make_axes(gs[1], bands_C, share_ax)
    axes_L2, share_ax = make_axes(gs[2], bands_L2, share_ax)

    def style(axes, bands):
        for row in range(len(bands)):
            ax = axes[row]
            lo, hi = bands[len(bands) - 1 - row]
            ax.set_ylim(lo, hi)
            ax.ticklabel_format(axis="y", useOffset=False, style="plain")
            if row > 0:
                ax.spines["top"].set_visible(False)
            if row < len(bands) - 1:
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(bottom=False)

    # ----- L1 panel -----
    for ax_L1 in axes_L1:
        ax_L1.plot(C_smooth , total_wo_L1_s, color="red", linestyle="-", linewidth=4.0)
        ax_L1.plot(C_smooth , total_w_L1_s, color="blue", linestyle="-", linewidth=2.0)
    style(axes_L1, bands_L1)
    axes_L1[0].set_title("Inverter-side inductor L1")

    # ----- C panel -----
    for ax_C in axes_C:
        ax_C.plot(C_smooth , total_wo_C_s, color="red", linestyle="-", linewidth=4.0)
        ax_C.plot(C_smooth , total_w_C_s, color="blue", linestyle="-", linewidth=2.0)
    style(axes_C, bands_C)
    axes_C[0].set_title("Filter capacitor C")

    # ----- L2 panel -----
    for ax_L2 in axes_L2:
        ax_L2.plot(C_smooth , total_wo_L2_s, color="red", linestyle="-", linewidth=4.0)
        ax_L2.plot(C_smooth , total_w_L2_s, color="blue", linestyle="-", linewidth=2.0)
    style(axes_L2, bands_L2)
    axes_L2[0].set_title("Grid-side inductor L2")

    # hide x tick labels on every row except the very last
    all_axes = axes_L1 + axes_C + axes_L2
    for ax in all_axes[:-1]:
        ax.tick_params(labelbottom=False)

    all_axes[-1].set_xlim(100, 80)
    all_axes[-1].set_xlabel("Capacitance relative to initial value [%]")
    fig.supylabel("Voltage [V]", x=-0.04)

    handles = [Patch(facecolor="red", label="Capacitance fixed"),
               Patch(facecolor="blue", label="Capacitance degrading")]

    axes_L1[0].legend(handles=handles, ncol=2, loc="lower center",
                      bbox_to_anchor=(0.5, 1.075), frameon=True,
                      handlelength=2, markerscale=1.0, borderpad=0.5,
                      columnspacing=1, labelspacing=0.25)

    fig.savefig(f"{figures_dir}/{filename}", dpi=600, bbox_inches="tight")
    plt.close(fig)

######################################
# Power Loss Plot
######################################

def plot_power_all_components(df_without, df_with, figures_dir, filename, bands_L1=None, bands_C=None, bands_L2=None,band_variable=0.1):
    # ----------------------------------------#
    # Common x-axis
    # ----------------------------------------#

    Ds_wo, De_wo, ts_wo, te_wo = normalise(df_without["D_cycle_C"].values)
    Ds_w, De_w, ts_w, te_w = normalise(df_with["D_cycle_C"].values)

    Dm_wo = 0.5 * (Ds_wo + De_wo)
    Dm_w = 0.5 * (Ds_w + De_w)

    D_smooth = np.linspace(0.0, 1.0, 500)
    C_smooth = 100 - 20 * D_smooth

    def smooth(D_mid, y):
        return PchipInterpolator(D_mid, y, extrapolate=True)(D_smooth)

    def align(wo_s, w_s):
        offset = wo_s[0] - w_s[0]
        return w_s + offset

    # ----------------------------------------#
    # L1
    # ----------------------------------------#

    total_wo_L1 = df_without["P_total_L1"].values[:len(Ds_wo)]
    total_w_L1 = df_with["P_total_L1"].values[:len(Ds_w)]

    total_wo_L1_s = smooth(Dm_wo, total_wo_L1)
    total_w_L1_s = align(total_wo_L1_s, smooth(Dm_w, total_w_L1))

    # ----------------------------------------#
    # C
    # ----------------------------------------#

    total_wo_C = df_without["P_total_C"].values[:len(Ds_wo)]
    total_w_C = df_with["P_total_C"].values[:len(Ds_w)]

    total_wo_C_s = smooth(Dm_wo, total_wo_C)
    total_w_C_s = align(total_wo_C_s, smooth(Dm_w, total_w_C))

    # ----------------------------------------#
    # L2
    # ----------------------------------------#

    total_wo_L2 = df_without["P_total_L2"].values[:len(Ds_wo)]
    total_w_L2 = df_with["P_total_L2"].values[:len(Ds_w)]

    total_wo_L2_s = smooth(Dm_wo, total_wo_L2)
    total_w_L2_s = align(total_wo_L2_s, smooth(Dm_w, total_w_L2))


    if bands_L1 is None:
        bands_L1 = auto_band(total_wo_L1_s, total_w_L1_s, band_variable)
    if bands_C is None:
        bands_C = auto_band(total_wo_C_s, total_w_C_s, band_variable)
    if bands_L2 is None:
        bands_L2 = auto_band(total_wo_L2_s, total_w_L2_s, band_variable)

    # ----------------------------------------#
    # Figure
    # ----------------------------------------#

    fig = plt.figure(figsize=(6.4, 4.8 * 3 * 0.75))
    gs = fig.add_gridspec(3, 1, hspace=0.125)

    def make_axes(gs_cell, bands, share_ax):
        heights = [b[1] - b[0] for b in bands][::-1]
        sub = gs_cell.subgridspec(len(bands), 1, height_ratios=heights, hspace=0.1)
        axes = []
        for row in range(len(bands)):
            ax = fig.add_subplot(sub[row], sharex=share_ax) if share_ax is not None else fig.add_subplot(sub[row])
            axes.append(ax)
            if share_ax is None:
                share_ax = ax
        return axes, share_ax

    axes_L1, share_ax = make_axes(gs[0], bands_L1, None)
    axes_C, share_ax = make_axes(gs[1], bands_C, share_ax)
    axes_L2, share_ax = make_axes(gs[2], bands_L2, share_ax)

    def style(axes, bands):
        for row in range(len(bands)):
            ax = axes[row]
            lo, hi = bands[len(bands) - 1 - row]
            ax.set_ylim(lo, hi)
            ax.ticklabel_format(axis="y", useOffset=False, style="plain")
            if row > 0:
                ax.spines["top"].set_visible(False)
            if row < len(bands) - 1:
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(bottom=False)

    # ----- L1 panel -----
    for ax_L1 in axes_L1:
        ax_L1.plot(C_smooth, total_wo_L1_s, color="red", linestyle="-", linewidth=4.0)
        ax_L1.plot(C_smooth, total_w_L1_s, color="blue", linestyle="-", linewidth=2.0)
    style(axes_L1, bands_L1)
    axes_L1[0].set_title("Inverter-side inductor L1")

    # ----- C panel -----
    for ax_C in axes_C:
        ax_C.plot(C_smooth, total_wo_C_s, color="red", linestyle="-", linewidth=4.0)
        ax_C.plot(C_smooth, total_w_C_s, color="blue", linestyle="-", linewidth=2.0)
    style(axes_C, bands_C)
    axes_C[0].set_title("Filter capacitor C")

    # ----- L2 panel -----
    for ax_L2 in axes_L2:
        ax_L2.plot(C_smooth, total_wo_L2_s, color="red", linestyle="-", linewidth=4.0)
        ax_L2.plot(C_smooth, total_w_L2_s, color="blue", linestyle="-", linewidth=2.0)
    style(axes_L2, bands_L2)
    axes_L2[0].set_title("Grid-side inductor L2")

    # hide x tick labels on every row except the very last
    all_axes = axes_L1 + axes_C + axes_L2
    for ax in all_axes[:-1]:
        ax.tick_params(labelbottom=False)

    all_axes[-1].set_xlim(100, 80)
    all_axes[-1].set_xlabel("Capacitance relative to initial value [%]")

    fig.supylabel("Power loss [W]", x=-0.02)

    handles = [Patch(facecolor="red", label="Capacitance fixed"),
               Patch(facecolor="blue", label="Capacitance degrading")]

    axes_L1[0].legend(handles=handles, ncol=2, loc="lower center",
                      bbox_to_anchor=(0.5, 1.075), frameon=True,
                      handlelength=2, markerscale=1.0, borderpad=0.5,
                      columnspacing=1, labelspacing=0.25)

    fig.savefig(f"{figures_dir}/{filename}", dpi=600, bbox_inches="tight")
    plt.close(fig)

######################################
# Temperature Plot
######################################

def plot_temperature_all_components(df_without, df_with, figures_dir, filename, bands_L1=None, bands_C=None, bands_L2=None,band_variable=0.1):
    # ----------------------------------------#
    # Common x-axis
    # ----------------------------------------#

    Ds_wo, De_wo, ts_wo, te_wo = normalise(df_without["D_cycle_C"].values)
    Ds_w, De_w, ts_w, te_w = normalise(df_with["D_cycle_C"].values)

    Dm_wo = 0.5 * (Ds_wo + De_wo)
    Dm_w = 0.5 * (Ds_w + De_w)

    D_smooth = np.linspace(0.0, 1.0, 500)
    C_smooth = 100 - 20 * D_smooth

    def smooth(D_mid, y):
        return PchipInterpolator(D_mid, y, extrapolate=True)(D_smooth)

    def align(wo_s, w_s):
        offset = wo_s[0] - w_s[0]
        return w_s + offset

    # ----------------------------------------#
    # L1
    # ----------------------------------------#

    df_without["T_L1"] = df_without["T_L1"] - 273.15
    df_with["T_L1"] = df_with["T_L1"] - 273.15

    total_wo_L1 = df_without["T_L1"].values[:len(Ds_wo)]
    total_w_L1 = df_with["T_L1"].values[:len(Ds_w)]

    total_wo_L1_s = smooth(Dm_wo, total_wo_L1)
    total_w_L1_s = align(total_wo_L1_s, smooth(Dm_w, total_w_L1))

    # ----------------------------------------#
    # C
    # ----------------------------------------#

    df_without["T_C"] = df_without["T_C"] - 273.15
    df_with["T_C"] = df_with["T_C"] - 273.15

    total_wo_C = df_without["T_C"].values[:len(Ds_wo)]
    total_w_C = df_with["T_C"].values[:len(Ds_w)]

    total_wo_C_s = smooth(Dm_wo, total_wo_C)
    total_w_C_s = align(total_wo_C_s, smooth(Dm_w, total_w_C))

    # ----------------------------------------#
    # L2
    # ----------------------------------------#

    df_without["T_L2"] = df_without["T_L2"] - 273.15
    df_with["T_L2"] = df_with["T_L2"] - 273.15

    total_wo_L2 = df_without["T_L2"].values[:len(Ds_wo)]
    total_w_L2 = df_with["T_L2"].values[:len(Ds_w)]

    total_wo_L2_s = smooth(Dm_wo, total_wo_L2)
    total_w_L2_s = align(total_wo_L2_s, smooth(Dm_w, total_w_L2))


    if bands_L1 is None:
        bands_L1 = auto_band(total_wo_L1_s, total_w_L1_s, band_variable)
    if bands_C is None:
        bands_C = auto_band(total_wo_C_s, total_w_C_s, band_variable)
    if bands_L2 is None:
        bands_L2 = auto_band(total_wo_L2_s, total_w_L2_s, band_variable)

    # ----------------------------------------#
    # Figure
    # ----------------------------------------#

    fig = plt.figure(figsize=(6.4, 4.8 * 3 * 0.75))
    gs = fig.add_gridspec(3, 1, hspace=0.125)

    def make_axes(gs_cell, bands, share_ax):
        heights = [b[1] - b[0] for b in bands][::-1]
        sub = gs_cell.subgridspec(len(bands), 1, height_ratios=heights, hspace=0.1)
        axes = []
        for row in range(len(bands)):
            ax = fig.add_subplot(sub[row], sharex=share_ax) if share_ax is not None else fig.add_subplot(sub[row])
            axes.append(ax)
            if share_ax is None:
                share_ax = ax
        return axes, share_ax

    axes_L1, share_ax = make_axes(gs[0], bands_L1, None)
    axes_C, share_ax = make_axes(gs[1], bands_C, share_ax)
    axes_L2, share_ax = make_axes(gs[2], bands_L2, share_ax)

    def style(axes, bands):
        for row in range(len(bands)):
            ax = axes[row]
            lo, hi = bands[len(bands) - 1 - row]
            ax.set_ylim(lo, hi)
            ax.ticklabel_format(axis="y", useOffset=False, style="plain")
            if row > 0:
                ax.spines["top"].set_visible(False)
            if row < len(bands) - 1:
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(bottom=False)

    # ----- L1 panel -----
    for ax_L1 in axes_L1:
        ax_L1.plot(C_smooth, total_wo_L1_s, color="red", linestyle="-", linewidth=4.0)
        ax_L1.plot(C_smooth, total_w_L1_s, color="blue", linestyle="-", linewidth=2.0)
    style(axes_L1, bands_L1)
    axes_L1[0].set_title("Inverter-side inductor L1")

    # ----- C panel -----
    for ax_C in axes_C:
        ax_C.plot(C_smooth, total_wo_C_s, color="red", linestyle="-", linewidth=4.0)
        ax_C.plot(C_smooth, total_w_C_s, color="blue", linestyle="-", linewidth=2.0)
    style(axes_C, bands_C)
    axes_C[0].set_title("Filter capacitor C")

    # ----- L2 panel -----
    for ax_L2 in axes_L2:
        ax_L2.plot(C_smooth, total_wo_L2_s, color="red", linestyle="-", linewidth=4.0)
        ax_L2.plot(C_smooth, total_w_L2_s, color="blue", linestyle="-", linewidth=2.0)
    style(axes_L2, bands_L2)
    axes_L2[0].set_title("Grid-side inductor L2")

    # hide x tick labels on every row except the very last
    all_axes = axes_L1 + axes_C + axes_L2
    for ax in all_axes[:-1]:
        ax.tick_params(labelbottom=False)

    all_axes[-1].set_xlim(100, 80)
    all_axes[-1].set_xlabel("Capacitance relative to initial value [%]")


    fig.supylabel("Temperature [°C]",x=0.0)

    handles = [Patch(facecolor="red", label="Capacitance fixed"),
               Patch(facecolor="blue", label="Capacitance degrading")]

    axes_L1[0].legend(handles=handles, ncol=2, loc="lower center",
                      bbox_to_anchor=(0.5, 1.075), frameon=True,
                      handlelength=2, markerscale=1.0, borderpad=0.5,
                      columnspacing=1, labelspacing=0.25)

    fig.savefig(f"{figures_dir}/{filename}", dpi=600, bbox_inches="tight")
    plt.close(fig)

######################################
# Damage accumulation plot
######################################

def plot_damage_accumulation_all_components(df_without, df_with, figures_dir, filename, bands_L1=None, bands_C=None, bands_L2=None,band_variable=0.1):

    # ----------------------------------------#
    # Common x-axis: years, anchored at (0, 0)
    # ----------------------------------------#

    years_wo = np.concatenate(([0.0], df_without["year"].values))
    years_w = np.concatenate(([0.0], df_with["year"].values))

    year_smooth_wo = np.linspace(0.0, years_wo[-1], 500)
    year_smooth_w = np.linspace(0.0, years_w[-1], 500)

    def smooth(x, y, x_new):
        return PchipInterpolator(x, np.concatenate(([0.0], y)), extrapolate=True)(x_new)

    def crossing(x_new, y_s):
        if y_s[-1] < 1.0:
            return None
        return np.interp(1.0, y_s, x_new)

    def clip(y_s):
        y_s = y_s.copy()
        y_s[y_s > 1.0] = np.nan
        return y_s

    # ----------------------------------------#
    # L1
    # ----------------------------------------#

    total_wo_L1_s = smooth(years_wo, df_without["D_cum_L1"].values, year_smooth_wo)
    total_w_L1_s = smooth(years_w, df_with["D_cum_L1"].values, year_smooth_w)

    # ----------------------------------------#
    # C
    # ----------------------------------------#

    total_wo_C_s = smooth(years_wo, df_without["D_cum_C"].values, year_smooth_wo)
    total_w_C_s = smooth(years_w, df_with["D_cum_C"].values, year_smooth_w)

    # ----------------------------------------#
    # L2
    # ----------------------------------------#

    total_wo_L2_s = smooth(years_wo, df_without["D_cum_L2"].values, year_smooth_wo)
    total_w_L2_s = smooth(years_w, df_with["D_cum_L2"].values, year_smooth_w)


    if bands_L1 is None:
        bands_L1 = auto_band(total_wo_L1_s, total_w_L1_s, band_variable)
    if bands_C is None:
        bands_C = auto_band(total_wo_C_s, total_w_C_s, band_variable)
    if bands_L2 is None:
        bands_L2 = auto_band(total_wo_L2_s, total_w_L2_s, band_variable)

    # ----------------------------------------#
    # Figure
    # ----------------------------------------#

    fig = plt.figure(figsize=(6.4, 4.8 * 3 * 0.75))
    gs = fig.add_gridspec(3, 1, hspace=0.125)

    def make_axes(gs_cell, bands, share_ax):
        heights = [b[1] - b[0] for b in bands][::-1]
        sub = gs_cell.subgridspec(len(bands), 1, height_ratios=heights, hspace=0.1)
        axes = []
        for row in range(len(bands)):
            ax = fig.add_subplot(sub[row], sharex=share_ax) if share_ax is not None else fig.add_subplot(sub[row])
            axes.append(ax)
            if share_ax is None:
                share_ax = ax
        return axes, share_ax

    axes_L1, share_ax = make_axes(gs[0], bands_L1, None)
    axes_C, share_ax = make_axes(gs[1], bands_C, share_ax)
    axes_L2, share_ax = make_axes(gs[2], bands_L2, share_ax)

    def style(axes, bands):
        for row in range(len(bands)):
            ax = axes[row]
            lo, hi = bands[len(bands) - 1 - row]
            ax.set_ylim(lo, hi)
            ax.ticklabel_format(axis="y", useOffset=False, style="plain")
            if row > 0:
                ax.spines["top"].set_visible(False)
            if row < len(bands) - 1:
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(bottom=False)

    def markers(ax, wo_s, w_s):
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5)
        x_wo = crossing(year_smooth_wo, wo_s)
        x_w = crossing(year_smooth_w, w_s)
        trans = ax.get_xaxis_transform()
        if x_wo is not None:
            ax.axvline(x_wo, color="red", linestyle="--", linewidth=1.5)
            ax.text(x_wo-0.25, 0.03, f"{x_wo:.2f}", color="red",
                    transform=trans, ha="right", va="bottom", fontsize=15)
        if x_w is not None:
            ax.axvline(x_w, color="blue", linestyle="--", linewidth=1.5)
            ax.text(x_w+0.25, 0.03, f"{x_w:.2f}", color="blue",
                    transform=trans, ha="left", va="bottom", fontsize=15)

    # ----- L1 panel -----
    for ax_L1 in axes_L1:
        ax_L1.plot(year_smooth_wo, clip(total_wo_L1_s), color="red", linestyle="-", linewidth=4.0)
        ax_L1.plot(year_smooth_w, clip(total_w_L1_s), color="blue", linestyle="-", linewidth=2.0)
        markers(ax_L1, total_wo_L1_s, total_w_L1_s)
    style(axes_L1, bands_L1)
    axes_L1[0].set_title("Inverter-side inductor L1")

    # ----- C panel -----
    for ax_C in axes_C:
        ax_C.plot(year_smooth_wo, clip(total_wo_C_s), color="red", linestyle="-", linewidth=4.0)
        ax_C.plot(year_smooth_w, clip(total_w_C_s), color="blue", linestyle="-", linewidth=2.0)
        markers(ax_C, total_wo_C_s, total_w_C_s)
    style(axes_C, bands_C)
    axes_C[0].set_title("Filter capacitor C")

    # ----- L2 panel -----
    for ax_L2 in axes_L2:
        ax_L2.plot(year_smooth_wo, clip(total_wo_L2_s), color="red", linestyle="-", linewidth=4.0)
        ax_L2.plot(year_smooth_w, clip(total_w_L2_s), color="blue", linestyle="-", linewidth=2.0)
        markers(ax_L2, total_wo_L2_s, total_w_L2_s)
    style(axes_L2, bands_L2)
    axes_L2[0].set_title("Grid-side inductor L2")

    # hide x tick labels on every row except the very last
    all_axes = axes_L1 + axes_C + axes_L2
    for ax in all_axes[:-1]:
        ax.tick_params(labelbottom=False)

    all_axes[-1].set_xlim(0, 15.5)
    all_axes[-1].set_xlabel("Time [years]")

    fig.supylabel("Damage [-]", x=0.02)

    handles = [Patch(facecolor="red", label="Capacitance fixed"),
               Patch(facecolor="blue", label="Capacitance degrading")]

    axes_L1[0].legend(handles=handles, ncol=2, loc="lower center",
                      bbox_to_anchor=(0.5, 1.075), frameon=True,
                      handlelength=2, markerscale=1.0, borderpad=0.5,
                      columnspacing=1, labelspacing=0.25)

    fig.savefig(f"{figures_dir}/{filename}", dpi=600, bbox_inches="tight")
    plt.close(fig)

######################################
# Monte Carlo Simulation
######################################

def Monte_carlo_frame_building():

    number_of_samples = 100000
    normal_distribution = 0.01
    rng = np.random.default_rng(42)

    # L1 — with drift

    L_eq_L1_years_with = death_year(df_with["D_cum_L1"])
    T_eq_L1_with = Calculation_functions.equivalent_temperature(L_eq_L1_years_with, L1_specs["T_insulation_rated"], L1_specs["L_insulation_rated"], L1_specs["Ea_insulation"], L1_specs["kb"])
    T_L1_samples_with = Calculation_functions.normal_distribution_function(T_eq_L1_with, normal_distribution, number_of_samples, rng)
    T_rated_L1_samples_with = Calculation_functions.normal_distribution_function(L1_specs["T_insulation_rated"], normal_distribution, number_of_samples, rng)
    L_rated_L1_samples_with = Calculation_functions.normal_distribution_function(L1_specs["L_insulation_rated"], normal_distribution, number_of_samples, rng)
    Ea_L1_samples_with = Calculation_functions.normal_distribution_function(L1_specs["Ea_insulation"], normal_distribution, number_of_samples, rng)
    Lifetime_L1_MC_with_drift = Calculation_functions.calculate_inductor_lifetime(T_operating=T_L1_samples_with, T_rated=T_rated_L1_samples_with, L_rated=L_rated_L1_samples_with, Ea=Ea_L1_samples_with, kb=L1_specs["kb"], L_max_years=L1_specs["L_max_years"],)

    # L1 — without drift

    L_eq_L1_years_without = death_year(df_without["D_cum_L1"])
    T_eq_L1_without = Calculation_functions.equivalent_temperature(L_eq_L1_years_without, L1_specs["T_insulation_rated"], L1_specs["L_insulation_rated"], L1_specs["Ea_insulation"], L1_specs["kb"])
    T_L1_samples_without = Calculation_functions.normal_distribution_function(T_eq_L1_without, normal_distribution, number_of_samples, rng)
    T_rated_L1_samples_without = Calculation_functions.normal_distribution_function(L1_specs["T_insulation_rated"], normal_distribution, number_of_samples, rng)
    L_rated_L1_samples_without = Calculation_functions.normal_distribution_function(L1_specs["L_insulation_rated"], normal_distribution, number_of_samples, rng)
    Ea_L1_samples_without = Calculation_functions.normal_distribution_function(L1_specs["Ea_insulation"], normal_distribution, number_of_samples, rng)
    Lifetime_L1_MC_without_drift = Calculation_functions.calculate_inductor_lifetime(T_operating=T_L1_samples_without, T_rated=T_rated_L1_samples_without, L_rated=L_rated_L1_samples_without, Ea=Ea_L1_samples_without, kb=L1_specs["kb"], L_max_years=L1_specs["L_max_years"],)

    # L2 — with drift

    L_eq_L2_years_with = death_year(df_with["D_cum_L2"])
    T_eq_L2_with = Calculation_functions.equivalent_temperature(L_eq_L2_years_with, L2_specs["T_insulation_rated"], L2_specs["L_insulation_rated"], L2_specs["Ea_insulation"], L2_specs["kb"])
    T_L2_samples_with = Calculation_functions.normal_distribution_function(T_eq_L2_with, normal_distribution, number_of_samples, rng)
    T_rated_L2_samples_with = Calculation_functions.normal_distribution_function(L2_specs["T_insulation_rated"], normal_distribution, number_of_samples, rng)
    L_rated_L2_samples_with = Calculation_functions.normal_distribution_function(L2_specs["L_insulation_rated"], normal_distribution, number_of_samples, rng)
    Ea_L2_samples_with = Calculation_functions.normal_distribution_function(L2_specs["Ea_insulation"], normal_distribution, number_of_samples, rng)
    Lifetime_L2_MC_with_drift = Calculation_functions.calculate_inductor_lifetime(T_operating=T_L2_samples_with, T_rated=T_rated_L2_samples_with, L_rated=L_rated_L2_samples_with, Ea=Ea_L2_samples_with, kb=L2_specs["kb"], L_max_years=L2_specs["L_max_years"],)

    # L2 — without drift

    L_eq_L2_years_without = death_year(df_without["D_cum_L2"])
    T_eq_L2_without = Calculation_functions.equivalent_temperature(L_eq_L2_years_without, L2_specs["T_insulation_rated"], L2_specs["L_insulation_rated"], L2_specs["Ea_insulation"], L2_specs["kb"])
    T_L2_samples_without = Calculation_functions.normal_distribution_function(T_eq_L2_without, normal_distribution, number_of_samples, rng)
    T_rated_L2_samples_without = Calculation_functions.normal_distribution_function(L2_specs["T_insulation_rated"], normal_distribution, number_of_samples, rng)
    L_rated_L2_samples_without = Calculation_functions.normal_distribution_function(L2_specs["L_insulation_rated"], normal_distribution, number_of_samples, rng)
    Ea_L2_samples_without = Calculation_functions.normal_distribution_function(L2_specs["Ea_insulation"], normal_distribution, number_of_samples, rng)
    Lifetime_L2_MC_without_drift = Calculation_functions.calculate_inductor_lifetime(T_operating=T_L2_samples_without, T_rated=T_rated_L2_samples_without, L_rated=L_rated_L2_samples_without, Ea=Ea_L2_samples_without, kb=L2_specs["kb"], L_max_years=L2_specs["L_max_years"],)



    # ----------------------------------------#
    # Capacitor — without drift
    # ----------------------------------------#

    L_eq_C_years_without = death_year(df_without["D_cum_C"])

    T_eq_C_without = Calculation_functions.equivalent_temperature_capacitor(L_eq_years=L_eq_C_years_without, V_C_RMS=df_without["V_C_RMS"].mean(), V_C_RMS_Rated=C_specs["V_C_RMS_Rated"], t1=C_specs["Lifetime_Rated"], T1=C_specs["Temperature_Rated"], A=C_specs["A"], n=C_specs["n"])

    T_C_samples_without = Calculation_functions.normal_distribution_function(T_eq_C_without, normal_distribution, number_of_samples, rng)
    V_C_samples_without = Calculation_functions.normal_distribution_function(df_without["V_C_RMS"].mean(), normal_distribution, number_of_samples, rng)
    V_C_RMS_Rated_samples_without = Calculation_functions.normal_distribution_function(C_specs["V_C_RMS_Rated"], normal_distribution, number_of_samples, rng)
    t1_samples_without = Calculation_functions.normal_distribution_function(C_specs["Lifetime_Rated"], normal_distribution, number_of_samples, rng)
    T1_samples_without = Calculation_functions.normal_distribution_function(C_specs["Temperature_Rated"], normal_distribution, number_of_samples, rng)
    A_samples_without = Calculation_functions.normal_distribution_function(C_specs["A"], normal_distribution, number_of_samples, rng)
    n_samples_without = Calculation_functions.normal_distribution_function(C_specs["n"], normal_distribution, number_of_samples, rng)

    Lifetime_C_MC_without_drift = Calculation_functions.calculate_capacitor_lifetime_analytical(T_operating=T_C_samples_without, V_C_RMS=V_C_samples_without, V_C_RMS_Rated=V_C_RMS_Rated_samples_without, t1=t1_samples_without, T1=T1_samples_without, A=A_samples_without, n=n_samples_without,)

    # Capacitor — with drift

    L_eq_C_years_with = death_year(df_with["D_cum_C"])
    T_eq_C_with = Calculation_functions.equivalent_temperature_capacitor(L_eq_years=L_eq_C_years_with, V_C_RMS=df_with["V_C_RMS"].mean(), V_C_RMS_Rated=C_specs["V_C_RMS_Rated"], t1=C_specs["Lifetime_Rated"], T1=C_specs["Temperature_Rated"], A=C_specs["A"], n=C_specs["n"])
    T_C_samples_with = Calculation_functions.normal_distribution_function(T_eq_C_with, normal_distribution, number_of_samples, rng)
    V_C_samples_with = Calculation_functions.normal_distribution_function(df_with["V_C_RMS"].mean(), normal_distribution, number_of_samples, rng)
    V_C_RMS_Rated_samples_with = Calculation_functions.normal_distribution_function(C_specs["V_C_RMS_Rated"], normal_distribution, number_of_samples, rng)
    t1_samples_with = Calculation_functions.normal_distribution_function(C_specs["Lifetime_Rated"], normal_distribution, number_of_samples, rng)
    T1_samples_with = Calculation_functions.normal_distribution_function(C_specs["Temperature_Rated"], normal_distribution, number_of_samples, rng)
    A_samples_with = Calculation_functions.normal_distribution_function(C_specs["A"], normal_distribution, number_of_samples, rng)
    n_samples_with = Calculation_functions.normal_distribution_function(C_specs["n"], normal_distribution, number_of_samples, rng)
    Lifetime_C_MC_with_drift = Calculation_functions.calculate_capacitor_lifetime_analytical(T_operating=T_C_samples_with, V_C_RMS=V_C_samples_with, V_C_RMS_Rated=V_C_RMS_Rated_samples_with, t1=t1_samples_with, T1=T1_samples_with, A=A_samples_with, n=n_samples_with,)


    # Filter lifetime: the weakest element


    Lifetime_LCL_MC_with_drift = np.minimum.reduce([Lifetime_C_MC_with_drift, Lifetime_L1_MC_with_drift, Lifetime_L2_MC_with_drift])
    Lifetime_LCL_MC_without_drift = np.minimum.reduce([Lifetime_C_MC_without_drift, Lifetime_L1_MC_without_drift, Lifetime_L2_MC_without_drift])


    # Monte Carlo dataframe

    df_MC = pd.DataFrame()
    df_MC["Lifetime_L1_with_drift"] = Lifetime_L1_MC_with_drift
    df_MC["Lifetime_L1_without_drift"] = Lifetime_L1_MC_without_drift
    df_MC["Lifetime_C_with_drift"] = Lifetime_C_MC_with_drift
    df_MC["Lifetime_C_without_drift"] = Lifetime_C_MC_without_drift
    df_MC["Lifetime_L2_with_drift"] = Lifetime_L2_MC_with_drift
    df_MC["Lifetime_L2_without_drift"] = Lifetime_L2_MC_without_drift
    df_MC["Lifetime_LCL_with_drift"] = Lifetime_LCL_MC_with_drift
    df_MC["Lifetime_LCL_without_drift"] = Lifetime_LCL_MC_without_drift
    df_MC.to_parquet(f"{ROOT}/OAPE_results/Results/df_MC.parquet")

    return df_MC

#df_MC = Monte_carlo_frame_building()

######################################
# Unreliability / B10 Plot
######################################

df_MC = pd.read_parquet(f"{ROOT}/OAPE_results/Results/df_MC.parquet")

def plot_unreliability(df_MC, figures_dir, filename, x_max=30):

    components = ["L1", "C", "L2", "LCL"]
    titles = ["Inverter-side inductor L1",
              "Filter capacitor C",
              "Grid-side inductor L2",
              "LCL filter"]

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(6.4, 4.8 * 4 * 0.75),
                             gridspec_kw={"hspace": 0.25})

    for row in range(4):

        ax = axes[row]
        comp = components[row]

        life_wo = np.sort(df_MC[f"Lifetime_{comp}_without_drift"].values)
        life_w = np.sort(df_MC[f"Lifetime_{comp}_with_drift"].values)

        # empirical unreliability: fraction failed by each life value
        F = np.arange(1, len(life_wo) + 1) / len(life_wo)

        ax.plot(life_wo, F, color="red", linestyle="-", linewidth=4.0)
        ax.plot(life_w, F, color="blue", linestyle="-", linewidth=2.0)

        # B10 markers
        B10_wo = np.percentile(df_MC[f"Lifetime_{comp}_without_drift"].values, 10)
        B10_w = np.percentile(df_MC[f"Lifetime_{comp}_with_drift"].values, 10)

        ax.axhline(0.10, color="black", linestyle="--", linewidth=1.5)
        ax.axvline(B10_wo, color="red", linestyle="--", linewidth=1.5)
        ax.axvline(B10_w, color="blue", linestyle="--", linewidth=1.5)

        trans = ax.get_xaxis_transform()
        ax.text(B10_wo - 0.3, 0.03, f"{B10_wo:.2f}", color="red",
                transform=trans, ha="right", va="bottom", fontsize=15)
        ax.text(B10_w + 0.3, 0.03, f"{B10_w:.2f}", color="blue",
                transform=trans, ha="left", va="bottom", fontsize=15)

        ax.set_ylim(0, 0.11)
        ax.set_title(titles[row])
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")

    axes[-1].set_xlim(0, x_max)
    axes[-1].set_xlabel("Time [years]")
    fig.supylabel("Unreliability [-]", x=-0.02)

    handles = [Patch(facecolor="red", label="Capacitance fixed"),
               Patch(facecolor="blue", label="Capacitance degrading")]

    axes[0].legend(handles=handles, ncol=2, loc="lower center",
                   bbox_to_anchor=(0.5, 1.075), frameon=True,
                   handlelength=2, markerscale=1.0, borderpad=0.5,
                   columnspacing=1, labelspacing=0.25)

    fig.savefig(f"{figures_dir}/{filename}", dpi=600, bbox_inches="tight")
    plt.close(fig)

######################################
# All functions running here
######################################

plot_currents_all_components(df_without, df_with, figures_dir, "1_currents_all_components.png", band_variable=0.1)
plot_df_7_harmonics_stacked(df_7_drift=df_7_drift, df_7_nominal=df_7_nominal,figures_dir=figures_dir,filename ="2_Harmonics_stacked", bands=((0, 30), (185, 215), (385, 415)))
plot_THD(df_without, df_with, figures_dir, "3_THD.png",band_variable=0.1)
plot_voltage_all_components(df_without, df_with, figures_dir, "4_voltage_all_components.png",band_variable=0.1)
plot_power_all_components(df_without, df_with, figures_dir, "5_power_all_components.png", band_variable=0.1)
plot_temperature_all_components(df_without, df_with, figures_dir, "6_temperature_all_components.png",band_variable=0.1)
plot_damage_accumulation_all_components(df_without, df_with, figures_dir, "7_damage_all_components.png",band_variable=0.1)
plot_unreliability(df_MC, figures_dir, "8_unreliability.png", x_max=9)