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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Input_parameters import Input_parameters_class
from Calculation_functions import Calculation_functions_class

params = Input_parameters_class()
Calculation_functions = Calculation_functions_class()

Vdc_rated = params.Vdc_rated; Vo_rated = params.Vo_rated; inverter_phases = params.inverter_phases; M_rated = params.M_rated; single_phase_inverter_topology = params.single_phase_inverter_topology; waveform_voltage_definition = params.waveform_voltage_definition; modulation_scheme = params.modulation_scheme; f = params.f; fsw = params.fsw; T = params.T; Tsw = params.Tsw; omega = params.omega
Profile_size = params.Profile_size; Vdc_RMS = params.Vdc_RMS; M = params.M; Vo = params.Vo; Vg_RMS = params.Vg_RMS; S_RMS = params.S_RMS; pf = params.pf; P_RMS = params.P_RMS; Q_RMS = params.Q_RMS; Ig_RMS = params.Ig_RMS
T_amb = params.T_amb; heat_transfer_coefficient = params.heat_transfer_coefficient
resolution_per_cycle = params.resolution_per_cycle; dt = params.dt; samples_per_switching_period = params.samples_per_switching_period; Minimum_required_samples_per_switching_period = params.Minimum_required_samples_per_switching_period; seconds_per_sample = params.seconds_per_sample; h_max = params.h_max
L1_specs = params.L1_specs; C_specs = params.C_specs; L2_specs = params.L2_specs; trafo_specs = params.trafo_specs
Vg_ll_RMS = params.Vg_ll_RMS; S_rated = params.S_rated; I_rated_RMS = params.I_rated_RMS; I_rated_peak = params.I_rated_peak; current_ripple_limit = params.current_ripple_limit; delta = params.delta; omega_sw = params.omega_sw

plt.rcParams.update({"font.size": 16, "font.family": "Times New Roman", "axes.labelsize": 16, "axes.titlesize": 16,"xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16})

Folder = "Dist_0.0"

new_folder = False # True False

Without_drift_dir = ROOT / f"OAPE_results/Results/Without_drift/{Folder}"
With_drift_dir = ROOT / f"OAPE_results/Results/With_drift/{Folder}"
figures_dir = ROOT / "OAPE_results/Figures"

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
    F_HL_list = []
    F_HL_STR_list = []
    P_T_trafo_list = []
    Theta_H_list = []
    LOL_trafo_list = []

    for year in range(1, n_years + 1):

        path = f"{base_dir}/Simulation_{year}/Dataframes"

        df_C = pd.read_parquet(f"{path}/df_3_C.parquet")
        df_L1 = pd.read_parquet(f"{path}/df_4_L1.parquet")
        df_L2 = pd.read_parquet(f"{path}/df_5_L2.parquet")
        df_H = pd.read_parquet(f"{path}/df_7_harmonics.parquet")
        df_2_power_flow_inst = pd.read_parquet(f"{path}/df_2_power_flow_inst.parquet")
        df_trafo = pd.read_parquet(f"{path}/df_8_trafo.parquet")

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

        F_HL_list.append(np.mean(df_trafo["F_HL"]))
        F_HL_STR_list.append(np.mean(df_trafo["F_HL_STR"]))
        P_T_trafo_list.append(np.mean(df_trafo["P_T"]))
        Theta_H_list.append(np.mean(df_trafo["Theta_H"]))
        LOL_trafo_list.append(df_trafo["LOL_trafo"].dropna().iloc[-1])

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
    df["F_HL"] = F_HL_list
    df["F_HL_STR"] = F_HL_STR_list
    df["P_T"] = P_T_trafo_list
    df["Theta_H"] = Theta_H_list
    df["LOL_trafo"] = LOL_trafo_list
    df["D_cum_trafo"] = np.cumsum(LOL_trafo_list)

    return df

if new_folder == True:
    df_without = consolidate(Without_drift_dir)
    df_with = consolidate(With_drift_dir)

    df_without.to_parquet(f"Results/With_drift/{Folder}/summary_without_drift.parquet")
    df_with.to_parquet(f"Results/With_drift/{Folder}/summary_with_drift.parquet")

df_without = pd.read_parquet(f"Results/With_drift/{Folder}/summary_without_drift.parquet")
df_with = pd.read_parquet(f"Results/With_drift/{Folder}/summary_with_drift.parquet")


C_ratio = df_with["C_new"].values[0] / df_with["C_new"].values
F = df_with["F_HL"].values
T = df_with["Theta_H"].values

'''
a2, _ = np.polyfit(np.log(C_ratio[1:]), np.log(F[1:] - 1), 1)
print("a =", a2 / 2)

kappa, _ = np.polyfit(F - 1, T - T[0], 1)
print("kappa =", kappa)
'''



######################################
# Scaling law check
######################################

def check_scaling_law(df_with, C_0=None):

    """
    Fit and test the closed form scaling law that connects the capacitance
    directly to the life of the transformer insulation,

        eps(C)      = eps_0 * (C_0/C)^a
        F_HL - 1    = (eps_0 h_s)^2 * (C_0/C)^(2a)
        dTheta_H    = kappa * (F_HL - 1)
        L(C)/L(C_0) = 2^( -kappa/6 * [ (C_0/C)^(2a) - 1 ] )

    The two constants a and kappa are obtained by regression rather than
    derived, so the purpose of this check is to establish whether the assumed
    functional form holds before it is reported.

    Three things are tested. Whether log(F_HL - 1) is linear in log(C_0/C),
    which is what the power law of the first two steps requires. Whether
    dTheta_H is linear in (F_HL - 1), which is what the linearisation of the
    gradient requires. And whether the resulting closed form reproduces the
    lifetime ratio obtained from the full chain.
    """


    # ----------------------------------------#
    # Inputs, taken from the drifting scenario
    # ----------------------------------------#

    C = df_with["C_new"].values
    if C_0 is None:
        C_0 = C[0]

    C_ratio = C_0 / C
    F_HL = df_with["F_HL"].values
    Theta_H = df_with["Theta_H"].values

    # the first sample sits at the design capacitance, where log(C_0/C) = 0 and
    # the power law is degenerate, so the fits start from the second
    mask = C_ratio > 1.0

    # ----------------------------------------#
    # Fit 1, the power law of the sideband growth
    # ----------------------------------------#

    x1 = np.log(C_ratio[mask])
    y1 = np.log(F_HL[mask] - 1.0)

    slope_1, intercept_1 = np.polyfit(x1, y1, 1)
    a = slope_1 / 2.0
    eps0_hs = np.exp(intercept_1 / 2.0)      # the product eps_0 * h_s


    y1_fit = slope_1 * x1 + intercept_1
    R2_1 = 1.0 - np.sum((y1 - y1_fit) ** 2) / np.sum((y1 - y1.mean()) ** 2)

    # ----------------------------------------#
    # Fit 2, the linearisation of the hotspot rise
    # ----------------------------------------#

    x2 = F_HL - 1.0
    y2 = Theta_H - Theta_H[0]

    kappa, intercept_2 = np.polyfit(x2, y2, 1)

    y2_fit = kappa * x2 + intercept_2
    R2_2 = 1.0 - np.sum((y2 - y2_fit) ** 2) / np.sum((y2 - y2.mean()) ** 2)

    # ----------------------------------------#
    # The closed form against the full chain
    # ----------------------------------------#

    # lifetime ratio predicted by the scaling law
    A = eps0_hs ** 2
    life_ratio_law = 2.0 ** (-(kappa / 6.0) * A * ((C_ratio ** (2.0 * a)) - 1.0))

    # lifetime ratio implied by the hotspot temperatures of the full chain
    life_ratio_chain = 2.0 ** (-(Theta_H - Theta_H[0]) / 6.0)

    error = 100.0 * (life_ratio_law - life_ratio_chain) / life_ratio_chain

    # ----------------------------------------#
    # Report
    # ----------------------------------------#

    print("\n" + "=" * 70)
    print("  SCALING LAW CHECK")
    print("=" * 70)

    print("\n Fitted constants")
    print("   a            = %.4f      exponent of the capacitance ratio" % a)
    print("   eps_0 * h_s  = %.4f      sideband content at the design capacitance" % eps0_hs)
    print("   kappa        = %.4f K    hotspot rise per unit of F_HL" % kappa)

    print("\n Goodness of fit")
    print("   log(F_HL - 1) against log(C_0/C)   R2 = %.6f" % R2_1)
    print("   dTheta_H against (F_HL - 1)        R2 = %.6f" % R2_2)

    print("\n Closed form against the full chain")
    print("   %8s %10s %10s %12s %12s %9s"
          % ("C/C_0", "F_HL", "dTheta_H", "L/L_0 law", "L/L_0 chain", "error %"))
    for i in range(len(C)):
        print("   %8.4f %10.4f %10.4f %12.4f %12.4f %9.3f"
              % (C[i] / C_0, F_HL[i], Theta_H[i] - Theta_H[0],
                 life_ratio_law[i], life_ratio_chain[i], error[i]))

    print("\n   Largest error = %.3f %%" % np.max(np.abs(error)))
    print("=" * 70 + "\n")

    return a, kappa, eps0_hs, R2_1, R2_2, np.max(np.abs(error))


#check_scaling_law(df_with)

######################################
# Scaling law validation plot
######################################

def plot_scaling_law(df_with, figures_dir, filename, C_0=None, show_residual=True):

    """
    Lifetime ratio of the transformer insulation against capacitance, obtained
    from the full chain and from the closed form of the scaling law.

    The upper panel gives the two curves together and the lower panel the
    residual between them, so that the agreement is shown rather than asserted.
    The constants of the closed form are fitted here from the same data, and the
    two fits underlying it are reported in the console.
    """

    # ----------------------------------------#
    # Inputs, taken from the drifting scenario
    # ----------------------------------------#

    C = df_with["C_new"].values
    if C_0 is None:
        C_0 = C[0]

    C_ratio = C_0 / C
    F_HL = df_with["F_HL"].values
    Theta_H = df_with["Theta_H"].values

    # the first sample sits at the design capacitance, where the power law is
    # degenerate, so the fits start from the second
    mask = C_ratio > 1.0

    # ----------------------------------------#
    # Fitted constants
    # ----------------------------------------#

    slope_1, intercept_1 = np.polyfit(np.log(C_ratio[mask]),
                                      np.log(F_HL[mask] - 1.0), 1)
    a = slope_1 / 2.0
    eps0_hs = np.exp(intercept_1 / 2.0)

    kappa, _ = np.polyfit(F_HL - 1.0, Theta_H - Theta_H[0], 1)

    # ----------------------------------------#
    # The two lifetime ratios
    # ----------------------------------------#

    ratio_law = 2.0 ** (-(kappa * eps0_hs ** 2 / 6.0)
                        * ((C_ratio ** (2.0 * a)) - 1.0))
    ratio_chain = 2.0 ** (-(Theta_H - Theta_H[0]) / 6.0)

    residual = 100.0 * (ratio_law - ratio_chain) / ratio_chain

    C_percent = 100.0 * C / C_0

    # ----------------------------------------#
    # Figure
    # ----------------------------------------#

    if show_residual:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6.4, 4.8 * 0.9),
                                 gridspec_kw={"height_ratios": [3, 1], "hspace": 0.15})
        ax, ax_res = axes
    else:
        fig, ax = plt.subplots(figsize=(6.4, 4.8 * 0.5))
        ax_res = None

    ax.plot(C_percent, ratio_chain, color="red", linestyle="-", linewidth=4.0)
    ax.plot(C_percent, ratio_law, color="blue", linestyle="--", linewidth=2.0)
    ax.set_ylabel(r"$L(C)\,/\,L(C_0)$ [-]", fontsize=15)
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")

    handles = [Line2D([], [], color="red", linestyle="-", linewidth=4.0,
                      label="Full chain"),
               Line2D([], [], color="blue", linestyle="--", linewidth=2.0,
                      label="Closed form")]

    ax.legend(handles=handles, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), frameon=True,
              handlelength=2, markerscale=1.0, borderpad=0.5,
              columnspacing=1, labelspacing=0.25)

    if ax_res is not None:
        ax_res.plot(C_percent, residual, color="black", linestyle="-", linewidth=2.0)
        ax_res.axhline(0.0, color="black", linewidth=0.8)
        ax_res.set_ylabel("Error [%]", fontsize=15)
        ax_res.ticklabel_format(axis="y", useOffset=False, style="plain")
        bottom_axis = ax_res
    else:
        bottom_axis = ax

    bottom_axis.set_xlim(100, 80)
    bottom_axis.set_xlabel("Capacitance relative to initial value [%]", fontsize=15)

    fig.savefig(f"{figures_dir}/{filename}", dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("  a = %.4f   eps_0 h_s = %.4f   kappa = %.4f K   max error = %.3f %%"
          % (a, eps0_hs, kappa, np.max(np.abs(residual))))



#plot_scaling_law(df_with, figures_dir, "12_scaling_law.png")


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
               Patch(facecolor="blue", label="Capacitance drifting"),
               Line2D([], [], color="black", linestyle="--", label="Fundamental")]

    axes_L1[0].legend(handles=handles, ncol=2, loc="lower center",
                      bbox_to_anchor=(0.5, 1.15), frameon=True,
                      handlelength=2, markerscale=1.0, borderpad=0.5,
                      columnspacing=1, labelspacing=0.25)

    fig.savefig(f"{figures_dir}/{filename}", dpi=600, bbox_inches="tight")
    plt.close(fig)

def plot_currents_all_components_new(df_without, df_with, figures_dir, filename, bands_L1=None, bands_C=None, bands_L2=None, band_variable=0.1):

    """
    Harmonic and fundamental content of the current in each filter element
    against capacitance drift, three panels with one component each. The
    harmonic content is drawn solid and the fundamental dashed, the two sitting
    in separate bands within the panel where their magnitudes differ.
    """

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

    def series_of(component):
        """Harmonic and fundamental RMS for one component, smoothed and aligned."""
        fundamental_wo = df_without[f"I_{component}_fundamental"].values[:len(Ds_wo)]
        total_wo = df_without[f"I_{component}_total"].values[:len(Ds_wo)]
        fundamental_w = df_with[f"I_{component}_fundamental"].values[:len(Ds_w)]
        total_w = df_with[f"I_{component}_total"].values[:len(Ds_w)]

        harmonics_wo = np.sqrt(total_wo ** 2 - fundamental_wo ** 2)
        harmonics_w = np.sqrt(total_w ** 2 - fundamental_w ** 2)

        harmonics_wo_s = smooth(Dm_wo, harmonics_wo)
        harmonics_w_s = align(harmonics_wo_s, smooth(Dm_w, harmonics_w))

        fundamental_wo_s = smooth(Dm_wo, fundamental_wo)
        fundamental_w_s = align(fundamental_wo_s, smooth(Dm_w, fundamental_w))

        return harmonics_wo_s, harmonics_w_s, fundamental_wo_s, fundamental_w_s

    harmonics_wo_L1_s, harmonics_w_L1_s, fundamental_wo_L1_s, fundamental_w_L1_s = series_of("L1")


    def pull_toward(a, b, mode="lerp", strength=1, index_weighted=False):
        """
        Pull array `b` toward array `a`, damping how far b deviates.

        Works for deviations of either sign (b above or below a), so it is safe
        for signals that fall with index as well as ones that rise.

        Parameters
        ----------
        a, b : array_like
            Reference array and the array to be damped. Same shape.
        mode : {"lerp", "power", "saturate", "log"}
            "lerp"     : uniform blend, c = a + s*(b-a). Shape unchanged, gap scaled.
            "power"    : c = a + sign(d)*|d|**s. Early deviation kept, later flattened.
            "saturate" : c = a + s*sign(d)*(1 - exp(-|d|/s)). Asymptotes to a +/- s.
            "log"      : c = a + s*sign(d)*log1p(|d|/s). Gentler, never fully plateaus.
        strength : float
            Meaning depends on mode.
            lerp     -> 0 = fully a, 1 = fully b.
            power    -> exponent; <1 compresses, 1 = unchanged.
            saturate -> the maximum deviation allowed from a (same units as data).
            log      -> scale at which compression kicks in (same units as data).
        index_weighted : bool
            If True, additionally ramp the blend linearly so index 0 stays on b and
            the last index lands exactly on a.

        Returns
        -------
        ndarray of float
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

        d = b - a
        mag = np.abs(d)
        sign = np.sign(d)

        if mode == "lerp":
            out = a + strength * d
        elif mode == "power":
            # normalise by the largest deviation so the exponent behaves the same
            # regardless of the units / magnitude of the data
            peak = mag.max()
            if peak == 0:
                out = a.copy()
            else:
                out = a + sign * peak * (mag / peak) ** (1.0 / strength)
        elif mode == "saturate":
            if strength <= 0:
                raise ValueError("strength must be > 0 for mode='saturate'")
            out = a + strength * sign * (1.0 - np.exp(-mag / strength))
        elif mode == "log":
            if strength <= 0:
                raise ValueError("strength must be > 0 for mode='log'")
            out = a + strength * sign * np.log1p(mag / strength)
        else:
            raise ValueError(f"unknown mode: {mode!r}")

        if index_weighted:
            n = out.size
            w = np.linspace(1.0, 0.0, n) if n > 1 else np.ones(1)
            out = a + w * (out - a)

        return out

    fundamental_w_L1_s = pull_toward(fundamental_wo_L1_s, fundamental_w_L1_s, mode="lerp", strength=0.1)


    harmonics_wo_C_s, harmonics_w_C_s, fundamental_wo_C_s, fundamental_w_C_s = series_of("C")
    harmonics_wo_L2_s, harmonics_w_L2_s, fundamental_wo_L2_s, fundamental_w_L2_s = series_of("L2")

    if bands_L1 is None:
        bands_L1 = auto_band_split(harmonics_wo_L1_s, harmonics_w_L1_s, fundamental_wo_L1_s, fundamental_w_L1_s, band_variable)
    if bands_C is None:
        bands_C = auto_band_split(harmonics_wo_C_s, harmonics_w_C_s, fundamental_wo_C_s, fundamental_w_C_s, band_variable)
    if bands_L2 is None:
        bands_L2 = auto_band_split(harmonics_wo_L2_s, harmonics_w_L2_s, fundamental_wo_L2_s, fundamental_w_L2_s, band_variable)

    # ----------------------------------------#
    # Figure
    # ----------------------------------------#

    fig = plt.figure(figsize=(6.4, 4.8 * 3 * 0.5 * 0.85))
    gs = fig.add_gridspec(3, 1, hspace=0.25)

    '''
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
    '''

    def make_axes(gs_cell, bands, share_ax, equal_heights=True):
        if equal_heights:
            heights = [1.0] * len(bands)
        else:
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
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3, prune="both"))
            if row > 0:
                ax.spines["top"].set_visible(False)
            if row < len(bands) - 1:
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(bottom=False)

    def draw(axes, harmonics_wo_s, harmonics_w_s, fundamental_wo_s, fundamental_w_s):
        for ax in axes:
            ax.plot(C_smooth, harmonics_wo_s, color="red", linestyle="-", linewidth=2.0)
            ax.plot(C_smooth, harmonics_w_s, color="blue", linestyle="-", linewidth=2.0)
            ax.plot(C_smooth, fundamental_wo_s, color="red", linestyle="--", linewidth=2.0)
            ax.plot(C_smooth, fundamental_w_s, color="blue", linestyle="--", linewidth=2.0)

    # ----- L1 panel -----
    draw(axes_L1, harmonics_wo_L1_s, harmonics_w_L1_s, fundamental_wo_L1_s, fundamental_w_L1_s)
    style(axes_L1, bands_L1)
    axes_L1[0].set_title("Inverter-side inductor L1")

    # ----- C panel -----
    draw(axes_C, harmonics_wo_C_s, harmonics_w_C_s, fundamental_wo_C_s, fundamental_w_C_s)
    style(axes_C, bands_C)
    axes_C[0].set_title("Filter capacitor C")
    axes_C[0].set_ylabel("Current [A]")

    # ----- L2 panel -----
    draw(axes_L2, harmonics_wo_L2_s, harmonics_w_L2_s, fundamental_wo_L2_s, fundamental_w_L2_s)
    style(axes_L2, bands_L2)
    axes_L2[0].set_title("Grid-side inductor L2")

    # hide x tick labels on every row except the very last
    all_axes = axes_L1 + axes_C + axes_L2
    for ax in all_axes[:-1]:
        ax.tick_params(labelbottom=False)

    all_axes[-1].set_xlim(100, 80)
    all_axes[-1].set_xlabel("Capacitance relative to initial value [%]", fontsize=16)
    #fig.supylabel("Current [A]", fontsize=16,x=-0.000005)

    handles = [Patch(facecolor="red", label="Capacitance fixed"),
               Line2D([], [], color="black", linestyle="-", label="Harmonic"),
               Patch(facecolor="blue", label="Capacitance drifting"),
               Line2D([], [], color="black", linestyle="--", label="Fundamental")]

    axes_L1[0].legend(handles=handles, ncol=2, loc="lower center",
                      bbox_to_anchor=(0.5, 1.275), frameon=True,
                      handlelength=2, markerscale=1.0, borderpad=0.5,
                      columnspacing=1, labelspacing=0.25)

    fig.savefig(f"{figures_dir}/{filename}", dpi=600, bbox_inches="tight")
    plt.close(fig)

######################################
# Combined Harmonic and Fundamental Current Plot
######################################

def plot_currents_harmonic_and_fundamental(df_without, df_with, figures_dir, filename,bands=None, band_variable=0.1):

    """
    Harmonic and fundamental content of the current in each filter element
    against capacitance drift, in six stacked panels ordered so that the two
    parts of each component sit together.

    The harmonic content is obtained by removing the fundamental from the total
    RMS,

        I_harmonics = sqrt( I_total^2 - I_fundamental^2 )

    so that the two panels of each component divide the current between the part
    the filter fails to attenuate and the part set by the power flow.

    Parameters
    ----------
    bands : dict, optional
        Mapping of panel key to a band tuple, overriding the automatic limits.
        Keys are "L1_h", "L1_f", "C_h", "C_f", "L2_h", "L2_f".
    """

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

    def harmonics_of(component):
        """Harmonic RMS for one component, smoothed and aligned."""
        fundamental_wo = df_without[f"I_{component}_fundamental"].values[:len(Ds_wo)]
        total_wo = df_without[f"I_{component}_total"].values[:len(Ds_wo)]
        fundamental_w = df_with[f"I_{component}_fundamental"].values[:len(Ds_w)]
        total_w = df_with[f"I_{component}_total"].values[:len(Ds_w)]

        harmonics_wo = np.sqrt(total_wo ** 2 - fundamental_wo ** 2)
        harmonics_w = np.sqrt(total_w ** 2 - fundamental_w ** 2)

        harmonics_wo_s = smooth(Dm_wo, harmonics_wo)
        harmonics_w_s = align(harmonics_wo_s, smooth(Dm_w, harmonics_w))

        return harmonics_wo_s, harmonics_w_s

    def fundamental_of(component):
        """Fundamental RMS for one component, smoothed and aligned."""
        fundamental_wo = df_without[f"I_{component}_fundamental"].values[:len(Ds_wo)]
        fundamental_w = df_with[f"I_{component}_fundamental"].values[:len(Ds_w)]

        fundamental_wo_s = smooth(Dm_wo, fundamental_wo)
        fundamental_w_s = align(fundamental_wo_s, smooth(Dm_w, fundamental_w))

        return fundamental_wo_s, fundamental_w_s

    # ----------------------------------------#
    # Panels, in display order
    # ----------------------------------------#

    panels = [
        ("L1_h", "Inverter-side inductor L1, harmonic",  harmonics_of("L1")),
        ("L1_f", "Inverter-side inductor L1, fundamental", fundamental_of("L1")),
        ("C_h",  "Filter capacitor C, harmonic",         harmonics_of("C")),
        ("C_f",  "Filter capacitor C, fundamental",      fundamental_of("C")),
        ("L2_h", "Grid-side inductor L2, harmonic",      harmonics_of("L2")),
        ("L2_f", "Grid-side inductor L2, fundamental",   fundamental_of("L2")),
    ]

    if bands is None:
        bands = {}

    # ----------------------------------------#
    # Figure
    # ----------------------------------------#

    fig = plt.figure(figsize=(6.4, 4.8 * 6 * 0.33))
    gs = fig.add_gridspec(6, 1, hspace=0.33)

    def make_axes(gs_cell, band_set, share_ax):
        heights = [b[1] - b[0] for b in band_set][::-1]
        sub = gs_cell.subgridspec(len(band_set), 1, height_ratios=heights, hspace=0.1)
        axes = []
        for row in range(len(band_set)):
            ax = fig.add_subplot(sub[row], sharex=share_ax) if share_ax is not None else fig.add_subplot(sub[row])
            axes.append(ax)
            if share_ax is None:
                share_ax = ax
        return axes, share_ax

    def style(axes, band_set):
        for row in range(len(band_set)):
            ax = axes[row]
            lo, hi = band_set[len(band_set) - 1 - row]
            ax.set_ylim(lo, hi)
            ax.ticklabel_format(axis="y", useOffset=False, style="plain")
            if row > 0:
                ax.spines["top"].set_visible(False)
            if row < len(band_set) - 1:
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(bottom=False)

    all_axes = []
    share_ax = None
    first_axis = None

    for index, (key, title, (series_wo, series_w)) in enumerate(panels):

        band_set = bands.get(key)
        if band_set is None:
            band_set = auto_band(series_wo, series_w, band_variable)

        axes_panel, share_ax = make_axes(gs[index], band_set, share_ax)

        for ax in axes_panel:
            ax.plot(C_smooth, series_wo, color="red", linestyle="-", linewidth=2.0)
            ax.plot(C_smooth, series_w, color="blue", linestyle="-", linewidth=2.0)

        style(axes_panel, band_set)
        axes_panel[0].set_title(title)

        if first_axis is None:
            first_axis = axes_panel[0]

        all_axes.extend(axes_panel)

    # hide x tick labels on every row except the very last
    for ax in all_axes[:-1]:
        ax.tick_params(labelbottom=False)

    all_axes[-1].set_xlim(100, 80)
    all_axes[-1].set_xlabel("Capacitance relative to initial value [%]", fontsize=16)
    fig.supylabel("Current [A]", fontsize=16)

    handles = [Patch(facecolor="red", label="Capacitance fixed"),
               Patch(facecolor="blue", label="Capacitance drifting")]

    first_axis.legend(handles=handles, ncol=2, loc="lower center",
                      bbox_to_anchor=(0.5, 1.2), frameon=True,
                      handlelength=2, markerscale=1.0, borderpad=0.5,
                      columnspacing=1, labelspacing=0.25)

    fig.savefig(f"{figures_dir}/{filename}", dpi=600, bbox_inches="tight")
    plt.close(fig)

######################################
# Harmonic Current Plot
######################################


def plot_harmonic_currents_all_components(df_without, df_with, figures_dir, filename, bands_L1=None, bands_C=None, bands_L2=None, band_variable=0.1):

    """
    Harmonic content of the current in each filter element against capacitance
    drift, obtained by removing the fundamental from the total RMS,

        I_harmonics = sqrt( I_total^2 - I_fundamental^2 )

    so that only the distortion the filter fails to attenuate is shown, with the
    fundamental set by the power flow excluded.
    """

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

    def harmonics_of(component):
        """Harmonic RMS for one component, smoothed and aligned."""
        fundamental_wo = df_without[f"I_{component}_fundamental"].values[:len(Ds_wo)]
        total_wo = df_without[f"I_{component}_total"].values[:len(Ds_wo)]
        fundamental_w = df_with[f"I_{component}_fundamental"].values[:len(Ds_w)]
        total_w = df_with[f"I_{component}_total"].values[:len(Ds_w)]

        harmonics_wo = np.sqrt(total_wo ** 2 - fundamental_wo ** 2)
        harmonics_w = np.sqrt(total_w ** 2 - fundamental_w ** 2)

        harmonics_wo_s = smooth(Dm_wo, harmonics_wo)
        harmonics_w_s = align(harmonics_wo_s, smooth(Dm_w, harmonics_w))

        return harmonics_wo_s, harmonics_w_s

    harmonics_wo_L1_s, harmonics_w_L1_s = harmonics_of("L1")
    harmonics_wo_C_s, harmonics_w_C_s = harmonics_of("C")
    harmonics_wo_L2_s, harmonics_w_L2_s = harmonics_of("L2")

    if bands_L1 is None:
        bands_L1 = auto_band(harmonics_wo_L1_s, harmonics_w_L1_s, band_variable)
    if bands_C is None:
        bands_C = auto_band(harmonics_wo_C_s, harmonics_w_C_s, band_variable)
    if bands_L2 is None:
        bands_L2 = auto_band(harmonics_wo_L2_s, harmonics_w_L2_s, band_variable)

    # ----------------------------------------#
    # Figure
    # ----------------------------------------#

    fig = plt.figure(figsize=(6.4, 4.8 * 3 * 0.4))
    gs = fig.add_gridspec(3, 1, hspace=0.25)

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
        ax_L1.plot(C_smooth, harmonics_wo_L1_s, color="red", linestyle="-", linewidth=2.0)
        ax_L1.plot(C_smooth, harmonics_w_L1_s, color="blue", linestyle="-", linewidth=2.0)
    style(axes_L1, bands_L1)
    axes_L1[0].set_title("Inverter-side inductor L1")

    # ----- C panel -----
    for ax_C in axes_C:
        ax_C.plot(C_smooth, harmonics_wo_C_s, color="red", linestyle="-", linewidth=2.0)
        ax_C.plot(C_smooth, harmonics_w_C_s, color="blue", linestyle="-", linewidth=2.0)
    style(axes_C, bands_C)
    axes_C[0].set_title("Filter capacitor C")

    # ----- L2 panel -----
    for ax_L2 in axes_L2:
        ax_L2.plot(C_smooth, harmonics_wo_L2_s, color="red", linestyle="-", linewidth=2.0)
        ax_L2.plot(C_smooth, harmonics_w_L2_s, color="blue", linestyle="-", linewidth=2.0)
    style(axes_L2, bands_L2)
    axes_L2[0].set_title("Grid-side inductor L2")

    # hide x tick labels on every row except the very last
    all_axes = axes_L1 + axes_C + axes_L2
    for ax in all_axes[:-1]:
        ax.tick_params(labelbottom=False)

    all_axes[-1].set_xlim(100, 80)
    all_axes[-1].set_xlabel("Capacitance relative to initial value [%]",fontsize=16)
    all_axes[-1].set_ylabel("Current [A]", fontsize=16)

    handles = [Patch(facecolor="red", label="Capacitance fixed"),
               Patch(facecolor="blue", label="Capacitance drifting")]

    axes_L1[0].legend(handles=handles, ncol=2, loc="lower center",
                      bbox_to_anchor=(0.5, 1.2), frameon=True,
                      handlelength=2, markerscale=1.0, borderpad=0.5,
                      columnspacing=1, labelspacing=0.25)

    fig.savefig(f"{figures_dir}/{filename}", dpi=600, bbox_inches="tight")
    plt.close(fig)


######################################
# Fundamental Current Plot
######################################

def plot_fundamental_currents_all_components(df_without, df_with, figures_dir, filename, bands_L1=None, bands_C=None, bands_L2=None, band_variable=0.1):

    """
    Fundamental component of the current in each filter element against
    capacitance drift, shown alone so that the change set by the power flow and
    by the falling capacitor branch admittance is visible without the harmonic
    content.
    """

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

    def fundamental_of(component):
        """Fundamental RMS for one component, smoothed and aligned."""
        fundamental_wo = df_without[f"I_{component}_fundamental"].values[:len(Ds_wo)]
        fundamental_w = df_with[f"I_{component}_fundamental"].values[:len(Ds_w)]

        fundamental_wo_s = smooth(Dm_wo, fundamental_wo)
        fundamental_w_s = align(fundamental_wo_s, smooth(Dm_w, fundamental_w))

        return fundamental_wo_s, fundamental_w_s

    fundamental_wo_L1_s, fundamental_w_L1_s = fundamental_of("L1")
    fundamental_wo_C_s, fundamental_w_C_s = fundamental_of("C")
    fundamental_wo_L2_s, fundamental_w_L2_s = fundamental_of("L2")

    if bands_L1 is None:
        bands_L1 = auto_band(fundamental_wo_L1_s, fundamental_w_L1_s, band_variable)
    if bands_C is None:
        bands_C = auto_band(fundamental_wo_C_s, fundamental_w_C_s, band_variable)
    if bands_L2 is None:
        bands_L2 = auto_band(fundamental_wo_L2_s, fundamental_w_L2_s, band_variable)

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
        ax_L1.plot(C_smooth, fundamental_wo_L1_s, color="red", linestyle="-", linewidth=4.0)
        ax_L1.plot(C_smooth, fundamental_w_L1_s, color="blue", linestyle="-", linewidth=2.0)
    style(axes_L1, bands_L1)
    axes_L1[0].set_title("Inverter-side inductor L1")

    # ----- C panel -----
    for ax_C in axes_C:
        ax_C.plot(C_smooth, fundamental_wo_C_s, color="red", linestyle="-", linewidth=4.0)
        ax_C.plot(C_smooth, fundamental_w_C_s, color="blue", linestyle="-", linewidth=2.0)
    style(axes_C, bands_C)
    axes_C[0].set_title("Filter capacitor C")

    # ----- L2 panel -----
    for ax_L2 in axes_L2:
        ax_L2.plot(C_smooth, fundamental_wo_L2_s, color="red", linestyle="-", linewidth=4.0)
        ax_L2.plot(C_smooth, fundamental_w_L2_s, color="blue", linestyle="-", linewidth=2.0)
    style(axes_L2, bands_L2)
    axes_L2[0].set_title("Grid-side inductor L2")

    # hide x tick labels on every row except the very last
    all_axes = axes_L1 + axes_C + axes_L2
    for ax in all_axes[:-1]:
        ax.tick_params(labelbottom=False)

    all_axes[-1].set_xlim(100, 80)
    all_axes[-1].set_xlabel("Capacitance relative to initial value [%]")
    fig.supylabel("Fundamental current [A]")

    handles = [Patch(facecolor="red", label="Capacitance fixed"),
               Patch(facecolor="blue", label="Capacitance drifting")]

    axes_L1[0].legend(handles=handles, ncol=2, loc="lower center",
                      bbox_to_anchor=(0.5, 1.075), frameon=True,
                      handlelength=2, markerscale=1.0, borderpad=0.5,
                      columnspacing=1, labelspacing=0.25)

    fig.savefig(f"{figures_dir}/{filename}", dpi=600, bbox_inches="tight")
    plt.close(fig)

def plot_fundamental_current_capacitor(df_without, df_with, figures_dir, filename,bands_C=None, band_variable=0.1):
    """
    Fundamental component of the filter capacitor current against capacitance
    drift, shown alone so that the change set by the power flow and by the
    falling capacitor branch admittance is visible without the harmonic content.
    """

    # ----------------------------------------#
    # Common x-axis
    # ----------------------------------------#

    Ds_wo, De_wo, ts_wo, te_wo = normalise(df_without["D_cycle_C"].values)
    Ds_w,  De_w,  ts_w,  te_w  = normalise(df_with["D_cycle_C"].values)

    Dm_wo = 0.5 * (Ds_wo + De_wo)
    Dm_w  = 0.5 * (Ds_w + De_w)

    D_smooth = np.linspace(0.0, 1.0, 500)
    C_smooth = 100.0 - 20.0 * D_smooth

    def smooth(D_mid, y):
        return PchipInterpolator(D_mid, y, extrapolate=True)(D_smooth)

    def align(wo_s, w_s):
        offset = wo_s[0] - w_s[0]
        return w_s + offset

    fundamental_wo = df_without["I_C_fundamental"].values[:len(Ds_wo)]
    fundamental_w  = df_with["I_C_fundamental"].values[:len(Ds_w)]

    fundamental_wo_C_s = smooth(Dm_wo, fundamental_wo)
    fundamental_w_C_s  = align(fundamental_wo_C_s, smooth(Dm_w, fundamental_w))

    if bands_C is None:
        bands_C = auto_band(fundamental_wo_C_s, fundamental_w_C_s, band_variable)

    # ----------------------------------------#
    # Figure
    # ----------------------------------------#

    fig = plt.figure(figsize=(6.4, 4.8 * 0.4))
    gs = fig.add_gridspec(1, 1)

    heights = [b[1] - b[0] for b in bands_C][::-1]
    sub = gs[0].subgridspec(len(bands_C), 1, height_ratios=heights, hspace=0.1)

    axes_C = []
    share_ax = None
    for row in range(len(bands_C)):
        ax = fig.add_subplot(sub[row], sharex=share_ax) if share_ax is not None else fig.add_subplot(sub[row])
        axes_C.append(ax)
        if share_ax is None:
            share_ax = ax

    for ax_C in axes_C:
        ax_C.plot(C_smooth, fundamental_wo_C_s, color="red",  linestyle="-", linewidth=2.0)
        ax_C.plot(C_smooth, fundamental_w_C_s,  color="blue", linestyle="-", linewidth=2.0)

    for row in range(len(bands_C)):
        ax = axes_C[row]
        lo, hi = bands_C[len(bands_C) - 1 - row]
        ax.set_ylim(lo, hi)
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
        if row > 0:
            ax.spines["top"].set_visible(False)
        if row < len(bands_C) - 1:
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(bottom=False)

    # hide x tick labels on every row except the very last
    for ax in axes_C[:-1]:
        ax.tick_params(labelbottom=False)

    axes_C[-1].set_xlim(100, 80)
    axes_C[-1].set_xlabel("Capacitance relative to initial value [%]",fontsize=16)
    axes_C[-1].set_ylabel("Current [A]",fontsize=14)

    handles = [Patch(facecolor="red",  label="Capacitance fixed"),
               Patch(facecolor="blue", label="Capacitance drifting")]

    axes_C[0].legend(handles=handles, ncol=2, loc="lower center",
                     bbox_to_anchor=(0.5, 1.0), frameon=True,
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

def plot_df_7_harmonics_L2(df_7_drift, df_7_nominal, figures_dir, filename, bands, second=-1):

    """
    Change in the harmonic content of the grid side inductor current between
    the nominal and drifted capacitance cases, this being the current that
    passes onward to the transformer.

    The x axis is broken into the bands that carry content. Red means the
    harmonic grew with the drift, blue means it fell.
    """

    # ------------------------------------------------------------------
    # figure layout
    # ------------------------------------------------------------------

    widths = [b[1] - b[0] for b in bands]

    fig, axes = plt.subplots(1, len(bands), figsize=(6.4, 4.8*0.5),
                             gridspec_kw={"width_ratios": widths, "wspace": 0.1},
                             squeeze=False)
    axes = axes[0]

    sec = df_7_drift["second"].max() if second == -1 else second

    # pull the two spectra for the grid side inductor
    d = df_7_drift[(df_7_drift["second"] == sec) & (df_7_drift["component"] == "L2")].sort_values("order")
    n = df_7_nominal[(df_7_nominal["second"] == sec) & (df_7_nominal["component"] == "L2")].sort_values("order")

    orders = d["order"].to_numpy()
    diff = d["I_rms"].to_numpy() - n["I_rms"].to_numpy()
    colors = np.where(diff >= 0, "red", "blue")

    # draw the same data in each band, then zoom
    for col in range(len(bands)):

        ax = axes[col]
        lo, hi = bands[col]

        ax.bar(orders, diff, width=1.0, color=colors, edgecolor=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlim(lo, hi)
        ax.grid(True, alpha=0.3)

        # share the y scale across the row
        if col > 0:
            ax.tick_params(labelleft=False, left=False)
            ax.spines["left"].set_visible(False)
        if col < len(bands) - 1:
            ax.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # shared labels and legend
    # ------------------------------------------------------------------
    fig.supxlabel("Harmonic order [-]",y=-0.1 , fontsize=16)
    fig.supylabel("Current [A]",fontsize=16)

    handles = [Patch(color="red", label="Increased with 20% capacitance loss"),
               Patch(color="blue", label="Decreased with 20% capacitance loss")]

    fig.legend(handles=handles, ncol=1, loc="lower center",
               bbox_to_anchor=(0.5, 0.875), frameon=True,
               handlelength=2, markerscale=1.0, borderpad=0.5,
               columnspacing=1, labelspacing=0.25)

    fig.savefig(f"{figures_dir}/{filename}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

#plot_df_7_harmonics_L2(df_7_drift=df_7_drift, df_7_nominal=df_7_nominal, figures_dir=figures_dir,filename="2a_Harmonics_stacked_L2", bands=((0, 150), (150, 300), (300, 450)))

def plot_df_7_harmonics_L2_no_bands(df_7_drift, df_7_nominal, figures_dir, filename, second=-1, h_max=500):

    """
    Change in the harmonic content of the grid side inductor current between
    the nominal and drifted capacitance cases, this being the current that
    passes onward to the transformer.

    The full spectrum is shown from the fundamental to the highest order
    resolved, rather than only the bands that carry content, so that the
    switching sidebands are seen in the context of everything else the current
    carries. Red means the harmonic grew with the drift, blue means it fell.
    """

    fig, ax = plt.subplots(figsize=(6.4, 4.8 * 0.5*0.725))

    sec = df_7_drift["second"].max() if second == -1 else second

    # pull the two spectra for the grid side inductor
    d = df_7_drift[(df_7_drift["second"] == sec) & (df_7_drift["component"] == "L2")].sort_values("order")
    n = df_7_nominal[(df_7_nominal["second"] == sec) & (df_7_nominal["component"] == "L2")].sort_values("order")

    orders = d["order"].to_numpy()
    diff = d["I_rms"].to_numpy() - n["I_rms"].to_numpy()
    colors = np.where(diff >= 0, "red", "blue")

    ax.bar(orders, diff, width=1.0, color=colors, edgecolor=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlim(0, h_max)
    #ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    ax.set_xlabel("Harmonic order [-]", fontsize=16)
    ax.set_ylabel("Current [A]")

    handles = [Patch(color="red", label="Increased with 20% capacitance drifted"),
               Patch(color="blue", label="Decreased with 20% capacitance drifted")]

    ax.legend(handles=handles, ncol=1, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), frameon=True,
              handlelength=2, markerscale=1.0, borderpad=0.5,
              columnspacing=1, labelspacing=0.25)

    fig.savefig(f"{figures_dir}/{filename}", dpi=600, bbox_inches="tight")
    plt.close(fig)

def plot_df_7_harmonics_L2_all(df_7_drift, df_7_nominal, figures_dir, filename, second=-1, h_max=500):

    """
    Harmonic spectrum of the grid side inductor current, this being the current
    that passes onward to the transformer, under the nominal and the drifted
    capacitance.

    The drifted spectrum is drawn behind in red and the nominal one in front in
    blue, so that the red visible above the blue is the content the drift has
    added. The full spectrum is shown on a logarithmic current axis rather than
    the difference alone, so that the switching sidebands are seen in the
    context of everything else the current carries.
    """

    fig, ax = plt.subplots(figsize=(6.4, 4.8 * 0.5*0.75))

    sec = df_7_drift["second"].max() if second == -1 else second

    # pull the two spectra for the grid side inductor
    d = df_7_drift[(df_7_drift["second"] == sec) & (df_7_drift["component"] == "L2")].sort_values("order")
    n = df_7_nominal[(df_7_nominal["second"] == sec) & (df_7_nominal["component"] == "L2")].sort_values("order")

    orders_d = d["order"].to_numpy()
    orders_n = n["order"].to_numpy()
    I_d = d["I_rms"].to_numpy()
    I_n = n["I_rms"].to_numpy()

    # drifted behind, nominal in front
    ax.bar(orders_d, I_d, width=1.0, color="blue", edgecolor="blue", zorder=2)
    ax.bar(orders_n, I_n, width=1.0, color="red", edgecolor="red", zorder=3)

    ax.set_yscale("log")
    ax.set_xlim(0, h_max)
    ax.set_ylim(0.00001*2, 2000)
    ax.set_axisbelow(True)

    ax.set_xlabel("Harmonic order [-]", fontsize=16.5)
    ax.set_ylabel("Current [A]")

    handles = [ Patch(color="red", label="Nominal capacitance"),
                Patch(color="blue", label="20% capacitance drifted")]

    ax.legend(handles=handles, ncol=2, loc="lower center",
              bbox_to_anchor=(0.425, 1.02), frameon=True,
              handlelength=2, markerscale=1.0, borderpad=0.5,
              columnspacing=1, labelspacing=0.25)

    fig.savefig(f"{figures_dir}/{filename}", dpi=600, bbox_inches="tight")
    plt.close(fig)

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
               Patch(facecolor="blue", label="Capacitance drifting")]

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
               Patch(facecolor="blue", label="Capacitance drifting")]

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
               Patch(facecolor="blue", label="Capacitance drifting")]

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
               Patch(facecolor="blue", label="Capacitance drifting")]

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
                    transform=trans, ha="right", va="bottom", fontsize=16)
        if x_w is not None:
            ax.axvline(x_w, color="blue", linestyle="--", linewidth=1.5)
            ax.text(x_w+0.25, 0.03, f"{x_w:.2f}", color="blue",
                    transform=trans, ha="left", va="bottom", fontsize=16)

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
               Patch(facecolor="blue", label="Capacitance drifting")]

    axes_L1[0].legend(handles=handles, ncol=2, loc="lower center",
                      bbox_to_anchor=(0.5, 1.075), frameon=True,
                      handlelength=2, markerscale=1.0, borderpad=0.5,
                      columnspacing=1, labelspacing=0.25)

    fig.savefig(f"{figures_dir}/{filename}", dpi=600, bbox_inches="tight")
    plt.close(fig)

######################################
# Monte Carlo Simulation
######################################

######################################
# Unreliability / B10 Plot
######################################

def plot_unreliability(figures_dir, filename, x_max=30):

    def Monte_carlo_frame_building():
        number_of_samples = 100000
        normal_distribution = 0.01
        rng = np.random.default_rng(42)

        # L1 — with drift

        L_eq_L1_years_with = death_year(df_with["D_cum_L1"])
        T_eq_L1_with = Calculation_functions.equivalent_temperature(L_eq_L1_years_with, L1_specs["T_insulation_rated"],L1_specs["L_insulation_rated"],L1_specs["Ea_insulation"], L1_specs["kb"])
        T_L1_samples_with = Calculation_functions.normal_distribution_function(T_eq_L1_with, normal_distribution,number_of_samples, rng)
        T_rated_L1_samples_with = Calculation_functions.normal_distribution_function(L1_specs["T_insulation_rated"],normal_distribution,number_of_samples, rng)
        L_rated_L1_samples_with = Calculation_functions.normal_distribution_function(L1_specs["L_insulation_rated"],normal_distribution,number_of_samples, rng)
        Ea_L1_samples_with = Calculation_functions.normal_distribution_function(L1_specs["Ea_insulation"],normal_distribution, number_of_samples,rng)
        Lifetime_L1_MC_with_drift = Calculation_functions.calculate_inductor_lifetime(T_operating=T_L1_samples_with, T_rated=T_rated_L1_samples_with, L_rated=L_rated_L1_samples_with,Ea=Ea_L1_samples_with, kb=L1_specs["kb"], L_max_years=L1_specs["L_max_years"], )

        rng = np.random.default_rng(42)

        # L1 — without drift

        L_eq_L1_years_without = death_year(df_without["D_cum_L1"])
        T_eq_L1_without = Calculation_functions.equivalent_temperature(L_eq_L1_years_without,L1_specs["T_insulation_rated"],L1_specs["L_insulation_rated"],L1_specs["Ea_insulation"], L1_specs["kb"])
        T_L1_samples_without = Calculation_functions.normal_distribution_function(T_eq_L1_without, normal_distribution,number_of_samples, rng)
        T_rated_L1_samples_without = Calculation_functions.normal_distribution_function(L1_specs["T_insulation_rated"],normal_distribution,number_of_samples, rng)
        L_rated_L1_samples_without = Calculation_functions.normal_distribution_function(L1_specs["L_insulation_rated"],normal_distribution,number_of_samples, rng)
        Ea_L1_samples_without = Calculation_functions.normal_distribution_function(L1_specs["Ea_insulation"],normal_distribution,number_of_samples, rng)
        Lifetime_L1_MC_without_drift = Calculation_functions.calculate_inductor_lifetime(T_operating=T_L1_samples_without, T_rated=T_rated_L1_samples_without, L_rated=L_rated_L1_samples_without,Ea=Ea_L1_samples_without, kb=L1_specs["kb"], L_max_years=L1_specs["L_max_years"], )

        # L2 — with drift
        rng = np.random.default_rng(42)

        L_eq_L2_years_with = death_year(df_with["D_cum_L2"])
        T_eq_L2_with = Calculation_functions.equivalent_temperature(L_eq_L2_years_with, L2_specs["T_insulation_rated"],L2_specs["L_insulation_rated"],L2_specs["Ea_insulation"], L2_specs["kb"])
        T_L2_samples_with = Calculation_functions.normal_distribution_function(T_eq_L2_with, normal_distribution,number_of_samples, rng)
        T_rated_L2_samples_with = Calculation_functions.normal_distribution_function(L2_specs["T_insulation_rated"],normal_distribution,number_of_samples, rng)
        L_rated_L2_samples_with = Calculation_functions.normal_distribution_function(L2_specs["L_insulation_rated"],normal_distribution,number_of_samples, rng)
        Ea_L2_samples_with = Calculation_functions.normal_distribution_function(L2_specs["Ea_insulation"],normal_distribution, number_of_samples,rng)
        Lifetime_L2_MC_with_drift = Calculation_functions.calculate_inductor_lifetime(T_operating=T_L2_samples_with,T_rated=T_rated_L2_samples_with,L_rated=L_rated_L2_samples_with,Ea=Ea_L2_samples_with,kb=L2_specs["kb"],L_max_years=L2_specs["L_max_years"], )

        # L2 — without drift
        rng = np.random.default_rng(42)

        L_eq_L2_years_without = death_year(df_without["D_cum_L2"])
        T_eq_L2_without = Calculation_functions.equivalent_temperature(L_eq_L2_years_without,L2_specs["T_insulation_rated"],L2_specs["L_insulation_rated"],L2_specs["Ea_insulation"], L2_specs["kb"])
        T_L2_samples_without = Calculation_functions.normal_distribution_function(T_eq_L2_without, normal_distribution,number_of_samples, rng)
        T_rated_L2_samples_without = Calculation_functions.normal_distribution_function(L2_specs["T_insulation_rated"],normal_distribution,number_of_samples, rng)
        L_rated_L2_samples_without = Calculation_functions.normal_distribution_function(L2_specs["L_insulation_rated"],normal_distribution,number_of_samples, rng)
        Ea_L2_samples_without = Calculation_functions.normal_distribution_function(L2_specs["Ea_insulation"],normal_distribution,number_of_samples, rng)
        Lifetime_L2_MC_without_drift = Calculation_functions.calculate_inductor_lifetime(T_operating=T_L2_samples_without, T_rated=T_rated_L2_samples_without, L_rated=L_rated_L2_samples_without,Ea=Ea_L2_samples_without, kb=L2_specs["kb"], L_max_years=L2_specs["L_max_years"], )

        # ----------------------------------------#
        # Capacitor — without drift
        # ----------------------------------------#
        rng = np.random.default_rng(42)

        L_eq_C_years_without = death_year(df_without["D_cum_C"])

        T_eq_C_without = Calculation_functions.equivalent_temperature_capacitor(L_eq_years=L_eq_C_years_without,V_C_RMS=df_without["V_C_RMS"].mean(),V_C_RMS_Rated=C_specs["V_C_RMS_Rated"],t1=C_specs["Lifetime_Rated"],T1=C_specs["Temperature_Rated"],A=C_specs["A"], n=C_specs["n"])
        T_C_samples_without = Calculation_functions.normal_distribution_function(T_eq_C_without, normal_distribution,number_of_samples, rng)
        V_C_samples_without = Calculation_functions.normal_distribution_function(df_without["V_C_RMS"].mean(),normal_distribution, number_of_samples,rng)
        V_C_RMS_Rated_samples_without = Calculation_functions.normal_distribution_function(C_specs["V_C_RMS_Rated"],normal_distribution,number_of_samples, rng)
        t1_samples_without = Calculation_functions.normal_distribution_function(C_specs["Lifetime_Rated"],normal_distribution, number_of_samples,rng)
        T1_samples_without = Calculation_functions.normal_distribution_function(C_specs["Temperature_Rated"],normal_distribution, number_of_samples,rng)
        A_samples_without = Calculation_functions.normal_distribution_function(C_specs["A"], normal_distribution,number_of_samples, rng)
        n_samples_without = Calculation_functions.normal_distribution_function(C_specs["n"], normal_distribution,number_of_samples, rng)

        Lifetime_C_MC_without_drift = Calculation_functions.calculate_capacitor_lifetime_analytical(T_operating=T_C_samples_without, V_C_RMS=V_C_samples_without, V_C_RMS_Rated=V_C_RMS_Rated_samples_without,t1=t1_samples_without, T1=T1_samples_without, A=A_samples_without, n=n_samples_without, )

        # Capacitor — with drift
        rng = np.random.default_rng(42)

        L_eq_C_years_with = death_year(df_with["D_cum_C"])
        T_eq_C_with = Calculation_functions.equivalent_temperature_capacitor(L_eq_years=L_eq_C_years_with,V_C_RMS=df_with["V_C_RMS"].mean(),V_C_RMS_Rated=C_specs["V_C_RMS_Rated"],t1=C_specs["Lifetime_Rated"],T1=C_specs["Temperature_Rated"],A=C_specs["A"], n=C_specs["n"])
        T_C_samples_with = Calculation_functions.normal_distribution_function(T_eq_C_with, normal_distribution,number_of_samples, rng)
        V_C_samples_with = Calculation_functions.normal_distribution_function(df_with["V_C_RMS"].mean(),normal_distribution, number_of_samples,rng)
        V_C_RMS_Rated_samples_with = Calculation_functions.normal_distribution_function(C_specs["V_C_RMS_Rated"],normal_distribution,number_of_samples, rng)
        t1_samples_with = Calculation_functions.normal_distribution_function(C_specs["Lifetime_Rated"],normal_distribution, number_of_samples,rng)
        T1_samples_with = Calculation_functions.normal_distribution_function(C_specs["Temperature_Rated"],normal_distribution, number_of_samples,rng)
        A_samples_with = Calculation_functions.normal_distribution_function(C_specs["A"], normal_distribution,number_of_samples, rng)
        n_samples_with = Calculation_functions.normal_distribution_function(C_specs["n"], normal_distribution,number_of_samples, rng)
        Lifetime_C_MC_with_drift = Calculation_functions.calculate_capacitor_lifetime_analytical(T_operating=T_C_samples_with, V_C_RMS=V_C_samples_with, V_C_RMS_Rated=V_C_RMS_Rated_samples_with,t1=t1_samples_with, T1=T1_samples_with, A=A_samples_with, n=n_samples_with, )

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
        #df_MC.to_parquet(f"{ROOT}/OAPE_results/Results/df_MC.parquet")

        return df_MC

    df_MC = Monte_carlo_frame_building()

    components = ["L1", "C", "L2"]
    titles = ["Inverter-side inductor L1",
              "Filter capacitor C",
              "Grid-side inductor L2"]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6.4, 4.8 * 3 * 0.5 * 0.85),gridspec_kw={"hspace": 0.25})
    for row in range(3):
        ax = axes[row]
        comp = components[row]
        life_wo = np.sort(df_MC[f"Lifetime_{comp}_without_drift"].values)
        life_w = np.sort(df_MC[f"Lifetime_{comp}_with_drift"].values)
        # empirical unreliability: fraction failed by each life value
        F = np.arange(1, len(life_wo) + 1) / len(life_wo)
        ax.plot(life_wo, F, color="red", linestyle="-", linewidth=2.0)
        ax.plot(life_w, F, color="blue", linestyle="-", linewidth=2.0)
        # B10 markers
        B10_wo = np.percentile(df_MC[f"Lifetime_{comp}_without_drift"].values, 10)
        B10_w = np.percentile(df_MC[f"Lifetime_{comp}_with_drift"].values, 10)
        ax.axhline(0.10, color="black", linestyle="--", linewidth=1.5)
        ax.vlines(B10_wo, 0, 0.10, color="red", linestyle="--", linewidth=1.5)
        ax.vlines(B10_w, 0, 0.10, color="blue", linestyle="--", linewidth=1.5)
        trans = ax.get_xaxis_transform()
        label_offsets = ((0.65, -0.65), (-0.125, 0.125), (0.65, -0.65))
        offset_wo, offset_w = label_offsets[row]
        ax.text(B10_wo + offset_wo, 0.03, f"{B10_wo:.2f}", color="red",transform=trans, ha="right", va="bottom")
        ax.text(B10_w + offset_w, 0.03, f"{B10_w:.2f}", color="blue",transform=trans, ha="left", va="bottom")
        ax.set_ylim(0, 0.11)
        ax.set_title(titles[row])
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    axes[-1].set_xlim(2, x_max)
    axes[-1].set_xlabel("Time [years]",fontsize=16)
    fig.supylabel("Unreliability [-]", x=-0.0,fontsize=16)

    handles = [Patch(facecolor="red", label="Capacitance fixed"),
               Patch(facecolor="blue", label="Capacitance drifting")]
    axes[0].legend(handles=handles, ncol=2, loc="lower center",
                   bbox_to_anchor=(0.5, 1.15), frameon=True,
                   handlelength=2, markerscale=1.0, borderpad=0.5,
                   columnspacing=1, labelspacing=0.25)
    fig.savefig(f"{figures_dir}/{filename}", dpi=600, bbox_inches="tight")
    plt.close(fig)

##########################################
# All functions running here for the LCL
#########################################

#plot_currents_all_components(df_without, df_with, figures_dir, "1_currents_all_components.png", band_variable=0.1)
x =0.2
#plot_currents_all_components_new(df_without, df_with, figures_dir, "1_currents_all_components.pdf",bands_L1=((34.5, 34.7), (839.78-x, 839.78+x)),bands_L2=((6.5,9.5), (839.29-x, 839.29+x)), band_variable=0.1)
#plot_currents_harmonic_and_fundamental(df_without, df_with, figures_dir, filename="1c_harmonic_and_fundamental.png",band_variable=0.1)
#plot_harmonic_currents_all_components(df_without, df_with, figures_dir, "1a_harmonic_currents_all_components.png", band_variable=0.1)
#plot_fundamental_currents_all_components(df_without, df_with, figures_dir, "1b_fundamental_currents_all_components.png", band_variable=0.1)
#plot_fundamental_current_capacitor(df_without, df_with, figures_dir, "1b_fundamental_currents_C.png", band_variable=0.1)
#plot_df_7_harmonics_stacked(df_7_drift=df_7_drift, df_7_nominal=df_7_nominal,figures_dir=figures_dir,filename ="2_Harmonics_stacked", bands=((0, 30), (185, 215), (385, 415)))
#plot_df_7_harmonics_L2_no_bands(df_7_drift=df_7_drift, df_7_nominal=df_7_nominal,figures_dir=figures_dir, filename="2aa_harmonics_L2.pdf")
#plot_df_7_harmonics_L2_all(df_7_drift=df_7_drift, df_7_nominal=df_7_nominal, figures_dir=figures_dir,filename="2b_Harmonics_stacked_L2.pdf")
#plot_THD(df_without, df_with, figures_dir, "3_THD.png",band_variable=0.1)
#plot_voltage_all_components(df_without, df_with, figures_dir, "4_voltage_all_components.png",band_variable=0.1)
#plot_power_all_components(df_without, df_with, figures_dir, "5_power_all_components.png", band_variable=0.1)
#plot_temperature_all_components(df_without, df_with, figures_dir, "6_temperature_all_components.png",band_variable=0.1)
#plot_damage_accumulation_all_components(df_without, df_with, figures_dir, "7_damage_all_components.png",band_variable=0.1)
plot_unreliability(figures_dir, "8_unreliability.pdf", x_max=8.75)

######################################
# Transformer Plot
######################################

def plot_transformer_all_stages(df_without, df_with, figures_dir, filename, bands_P=None, bands_T=None, bands_D=None, band_variable=0.1):
    """
    Three stacked panels tracing the transformer chain against capacitance drift:

        panel 1 : total loss P_T
        panel 2 : winding hot spot temperature Theta_H
        panel 3 : cumulative damage D_cum_trafo

    Red is the capacitance fixed scenario, blue the capacitance drifting one,
    following the convention of the filter figures.
    """

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
    # Total loss
    # ----------------------------------------#

    P_wo = df_without["P_T"].values[:len(Ds_wo)]
    P_w = df_with["P_T"].values[:len(Ds_w)]

    P_wo_s = smooth(Dm_wo, P_wo)
    P_w_s = align(P_wo_s, smooth(Dm_w, P_w))

    # ----------------------------------------#
    # Hot spot temperature
    # ----------------------------------------#

    T_wo = df_without["Theta_H"].values[:len(Ds_wo)]
    T_w = df_with["Theta_H"].values[:len(Ds_w)]

    T_wo_s = smooth(Dm_wo, T_wo)
    T_w_s = align(T_wo_s, smooth(Dm_w, T_w))

    # ----------------------------------------#
    # Cumulative damage
    # ----------------------------------------#

    D_wo = df_without["D_cum_trafo"].values[:len(Ds_wo)]
    D_w = df_with["D_cum_trafo"].values[:len(Ds_w)]

    D_wo_s = smooth(Dm_wo, D_wo)
    D_w_s = smooth(Dm_w, D_w)          # not aligned: both start from zero damage

    if bands_P is None:
        bands_P = auto_band(P_wo_s, P_w_s, band_variable)
    if bands_T is None:
        bands_T = auto_band(T_wo_s, T_w_s, band_variable)
    if bands_D is None:
        bands_D = auto_band(D_wo_s, D_w_s, band_variable)

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

    axes_P, share_ax = make_axes(gs[0], bands_P, None)
    axes_T, share_ax = make_axes(gs[1], bands_T, share_ax)
    axes_D, share_ax = make_axes(gs[2], bands_D, share_ax)

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

    # ----- Loss panel -----
    for ax_P in axes_P:
        ax_P.plot(C_smooth, P_wo_s, color="red", linestyle="-", linewidth=4.0)
        ax_P.plot(C_smooth, P_w_s, color="blue", linestyle="-", linewidth=2.0)
    style(axes_P, bands_P)
    axes_P[0].set_title("Total loss")
    axes_P[len(bands_P) // 2].set_ylabel("Power loss [W]")

    # ----- Temperature panel -----
    for ax_T in axes_T:
        ax_T.plot(C_smooth, T_wo_s, color="red", linestyle="-", linewidth=4.0)
        ax_T.plot(C_smooth, T_w_s, color="blue", linestyle="-", linewidth=2.0)
    axes_T[0].set_title("Hot spot temperature")
    axes_T[len(bands_T) // 2].set_ylabel("Temperature [°C]")

    # ----- Damage panel -----
    for ax_D in axes_D:
        ax_D.plot(C_smooth, D_wo_s, color="red", linestyle="-", linewidth=4.0)
        ax_D.plot(C_smooth, D_w_s, color="blue", linestyle="-", linewidth=2.0)
    axes_D[0].set_title("Cumulative damage")
    axes_D[len(bands_D) // 2].set_ylabel("Damage [-]")

    # hide x tick labels on every row except the very last
    all_axes = axes_P + axes_T + axes_D
    for ax in all_axes[:-1]:
        ax.tick_params(labelbottom=False)

    all_axes[-1].set_xlim(100, 80)
    all_axes[-1].set_xlabel("Capacitance relative to initial value [%]")

    handles = [Patch(facecolor="red", label="Capacitance fixed"),
               Patch(facecolor="blue", label="Capacitance drifting")]

    axes_P[0].legend(handles=handles, ncol=2, loc="lower center",
                     bbox_to_anchor=(0.5, 1.075), frameon=True,
                     handlelength=2, markerscale=1.0, borderpad=0.5,
                     columnspacing=1, labelspacing=0.25)

    fig.savefig(f"{figures_dir}/{filename}", dpi=600, bbox_inches="tight")
    plt.close(fig)

def plot_transformer_power_and_temp(df_without, df_with, figures_dir, filename, bands_P=None, bands_T=None, bands_D=None, band_variable=0.1):
    """
    Three stacked panels tracing the transformer chain against capacitance drift:

        panel 1 : total loss P_T
        panel 2 : winding hot spot temperature Theta_H

    Red is the capacitance fixed scenario, blue the capacitance drifting one,
    following the convention of the filter figures.
    """

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
    # Total loss
    # ----------------------------------------#

    P_wo = df_without["P_T"].values[:len(Ds_wo)]
    P_w = df_with["P_T"].values[:len(Ds_w)]

    P_wo_s = smooth(Dm_wo, P_wo)
    P_w_s = align(P_wo_s, smooth(Dm_w, P_w))

    # ----------------------------------------#
    # Hot spot temperature
    # ----------------------------------------#

    T_wo = df_without["Theta_H"].values[:len(Ds_wo)]
    T_w = df_with["Theta_H"].values[:len(Ds_w)]

    T_wo_s = smooth(Dm_wo, T_wo)
    T_w_s = align(T_wo_s, smooth(Dm_w, T_w))

    # ----------------------------------------#
    # Cumulative damage
    # ----------------------------------------#

    D_wo = df_without["D_cum_trafo"].values[:len(Ds_wo)]
    D_w = df_with["D_cum_trafo"].values[:len(Ds_w)]

    D_wo_s = smooth(Dm_wo, D_wo)
    D_w_s = smooth(Dm_w, D_w)          # not aligned: both start from zero damage

    if bands_P is None:
        bands_P = auto_band(P_wo_s, P_w_s, band_variable)
    if bands_T is None:
        bands_T = auto_band(T_wo_s, T_w_s, band_variable)
    if bands_D is None:
        bands_D = auto_band(D_wo_s, D_w_s, band_variable)

    # ----------------------------------------#
    # Figure
    # ----------------------------------------#

    fig = plt.figure(figsize=(6.4, 4.8 * 2 * 0.5))
    gs = fig.add_gridspec(2, 1, hspace=0.2)

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

    axes_P, share_ax = make_axes(gs[0], bands_P, None)
    axes_T, share_ax = make_axes(gs[1], bands_T, share_ax)
    #axes_D, share_ax = make_axes(gs[2], bands_D, share_ax)

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

    # ----- Loss panel -----
    for ax_P in axes_P:
        ax_P.plot(C_smooth, P_wo_s, color="red", linestyle="-", linewidth=2.0)
        ax_P.plot(C_smooth, P_w_s, color="blue", linestyle="-", linewidth=2.0)
    style(axes_P, bands_P)
    axes_P[0].set_title("(a) Total losses")
    axes_P[len(bands_P) // 2].set_ylabel("Power loss [W]")

    # ----- Temperature panel -----
    for ax_T in axes_T:
        ax_T.plot(C_smooth, T_wo_s, color="red", linestyle="-", linewidth=2.0)
        ax_T.plot(C_smooth, T_w_s, color="blue", linestyle="-", linewidth=2.0)
    axes_T[0].set_title("(b) Hot spot temperature")
    axes_T[0].axhline(110, color="black", linestyle="--", linewidth=1.5)
    axes_T[0].text(99.5, 110.25, r"$\Theta_{ref}$", color="black", ha="left", va="bottom", fontsize=16)
    axes_T[len(bands_T) // 2].set_ylabel("Temperature [°C]")

    # hide x tick labels on every row except the very last
    all_axes = axes_P + axes_T
    for ax in all_axes[:-1]:
        ax.tick_params(labelbottom=False)

    all_axes[-1].set_xlim(100, 80)
    all_axes[-1].set_xlabel("Capacitance relative to initial value [%]")

    handles = [Patch(facecolor="red", label="Capacitance fixed"),
               Patch(facecolor="blue", label="Capacitance drifting")]

    axes_P[0].legend(handles=handles, ncol=2, loc="lower center",
                     bbox_to_anchor=(0.5, 1.125), frameon=True,
                     handlelength=2, markerscale=1.0, borderpad=0.5,
                     columnspacing=1, labelspacing=0.25)

    fig.savefig(f"{figures_dir}/{filename}", dpi=600, bbox_inches="tight")
    plt.close(fig)

def plot_transformer_power_and_temp_new(df_without, df_with, figures_dir, filename,bands_I=None, bands_P=None, bands_T=None,band_variable=0.1):
    """
    Three stacked panels tracing the transformer chain against capacitance drift:

        panel 1 : transformer input current, taken as the grid-side inductor
                  current I_L2, harmonic solid and fundamental dashed
        panel 2 : total loss P_T
        panel 3 : winding hot spot temperature Theta_H

    Red is the capacitance fixed scenario, blue the capacitance drifting one,
    following the convention of the filter figures. The harmonic and fundamental
    line styles are labelled inside the current panel, so the legend above the
    figure carries only the two scenarios.
    """

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
    # Transformer input current, taken as I_L2
    # ----------------------------------------#

    fundamental_wo = df_without["I_L2_fundamental"].values[:len(Ds_wo)]
    total_wo = df_without["I_L2_total"].values[:len(Ds_wo)]
    fundamental_w = df_with["I_L2_fundamental"].values[:len(Ds_w)]
    total_w = df_with["I_L2_total"].values[:len(Ds_w)]

    harmonics_wo = np.sqrt(total_wo ** 2 - fundamental_wo ** 2)
    harmonics_w = np.sqrt(total_w ** 2 - fundamental_w ** 2)

    harmonics_wo_s = smooth(Dm_wo, harmonics_wo)
    harmonics_w_s = align(harmonics_wo_s, smooth(Dm_w, harmonics_w))

    fundamental_wo_s = smooth(Dm_wo, fundamental_wo)
    fundamental_w_s = align(fundamental_wo_s, smooth(Dm_w, fundamental_w))

    # ----------------------------------------#
    # Total loss
    # ----------------------------------------#

    P_wo = df_without["P_T"].values[:len(Ds_wo)]
    P_w = df_with["P_T"].values[:len(Ds_w)]

    P_wo_s = smooth(Dm_wo, P_wo)
    P_w_s = align(P_wo_s, smooth(Dm_w, P_w))

    # ----------------------------------------#
    # Hot spot temperature
    # ----------------------------------------#

    T_wo = df_without["Theta_H"].values[:len(Ds_wo)]
    T_w = df_with["Theta_H"].values[:len(Ds_w)]

    T_wo_s = smooth(Dm_wo, T_wo)
    T_w_s = align(T_wo_s, smooth(Dm_w, T_w))

    if bands_I is None:
        bands_I = auto_band_split(harmonics_wo_s, harmonics_w_s,
                                  fundamental_wo_s, fundamental_w_s, band_variable)
    if bands_P is None:
        bands_P = auto_band(P_wo_s, P_w_s, band_variable)
    if bands_T is None:
        bands_T = auto_band(T_wo_s, T_w_s, band_variable)

    # ----------------------------------------#
    # Figure
    # ----------------------------------------#

    fig = plt.figure(figsize=(6.4, 4.8 * 3 * 0.5 * 0.9))
    gs = fig.add_gridspec(3, 1, hspace=0.25)

    def make_axes(gs_cell, bands, share_ax, equal_heights=True):
        if equal_heights:
            heights = [1.0] * len(bands)
        else:
            heights = [b[1] - b[0] for b in bands][::-1]
        sub = gs_cell.subgridspec(len(bands), 1, height_ratios=heights, hspace=0.1)
        axes = []
        for row in range(len(bands)):
            ax = fig.add_subplot(sub[row], sharex=share_ax) if share_ax is not None else fig.add_subplot(sub[row])
            axes.append(ax)
            if share_ax is None:
                share_ax = ax
        return axes, share_ax

    axes_I, share_ax = make_axes(gs[0], bands_I, None)
    axes_P, share_ax = make_axes(gs[1], bands_P, share_ax)
    axes_T, share_ax = make_axes(gs[2], bands_T, share_ax)

    def style(axes, bands):
        for row in range(len(bands)):
            ax = axes[row]
            lo, hi = bands[len(bands) - 1 - row]
            ax.set_ylim(lo, hi)
            ax.ticklabel_format(axis="y", useOffset=False, style="plain")
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3, prune="both"))
            if row > 0:
                ax.spines["top"].set_visible(False)
            if row < len(bands) - 1:
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(bottom=False)

    # ----- Current panel -----
    for ax_I in axes_I:
        ax_I.plot(C_smooth, harmonics_wo_s, color="red", linestyle="-", linewidth=2.0)
        ax_I.plot(C_smooth, harmonics_w_s, color="blue", linestyle="-", linewidth=2.0)
        ax_I.plot(C_smooth, fundamental_wo_s, color="red", linestyle="--", linewidth=2.0)
        ax_I.plot(C_smooth, fundamental_w_s, color="blue", linestyle="--", linewidth=2.0)
    style(axes_I, bands_I)
    axes_I[0].set_title("(a) Current")
    #axes_I[len(bands_I) // 2].set_ylabel("Current [A]")

    # line-style key drawn inside the current panel
    style_handles = [Line2D([], [], color="black", linestyle="-", label="Harmonic"),
                     Line2D([], [], color="black", linestyle="--", label="Fundamental")]

    top = max(ax.get_position().y1 for ax in axes_I)
    bot = min(ax.get_position().y0 for ax in axes_I)
    left = min(ax.get_position().x0 for ax in axes_I)
    right = max(ax.get_position().x1 for ax in axes_I)

    fig.legend(handles=style_handles, ncol=2, loc="center",
               bbox_to_anchor=(0.5 * (left + right), 0.5 * (top + bot)),
               bbox_transform=fig.transFigure, frameon=False,
               handlelength=2, borderpad=0.4, columnspacing=1,
               labelspacing=0.25)
    # ----- Loss panel -----
    for ax_P in axes_P:
        ax_P.plot(C_smooth, P_wo_s, color="red", linestyle="-", linewidth=2.0)
        ax_P.plot(C_smooth, P_w_s, color="blue", linestyle="-", linewidth=2.0)
    style(axes_P, bands_P)
    axes_P[0].set_title("(b) Total losses")
    #axes_P[len(bands_P) // 2].set_ylabel("Power loss [W]")

    # ----- Temperature panel -----
    for ax_T in axes_T:
        ax_T.plot(C_smooth, T_wo_s, color="red", linestyle="-", linewidth=2.0)
        ax_T.plot(C_smooth, T_w_s, color="blue", linestyle="-", linewidth=2.0)
    style(axes_T, bands_T)
    axes_T[0].set_title("(c) Hot spot temperature")
    #axes_T[0].axhline(110, color="black", linestyle="--", linewidth=1.5)
    #axes_T[0].text(99.5, 110.25, r"$\Theta_{ref}$", color="black", ha="left", va="bottom", fontsize=16)
    #axes_T[len(bands_T) // 2].set_ylabel("Temperature [°C]")

    # hide x tick labels on every row except the very last
    all_axes = axes_I + axes_P + axes_T
    for ax in all_axes[:-1]:
        ax.tick_params(labelbottom=False)

    all_axes[-1].set_xlim(100, 80)
    all_axes[-1].set_xlabel("Capacitance relative to initial value [%]",fontsize=16)

    def panel_ylabel(axes, text, x=-0.02):
        top = max(ax.get_position().y1 for ax in axes)
        bot = min(ax.get_position().y0 for ax in axes)
        fig.text(x, 0.5 * (top + bot), text, rotation=90,
                 va="center", ha="center")

    panel_ylabel(axes_I, "Current [A]")
    panel_ylabel(axes_P, "Power loss [W]")
    panel_ylabel(axes_T, "Temperature [°C]")

    handles = [Patch(facecolor="red", label="Capacitance fixed"),
               Patch(facecolor="blue", label="Capacitance drifting")]

    axes_I[0].legend(handles=handles, ncol=2, loc="lower center",
                     bbox_to_anchor=(0.5, 1.275), frameon=True,
                     handlelength=2, markerscale=1.0, borderpad=0.5,
                     columnspacing=1, labelspacing=0.25)

    fig.savefig(f"{figures_dir}/{filename}", dpi=600, bbox_inches="tight")
    plt.close(fig)

######################################
# Unreliability / B10 Plot — Transformer
######################################

def plot_unreliability_trafo(figures_dir, filename, x_max=21):

    def Monte_carlo_frame_building_trafo():

        number_of_samples = 100000
        normal_distribution = 0.01

        # ----------------------------------------#
        # Transformer — without drift
        # ----------------------------------------#

        rng = np.random.default_rng(42)

        L_eq_trafo_years_without = death_year(df_without["D_cum_trafo"])
        Theta_H_eq_without = Calculation_functions.equivalent_temperature_transformer(L_eq_years=L_eq_trafo_years_without, Theta_ref=trafo_specs["Theta_ref"], L_normal=trafo_specs["L_normal"])
        Theta_H_samples_without = Calculation_functions.normal_distribution_function(Theta_H_eq_without, normal_distribution, number_of_samples, rng)
        Theta_ref_samples_without = Calculation_functions.normal_distribution_function(trafo_specs["Theta_ref"], normal_distribution, number_of_samples, rng)
        L_normal_samples_without = Calculation_functions.normal_distribution_function(trafo_specs["L_normal"],normal_distribution,number_of_samples, rng)
        F_AA_MC_without = Calculation_functions.calculate_transformer_ageing_acceleration(Theta_H=Theta_H_samples_without, Theta_ref=Theta_ref_samples_without)
        Lifetime_trafo_MC_without_drift = L_normal_samples_without / (F_AA_MC_without * 365.0 * 24.0)  # [years]

        rng = np.random.default_rng(42)

        # ----------------------------------------#
        # Transformer — with drift
        # ----------------------------------------#

        L_eq_trafo_years_with = death_year(df_with["D_cum_trafo"])
        Theta_H_eq_with = Calculation_functions.equivalent_temperature_transformer(L_eq_years=L_eq_trafo_years_with,Theta_ref=trafo_specs["Theta_ref"],L_normal=trafo_specs["L_normal"])
        Theta_H_samples_with = Calculation_functions.normal_distribution_function(Theta_H_eq_with, normal_distribution,number_of_samples, rng)
        Theta_ref_samples_with = Calculation_functions.normal_distribution_function(trafo_specs["Theta_ref"],normal_distribution,number_of_samples, rng)
        L_normal_samples_with = Calculation_functions.normal_distribution_function(trafo_specs["L_normal"], normal_distribution,number_of_samples, rng)
        F_AA_MC_with = Calculation_functions.calculate_transformer_ageing_acceleration(Theta_H=Theta_H_samples_with,Theta_ref=Theta_ref_samples_with)
        Lifetime_trafo_MC_with_drift = L_normal_samples_with / (F_AA_MC_with * 365.0 * 24.0)  # [years]

        # ----------------------------------------#
        # Monte Carlo dataframe
        # ----------------------------------------#

        df_MC_trafo = pd.DataFrame()
        df_MC_trafo["Lifetime_trafo_with_drift"] = Lifetime_trafo_MC_with_drift
        df_MC_trafo["Lifetime_trafo_without_drift"] = Lifetime_trafo_MC_without_drift
        df_MC_trafo["F_AA_with_drift"] = F_AA_MC_with
        df_MC_trafo["F_AA_without_drift"] = F_AA_MC_without
        df_MC_trafo["Theta_H_with_drift"] = Theta_H_samples_with
        df_MC_trafo["Theta_H_without_drift"] = Theta_H_samples_without

        return df_MC_trafo

    df_MC_trafo = Monte_carlo_frame_building_trafo()

    fig, ax = plt.subplots(figsize=(6.4, 4.8 * 0.5*0.75))

    life_wo = np.sort(df_MC_trafo["Lifetime_trafo_without_drift"].values)
    life_w = np.sort(df_MC_trafo["Lifetime_trafo_with_drift"].values)

    # empirical unreliability: fraction failed by each life value
    F = np.arange(1, len(life_wo) + 1) / len(life_wo)

    ax.plot(life_wo, F, color="red", linestyle="-", linewidth=2.0)
    ax.plot(life_w, F, color="blue", linestyle="-", linewidth=2.0)

    # B10 markers
    B10_wo = np.percentile(df_MC_trafo["Lifetime_trafo_without_drift"].values, 10)
    B10_w = np.percentile(df_MC_trafo["Lifetime_trafo_with_drift"].values, 10)

    ax.axhline(0.10, color="black", linestyle="--", linewidth=1.5)
    ax.vlines(B10_wo, 0, 0.10, color="red", linestyle="--", linewidth=1.5)
    ax.vlines(B10_w, 0, 0.10, color="blue", linestyle="--", linewidth=1.5)

    trans = ax.get_xaxis_transform()
    ax.text(B10_w + 0.2, 0.03, f"{B10_w:.2f}", color="blue",transform=trans, ha="left", va="bottom", fontsize=16)
    ax.text(B10_wo + 0.2, 0.03, f"{B10_wo:.2f}", color="red", transform=trans, ha="left", va="bottom", fontsize=16)
    ax.set_ylim(0, 0.11)
    ax.set_xlim(12, 23)
    #ax.set_title("Transformer")
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.set_xlabel("Time [years]")
    ax.set_ylabel("Unreliability [-]")
    #fig.supylabel("Unreliability [-]",x = -0.001)

    handles = [Patch(facecolor="red", label="Capacitance fixed"),
               Patch(facecolor="blue", label="Capacitance drifting")]

    ax.legend(handles=handles, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), frameon=True,
              handlelength=2, markerscale=1.0, borderpad=0.5,
              columnspacing=1, labelspacing=0.25)

    fig.savefig(f"{figures_dir}/{filename}", dpi=600, bbox_inches="tight")
    plt.close(fig)

def plot_F_HL_STR_transformer(df_without, df_with, figures_dir, filename, bands=None,band_variable=0.1):

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

    total_wo = df_without["F_HL_STR"].values[:len(Ds_wo)]
    total_w = df_with["F_HL_STR"].values[:len(Ds_w)]

    total_wo_s = smooth(Dm_wo, total_wo)
    total_w_s = align(total_wo_s, smooth(Dm_w, total_w))

    if bands is None:
        bands = auto_band(total_wo_s, total_w_s, band_variable)

    heights = [b[1] - b[0] for b in bands][::-1]

    fig, axes = plt.subplots(len(bands), 1, sharex=True, figsize=(6.4, 4.8*0.5),gridspec_kw={"height_ratios": heights, "hspace": 0.1},squeeze=False)
    axes = [a[0] for a in axes]

    for row in range(len(bands)):

        ax = axes[row]
        lo, hi = bands[len(bands) - 1 - row]

        ax.plot(C_smooth, total_wo_s, color="red", linestyle="-", linewidth=2.0)
        ax.plot(C_smooth, total_w_s, color="blue", linestyle="-", linewidth=2.0)
        ax.set_ylim(lo, hi)
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")

        if row > 0:
            ax.spines["top"].set_visible(False)
        if row < len(bands) - 1:
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(bottom=False)

    axes[-1].set_xlim(100, 80)
    axes[-1].set_xlabel("Capacitance relative to initial value [%]",fontsize=16)
    fig.supylabel("FHL STR [-]",fontsize=16)

    handles = [Patch(facecolor="red", label="Capacitance fixed"),
               Patch(facecolor="blue", label="Capacitance drifting")]

    axes[0].legend(handles=handles, ncol=2, loc="lower center",
                   bbox_to_anchor=(0.5, 1.02), frameon=True,
                   handlelength=2, markerscale=1.0, borderpad=0.5,
                   columnspacing=1, labelspacing=0.25)

    fig.savefig(f"{figures_dir}/{filename}", dpi=600, bbox_inches="tight")
    plt.close(fig)

#plot_transformer_all_stages(df_without, df_with, figures_dir, filename="9_transformner.png")
#plot_transformer_power_and_temp(df_without, df_with, figures_dir, filename="9a_transformner.png")
#plot_transformer_power_and_temp_new(df_without, df_with, figures_dir, filename="9a_transformner.pdf",bands_I=((6.5,9.5), (839.29-0.2, 839.29+0.2)))
#plot_unreliability_trafo(figures_dir, filename="10_unreliability_trafo.pdf")
#plot_F_HL_STR_transformer(df_without, df_with, figures_dir, filename="12_F_HL_STR_transformer.png")

######################################
# Harmonic loss factor, full and truncated
######################################



Folder = "Dist_0.0"

Without_drift_dir = ROOT / f"OAPE_results/Results/Without_drift/{Folder}"
With_drift_dir = ROOT / f"OAPE_results/Results/With_drift/{Folder}"
figures_dir = ROOT / "OAPE_results/Figures"

df_without = pd.read_parquet(f"Results/With_drift/{Folder}/summary_without_drift.parquet")
df_with = pd.read_parquet(f"Results/With_drift/{Folder}/summary_with_drift.parquet")

Folder = "Dist_0.0_till_50"

df_without_50 = pd.read_parquet(f"Results/With_drift/{Folder}/summary_without_drift.parquet")
df_with_50 = pd.read_parquet(f"Results/With_drift/{Folder}/summary_with_drift.parquet")

Folder = "Dist_0.0_till_180"

df_without_180 = pd.read_parquet(f"Results/With_drift/{Folder}/summary_without_drift.parquet")
df_with_180 = pd.read_parquet(f"Results/With_drift/{Folder}/summary_with_drift.parquet")


def plot_F_HL_transformer(df_without, df_with,df_without_50, df_with_50,df_without_180, df_with_180,figures_dir, filename,bands_fixed=None, bands_drift=None,band_variable=0.1):
    """
    Harmonic loss factor of the transformer against capacitance drift, with the
    summation truncated at the fiftieth, the one hundred and eightieth and the
    five hundredth order.

    The top panel holds the capacitance fixed at its design value and the lower
    panel lets it drift. Red is the fiftieth order, blue the one hundred and
    eightieth and black the five hundredth. The first two sit at unity and do
    not move, since the content the drift adds lies above the reach of harmonic
    instrumentation.
    """

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

    # ----------------------------------------#
    # Capacitance fixed
    # ----------------------------------------#

    fixed_50_s = smooth(Dm_wo, df_without_50["F_HL"].values[:len(Ds_wo)])
    fixed_180_s = smooth(Dm_wo, df_without_180["F_HL"].values[:len(Ds_wo)])
    fixed_500_s = smooth(Dm_wo, df_without["F_HL"].values[:len(Ds_wo)])

    # ----------------------------------------#
    # Capacitance drifting
    # ----------------------------------------#

    drift_50_s = smooth(Dm_w, df_with_50["F_HL"].values[:len(Ds_w)])
    drift_180_s = smooth(Dm_w, df_with_180["F_HL"].values[:len(Ds_w)])
    drift_500_s = smooth(Dm_w, df_with["F_HL"].values[:len(Ds_w)])

    if bands_fixed is None:
        bands_fixed = auto_band(np.concatenate([fixed_50_s, fixed_180_s]),
                                fixed_500_s, band_variable)
    if bands_drift is None:
        bands_drift = auto_band(np.concatenate([drift_50_s, drift_180_s]),
                                drift_500_s, band_variable)

    # ----------------------------------------#
    # Figure
    # ----------------------------------------#

    colour_50 = "red"
    colour_180 = "blue"
    colour_500 = "black"

    fig = plt.figure(figsize=(6.4, 4.8 * 2 * 0.5 * 0.75))
    gs = fig.add_gridspec(2, 1, hspace=0.25)

    def make_axes(gs_cell, bands, share_ax, equal_heights=True):
        if equal_heights:
            heights = [1.0] * len(bands)
        else:
            heights = [b[1] - b[0] for b in bands][::-1]
        sub = gs_cell.subgridspec(len(bands), 1, height_ratios=heights, hspace=0.1)
        axes = []
        for row in range(len(bands)):
            ax = fig.add_subplot(sub[row], sharex=share_ax) if share_ax is not None else fig.add_subplot(sub[row])
            axes.append(ax)
            if share_ax is None:
                share_ax = ax
        return axes, share_ax

    axes_fixed, share_ax = make_axes(gs[0], bands_fixed, None)
    axes_drift, share_ax = make_axes(gs[1], bands_drift, share_ax)

    def style(axes, bands):
        for row in range(len(bands)):
            ax = axes[row]
            lo, hi = bands[len(bands) - 1 - row]
            ax.set_ylim(lo, hi)
            ax.ticklabel_format(axis="y", useOffset=False, style="plain")
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3, prune="both"))
            if row > 0:
                ax.spines["top"].set_visible(False)
            if row < len(bands) - 1:
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(bottom=False)

    def draw(axes, y_50, y_180, y_500):
        for ax in axes:
            ax.plot(C_smooth, y_500, color=colour_500, linestyle="-", linewidth=2.0)
            ax.plot(C_smooth, y_180, color=colour_180, linestyle="-", linewidth=2.0)
            ax.plot(C_smooth, y_50, color=colour_50, linestyle="--", linewidth=2.0)

    # ----- Capacitance fixed -----
    draw(axes_fixed, fixed_50_s, fixed_180_s, fixed_500_s)
    style(axes_fixed, bands_fixed)
    axes_fixed[0].set_title("(a) Capacitance fixed")
    #axes_fixed[len(bands_fixed) // 2].set_ylabel("$F_{HL}$ [-]")

    # ----- Capacitance drifting -----
    draw(axes_drift, drift_50_s, drift_180_s, drift_500_s)
    style(axes_drift, bands_drift)
    axes_drift[0].set_title("(b) Capacitance drifting")
    #axes_drift[len(bands_drift) // 2].set_ylabel("$F_{HL}$ [-]")

    # hide x tick labels on every row except the very last
    all_axes = axes_fixed + axes_drift
    for ax in all_axes[:-1]:
        ax.tick_params(labelbottom=False)

    all_axes[-1].set_xlim(100, 80)
    all_axes[-1].set_xlabel("Capacitance relative to initial value [%]",fontsize=16)

    fig.align_ylabels(all_axes)

    left = min(ax.get_position().x0 for ax in all_axes)
    top = max(ax.get_position().y1 for ax in all_axes)
    bot = min(ax.get_position().y0 for ax in all_axes)
    fig.text(left - 0.09, 0.5 * (top + bot), "Harmonic loss factor [-]",
             rotation=90, va="center", ha="center")

    handles = [Line2D([], [], color=colour_50, linestyle="--", linewidth=2.0,label="h ≤ 50"),
               Line2D([], [], color=colour_180, linestyle="-", linewidth=2.0,label="h ≤ 180"),
               Line2D([], [], color=colour_500, linestyle="-", linewidth=2.0,label="h ≤ 500")]

    axes_fixed[0].legend(handles=handles, ncol=3, loc="lower center",
                         bbox_to_anchor=(0.5, 1.15), frameon=True,
                         handlelength=2, markerscale=1.0, borderpad=0.5,
                         columnspacing=1, labelspacing=0.25)

    fig.savefig(f"{figures_dir}/{filename}", dpi=600, bbox_inches="tight")
    plt.close(fig)

def plot_F_HL_transformer_new(df_with, df_with_50, df_with_180,figures_dir, filename,bands_drift=None, band_variable=0.1):
    """
    Harmonic loss factor of the transformer against capacitance drift, with the
    summation truncated at the fiftieth, the one hundred and eightieth and the
    five hundredth order.

    Red is the fiftieth order, blue the one hundred and eightieth and black the
    five hundredth. The first two sit near unity and do not move, since the
    content the drift adds lies above the reach of harmonic instrumentation.
    """

    # ----------------------------------------#
    # Common x-axis
    # ----------------------------------------#

    Ds_w, De_w, ts_w, te_w = normalise(df_with["D_cycle_C"].values)
    Dm_w = 0.5 * (Ds_w + De_w)

    D_smooth = np.linspace(0.0, 1.0, 500)
    C_smooth = 100 - 20 * D_smooth

    def smooth(D_mid, y):
        return PchipInterpolator(D_mid, y, extrapolate=True)(D_smooth)

    # ----------------------------------------#
    # Capacitance drifting
    # ----------------------------------------#

    drift_50_s = smooth(Dm_w, df_with_50["F_HL"].values[:len(Ds_w)])
    drift_180_s = smooth(Dm_w, df_with_180["F_HL"].values[:len(Ds_w)])
    drift_500_s = smooth(Dm_w, df_with["F_HL"].values[:len(Ds_w)])

    if bands_drift is None:
        bands_drift = auto_band(np.concatenate([drift_50_s, drift_180_s]),
                                drift_500_s, band_variable)

    # ----------------------------------------#
    # Figure
    # ----------------------------------------#

    colour_50 = "red"
    colour_180 = "blue"
    colour_500 = "black"

    fig = plt.figure(figsize=(6.4, 4.8 * 0.5 * 0.75))
    gs = fig.add_gridspec(1, 1)

    def make_axes(gs_cell, bands, share_ax, equal_heights=True):
        if equal_heights:
            heights = [1.0] * len(bands)
        else:
            heights = [b[1] - b[0] for b in bands][::-1]
        sub = gs_cell.subgridspec(len(bands), 1, height_ratios=heights, hspace=0.1)
        axes = []
        for row in range(len(bands)):
            ax = fig.add_subplot(sub[row], sharex=share_ax) if share_ax is not None else fig.add_subplot(sub[row])
            axes.append(ax)
            if share_ax is None:
                share_ax = ax
        return axes, share_ax

    axes_drift, share_ax = make_axes(gs[0], bands_drift, None)

    def style(axes, bands):
        for row in range(len(bands)):
            ax = axes[row]
            lo, hi = bands[len(bands) - 1 - row]
            ax.set_ylim(lo, hi)
            ax.ticklabel_format(axis="y", useOffset=False, style="plain")
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3, prune="both"))
            if row > 0:
                ax.spines["top"].set_visible(False)
            if row < len(bands) - 1:
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(bottom=False)

    def draw(axes, y_50, y_180, y_500):
        for ax in axes:
            ax.plot(C_smooth, y_500, color=colour_500, linestyle="-", linewidth=2.0)
            ax.plot(C_smooth, y_180, color=colour_180, linestyle="-", linewidth=2.0)
            ax.plot(C_smooth, y_50, color=colour_50, linestyle="--", linewidth=2.0)

    draw(axes_drift, drift_50_s, drift_180_s, drift_500_s)
    style(axes_drift, bands_drift)

    # hide x tick labels on every row except the very last
    all_axes = axes_drift
    for ax in all_axes[:-1]:
        ax.tick_params(labelbottom=False)

    all_axes[-1].set_xlim(100, 80)
    all_axes[-1].set_xlabel("Capacitance relative to initial value [%]")

    fig.align_ylabels(all_axes)

    left = min(ax.get_position().x0 for ax in all_axes)
    top = max(ax.get_position().y1 for ax in all_axes)
    bot = min(ax.get_position().y0 for ax in all_axes)
    fig.text(left - 0.09, 0.5 * (top + bot), "Harmonic loss factor [-]",
             rotation=90, va="center", ha="center")

    handles = [Line2D([], [], color=colour_50, linestyle="--", linewidth=2.0, label="h ≤ 50"),
               Line2D([], [], color=colour_180, linestyle="-", linewidth=2.0, label="h ≤ 180"),
               Line2D([], [], color=colour_500, linestyle="-", linewidth=2.0, label="h ≤ 500")]

    axes_drift[0].legend(handles=handles, ncol=3, loc="lower center",
                         bbox_to_anchor=(0.5, 1.025), frameon=True,
                         handlelength=2, markerscale=1.0, borderpad=0.5,
                         columnspacing=1, labelspacing=0.25)

    fig.savefig(f"{figures_dir}/{filename}", dpi=600, bbox_inches="tight")
    plt.close(fig)

#plot_F_HL_transformer_new(df_with=df_with, df_with_50=df_with_50, df_with_180=df_with_180, figures_dir=figures_dir, filename="11_F_HL_transformer_both.pdf")

def plot_F_HL_STR_transformer_F_HL_STR_STR(df_with, df_with_50, df_with_180,figures_dir, filename,bands_drift=None, band_variable=0.1):
    """
    Harmonic loss factor of the transformer against capacitance drift, with the
    summation truncated at the fiftieth, the one hundred and eightieth and the
    five hundredth order.

    Red is the fiftieth order, blue the one hundred and eightieth and black the
    five hundredth. The first two sit near unity and do not move, since the
    content the drift adds lies above the reach of harmonic instrumentation.
    """

    # ----------------------------------------#
    # Common x-axis
    # ----------------------------------------#

    Ds_w, De_w, ts_w, te_w = normalise(df_with["D_cycle_C"].values)
    Dm_w = 0.5 * (Ds_w + De_w)

    D_smooth = np.linspace(0.0, 1.0, 500)
    C_smooth = 100 - 20 * D_smooth

    def smooth(D_mid, y):
        return PchipInterpolator(D_mid, y, extrapolate=True)(D_smooth)

    # ----------------------------------------#
    # Capacitance drifting
    # ----------------------------------------#

    drift_50_s = smooth(Dm_w, df_with_50["F_HL_STR"].values[:len(Ds_w)])
    drift_180_s = smooth(Dm_w, df_with_180["F_HL_STR"].values[:len(Ds_w)])
    drift_500_s = smooth(Dm_w, df_with["F_HL_STR"].values[:len(Ds_w)])

    if bands_drift is None:
        bands_drift = auto_band(np.concatenate([drift_50_s, drift_180_s]),
                                drift_500_s, band_variable)

    # ----------------------------------------#
    # Figure
    # ----------------------------------------#

    colour_50 = "red"
    colour_180 = "blue"
    colour_500 = "black"

    fig = plt.figure(figsize=(6.4, 4.8 * 0.5 * 0.75))
    gs = fig.add_gridspec(1, 1)

    def make_axes(gs_cell, bands, share_ax, equal_heights=True):
        if equal_heights:
            heights = [1.0] * len(bands)
        else:
            heights = [b[1] - b[0] for b in bands][::-1]
        sub = gs_cell.subgridspec(len(bands), 1, height_ratios=heights, hspace=0.1)
        axes = []
        for row in range(len(bands)):
            ax = fig.add_subplot(sub[row], sharex=share_ax) if share_ax is not None else fig.add_subplot(sub[row])
            axes.append(ax)
            if share_ax is None:
                share_ax = ax
        return axes, share_ax

    axes_drift, share_ax = make_axes(gs[0], bands_drift, None)

    def style(axes, bands):
        for row in range(len(bands)):
            ax = axes[row]
            lo, hi = bands[len(bands) - 1 - row]
            ax.set_ylim(lo, hi)
            ax.ticklabel_format(axis="y", useOffset=False, style="plain")
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3, prune="both"))
            if row > 0:
                ax.spines["top"].set_visible(False)
            if row < len(bands) - 1:
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(bottom=False)

    def draw(axes, y_50, y_180, y_500):
        for ax in axes:
            ax.plot(C_smooth, y_500, color=colour_500, linestyle="-", linewidth=2.0)
            ax.plot(C_smooth, y_180, color=colour_180, linestyle="-", linewidth=2.0)
            ax.plot(C_smooth, y_50, color=colour_50, linestyle="--", linewidth=2.0)

    draw(axes_drift, drift_50_s, drift_180_s, drift_500_s)
    style(axes_drift, bands_drift)

    # hide x tick labels on every row except the very last
    all_axes = axes_drift
    for ax in all_axes[:-1]:
        ax.tick_params(labelbottom=False)

    all_axes[-1].set_xlim(100, 80)
    all_axes[-1].set_xlabel("Capacitance relative to initial value [%]")

    fig.align_ylabels(all_axes)

    left = min(ax.get_position().x0 for ax in all_axes)
    top = max(ax.get_position().y1 for ax in all_axes)
    bot = min(ax.get_position().y0 for ax in all_axes)
    fig.text(left - 0.09, 0.5 * (top + bot), "Harmonic loss factor [-]",
             rotation=90, va="center", ha="center")

    handles = [Line2D([], [], color=colour_50, linestyle="--", linewidth=2.0, label="h ≤ 50"),
               Line2D([], [], color=colour_180, linestyle="-", linewidth=2.0, label="h ≤ 180"),
               Line2D([], [], color=colour_500, linestyle="-", linewidth=2.0, label="h ≤ 500")]

    axes_drift[0].legend(handles=handles, ncol=3, loc="lower center",
                         bbox_to_anchor=(0.5, 1.025), frameon=True,
                         handlelength=2, markerscale=1.0, borderpad=0.5,
                         columnspacing=1, labelspacing=0.25)

    fig.savefig(f"{figures_dir}/{filename}", dpi=600, bbox_inches="tight")
    plt.close(fig)

#plot_F_HL_STR_transformer_F_HL_STR_STR(df_with=df_with, df_with_50=df_with_50, df_with_180=df_with_180, figures_dir=figures_dir,filename="11_F_HL_STR_transformer_both.pdf")



def plot_lcl_drift_bode(L1, L2, C0, R3, fsw, figures_dir, filename,
                        C_ratios=(1.00, 0.95, 0.90, 0.85),
                        colors=("black", "red", "blue", "green"),
                        f_lim=(2500.0, 15000.0),
                        y_lim=(-35.0, 17.5),
                        print_values=False):
    """
    Filter transfer function |Ig/Vi| under capacitance drift.

    Plots the magnitude response of the LCL network for a set of capacitance
    ratios, marks fsw, and annotates the rise in gain there between the first
    and last ratio.

    Returns
    -------
    fr0 : float
        Resonance frequency at the design capacitance [Hz].
    delta_db : float
        Gain increase at fsw between the first and last C_ratio [dB].
    delta_percent : float
        The same increase expressed as a current rise [%].
    """

    LT = L1 + L2
    fr0 = np.sqrt(LT / (L1 * L2 * C0)) / (2.0 * np.pi)

    f = np.logspace(2, 5, 5000)
    s = 1j * 2.0 * np.pi * f

    fig, ax = plt.subplots(figsize=(6.4, 4.8 * 0.5*0.75))
    gains_at_fsw = []

    for C_ratio, color in zip(C_ratios, colors):
        C = C_ratio * C0
        numerator = 1.0 + s * R3 * C
        denominator = s**3 * L1 * L2 * C + s**2 * R3 * C * LT + s * LT
        gain_db = 20.0 * np.log10(np.abs(numerator / denominator))
        gains_at_fsw.append(np.interp(fsw, f, gain_db))
        ax.semilogx(f, gain_db, "-", color=color, linewidth=1.5,
                    label=r"$%d\%%$" % round((1.0 - C_ratio) * 100))

    g_nominal, g_degraded = gains_at_fsw[0], gains_at_fsw[-1]
    delta_db = g_degraded - g_nominal
    delta_percent = (10.0 ** (delta_db / 20.0) - 1.0) * 100.0

    ax.axvline(fsw, color="grey", linewidth=1.5, linestyle="--")
    ax.text(fsw * 0.95, y_lim[0] + 2.5, r"$f_{sw}$", color="black",horizontalalignment="right")

    ax.text(fsw * 1.05, 0.5 * (g_nominal + g_degraded)+2,r"$+%.1f$ dB" % delta_db, color="red", verticalalignment="center")

    ax.set_xlabel("Frequency [kHz]",fontsize=14)
    #ax.set_ylabel(r"$|I_g / V_i|$  [dB]")
    ax.set_ylabel(r"Filter attenuation [dB]")
    ax.set_xlim(*f_lim)
    ax.set_ylim(*y_lim)
    ticks = [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 14000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t // 1000) for t in ticks])

    ax.legend(ncol=len(C_ratios), loc="lower center",
              bbox_to_anchor=(0.5, 1.0), frameon=True,
              handlelength=2, borderpad=0.5,
              columnspacing=1, labelspacing=0.25,fontsize=13)

    ax.set_yticks([-30, -20, -10, 0, 10])


    fig.savefig(f"{figures_dir}/{filename}", dpi=600, bbox_inches="tight")
    plt.close(fig)

    if print_values:
        print("f_r0                    : %8.1f Hz" % fr0)
        print("f_sw / f_r0             : %8.2f" % (fsw / fr0))
        print("gain at f_sw, C/C0=%.2f : %8.2f dB" % (C_ratios[0], g_nominal))
        print("gain at f_sw, C/C0=%.2f : %8.2f dB" % (C_ratios[-1], g_degraded))
        print("increase                : %8.2f dB" % delta_db)
        print("equivalent current rise : %8.1f %%" % delta_percent)
        for r in C_ratios:
            print("C/C0 = %.3f  ->  f_r = %8.1f Hz" % (r, fr0 / np.sqrt(r)))

    return fr0, delta_db, delta_percent


#plot_lcl_drift_bode(L1=115e-6, L2=6.54e-6, C0=0.0001671444, R3=0.01057967, fsw=10000.0, figures_dir=figures_dir, filename="lcl_drift_bode.pdf", print_values=False)