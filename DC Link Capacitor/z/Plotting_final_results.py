from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import weibull_min

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Calculation_functions import Calculation_functions_class
Calculation_functions = Calculation_functions_class()

plt.rcParams.update({"font.size": 15, "font.family": "Times New Roman", "axes.labelsize": 15, "axes.titlesize": 15,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15})

def a_Current_per_capacitor():

    pf_values = [1, 0, 0.9, -0.9, 0.8, -0.8, 0.7, -0.7, 0.6, -0.6,
                 0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1]

    Cap_current_dict = {}


    # --- Fill dictionary from simulations ---
    for i, pf in enumerate(pf_values, start=1):

        base_folder = Path("Final_results") / f"Simulation_{i}"


        df = pd.read_parquet(base_folder / "Dataframes" / "df.parquet")

        if pf < 0:
            key_prefix = f"pf__{abs(pf)}"   # inductive
        else:
            key_prefix = f"pf_{pf}"         # capacitive

        Cap_current_dict[f"{key_prefix}_Icap"] = float(df["I_per_cap"].mean())

        del df

    # --- Add synthetic inductive values at pf = 0 and pf = 1 ---
    for pf in (1, 0):
        cap_key = f"pf_{pf}_Icap"
        ind_key = f"pf__{pf}_Icap"

        if cap_key in Cap_current_dict and ind_key not in Cap_current_dict:
            Cap_current_dict[ind_key] = Cap_current_dict[cap_key]

    # ---- Extract pf values ----
    pf_abs = []
    I_cap_list = []
    is_inductive_list = []

    for key in Cap_current_dict:
        if key.startswith("pf__"):
            pf_str = key.split("__")[1].replace("_Icap", "")
            is_inductive = True
        else:
            pf_str = key.split("pf_")[1].replace("_Icap", "")
            is_inductive = False

        pf_abs.append(float(pf_str))
        I_cap_list.append(Cap_current_dict[key])
        is_inductive_list.append(is_inductive)

    pf_abs = np.array(pf_abs)
    I_cap_arr = np.array(I_cap_list)
    is_inductive_arr = np.array(is_inductive_list)

    idx = np.argsort(pf_abs)
    pf_abs = pf_abs[idx]
    I_cap_arr = I_cap_arr[idx]
    is_inductive_arr = is_inductive_arr[idx]

    ind = is_inductive_arr
    cap = ~is_inductive_arr

    # --------------------------------------
    # MAIN PLOT
    # --------------------------------------
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(pf_abs[cap], I_cap_arr[cap], "-",  marker="o",
             label="Capacitive", linewidth=2.5, markersize=10)

    plt.plot(pf_abs[ind], I_cap_arr[ind], "--", marker="o",
             label="Inductive")

    plt.xlabel("Power factor [-]")
    plt.ylabel("Current [A]")
    plt.xlim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Final_results/Figures/a_Current_per_capacitor.pdf")

    # --------------------------------------
    # REMOVE pf = 0 and pf = 1
    # --------------------------------------
    unique_pf = np.unique(pf_abs)
    valid_pf = unique_pf[(unique_pf > 0) & (unique_pf < 1)]

    # --------------------------------------
    # Δ CURRENT (inductive − capacitive)
    # --------------------------------------
    delta_I_cap = []

    for pf_val in valid_pf:
        mask = (pf_abs == pf_val)

        I_vals = I_cap_arr[mask]
        ind_vals = is_inductive_arr[mask]

        I_ind = I_vals[ind_vals][0]
        I_cap = I_vals[~ind_vals][0]

        delta_I_cap.append(I_ind - I_cap)

    delta_I_cap = np.array(delta_I_cap)

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(valid_pf, delta_I_cap, "-o", linewidth=2.5, markersize=10)
    plt.xlabel("Power factor [-]")
    plt.ylabel("Δ Current [A]\n(inductive - capacitive)")
    plt.grid(True)
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig("Final_results/Figures/aa_Capacitor_current_difference.pdf")
    plt.close()

def b_DC_ripple_RMS_Current():

    pf_values = [1, 0, 0.9, -0.9, 0.8, -0.8, 0.7, -0.7, 0.6, -0.6,
                 0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1]

    Cap_current_dict = {}


    # --- Fill dictionary from simulations ---
    for i, pf in enumerate(pf_values, start=1):

        base_folder = Path("Final_results") / f"Simulation_{i}"


        df = pd.read_parquet(base_folder / "Dataframes" / "df.parquet")

        if pf < 0:
            key_prefix = f"pf__{abs(pf)}"   # inductive
        else:
            key_prefix = f"pf_{pf}"         # capacitive

        Cap_current_dict[f"{key_prefix}_Icap"] = float(df["Idcl"].mean())

        del df

    # --- Add synthetic inductive values at pf = 0 and pf = 1 ---
    for pf in (1, 0):
        cap_key = f"pf_{pf}_Icap"
        ind_key = f"pf__{pf}_Icap"

        if cap_key in Cap_current_dict and ind_key not in Cap_current_dict:
            Cap_current_dict[ind_key] = Cap_current_dict[cap_key]

    # ---- Extract pf values ----
    pf_abs = []
    I_cap_list = []
    is_inductive_list = []

    for key in Cap_current_dict:
        if key.startswith("pf__"):
            pf_str = key.split("__")[1].replace("_Icap", "")
            is_inductive = True
        else:
            pf_str = key.split("pf_")[1].replace("_Icap", "")
            is_inductive = False

        pf_abs.append(float(pf_str))
        I_cap_list.append(Cap_current_dict[key])
        is_inductive_list.append(is_inductive)

    pf_abs = np.array(pf_abs)
    I_cap_arr = np.array(I_cap_list)
    is_inductive_arr = np.array(is_inductive_list)

    idx = np.argsort(pf_abs)
    pf_abs = pf_abs[idx]
    I_cap_arr = I_cap_arr[idx]
    is_inductive_arr = is_inductive_arr[idx]

    ind = is_inductive_arr
    cap = ~is_inductive_arr

    # --------------------------------------
    # MAIN PLOT
    # --------------------------------------
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(pf_abs[cap], I_cap_arr[cap], "-",  marker="o",
             label="Capacitive", linewidth=2.5, markersize=10)

    plt.plot(pf_abs[ind], I_cap_arr[ind], "--", marker="o",
             label="Inductive")

    plt.xlabel("Power factor [-]")
    plt.ylabel("Current [A]")
    plt.xlim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Final_results/Figures/b_DC_ripple_RMS_Current.pdf")

def c_Capacitor_core_temperature():

    pf_values = [1, 0, 0.9, -0.9, 0.8, -0.8, 0.7, -0.7, 0.6, -0.6,
                 0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1]

    Cap_current_dict = {}


    # --- Fill dictionary from simulations ---
    for i, pf in enumerate(pf_values, start=1):

        base_folder = Path("Final_results") / f"Simulation_{i}"


        df = pd.read_parquet(base_folder / "Dataframes" / "df.parquet")

        if pf < 0:
            key_prefix = f"pf__{abs(pf)}"   # inductive
        else:
            key_prefix = f"pf_{pf}"         # capacitive

        Cap_current_dict[f"{key_prefix}_Icap"] = float(df["T_core"].mean()) - 273.15

        del df

    # --- Add synthetic inductive values at pf = 0 and pf = 1 ---
    for pf in (1, 0):
        cap_key = f"pf_{pf}_Icap"
        ind_key = f"pf__{pf}_Icap"

        if cap_key in Cap_current_dict and ind_key not in Cap_current_dict:
            Cap_current_dict[ind_key] = Cap_current_dict[cap_key]

    # ---- Extract pf values ----
    pf_abs = []
    I_cap_list = []
    is_inductive_list = []

    for key in Cap_current_dict:
        if key.startswith("pf__"):
            pf_str = key.split("__")[1].replace("_Icap", "")
            is_inductive = True
        else:
            pf_str = key.split("pf_")[1].replace("_Icap", "")
            is_inductive = False

        pf_abs.append(float(pf_str))
        I_cap_list.append(Cap_current_dict[key])
        is_inductive_list.append(is_inductive)

    pf_abs = np.array(pf_abs)
    I_cap_arr = np.array(I_cap_list)
    is_inductive_arr = np.array(is_inductive_list)

    idx = np.argsort(pf_abs)
    pf_abs = pf_abs[idx]
    I_cap_arr = I_cap_arr[idx]
    is_inductive_arr = is_inductive_arr[idx]

    ind = is_inductive_arr
    cap = ~is_inductive_arr

    # --------------------------------------
    # MAIN PLOT
    # --------------------------------------
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(pf_abs[cap], I_cap_arr[cap], "-",  marker="o",
             label="Capacitive", linewidth=2.5, markersize=10)

    plt.plot(pf_abs[ind], I_cap_arr[ind], "--", marker="o",
             label="Inductive")

    plt.xlabel("Power factor [-]")
    plt.ylabel("Temperature [°C]")
    plt.xlim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Final_results/Figures/c_Capacitor_core_temperature.pdf")

def d_Voltage_per_capacitor():

    pf_values = [1, 0, 0.9, -0.9, 0.8, -0.8, 0.7, -0.7, 0.6, -0.6,
                 0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1]

    Cap_current_dict = {}


    # --- Fill dictionary from simulations ---
    for i, pf in enumerate(pf_values, start=1):

        base_folder = Path("Final_results") / f"Simulation_{i}"


        df = pd.read_parquet(base_folder / "Dataframes" / "df.parquet")

        if pf < 0:
            key_prefix = f"pf__{abs(pf)}"   # inductive
        else:
            key_prefix = f"pf_{pf}"         # capacitive

        Cap_current_dict[f"{key_prefix}_Icap"] = float(df["V_per_cap"].mean())

        del df

    # --- Add synthetic inductive values at pf = 0 and pf = 1 ---
    for pf in (1, 0):
        cap_key = f"pf_{pf}_Icap"
        ind_key = f"pf__{pf}_Icap"

        if cap_key in Cap_current_dict and ind_key not in Cap_current_dict:
            Cap_current_dict[ind_key] = Cap_current_dict[cap_key]

    # ---- Extract pf values ----
    pf_abs = []
    I_cap_list = []
    is_inductive_list = []

    for key in Cap_current_dict:
        if key.startswith("pf__"):
            pf_str = key.split("__")[1].replace("_Icap", "")
            is_inductive = True
        else:
            pf_str = key.split("pf_")[1].replace("_Icap", "")
            is_inductive = False

        pf_abs.append(float(pf_str))
        I_cap_list.append(Cap_current_dict[key])
        is_inductive_list.append(is_inductive)

    pf_abs = np.array(pf_abs)
    I_cap_arr = np.array(I_cap_list)
    is_inductive_arr = np.array(is_inductive_list)

    idx = np.argsort(pf_abs)
    pf_abs = pf_abs[idx]
    I_cap_arr = I_cap_arr[idx]
    is_inductive_arr = is_inductive_arr[idx]

    ind = is_inductive_arr
    cap = ~is_inductive_arr

    # --------------------------------------
    # MAIN PLOT
    # --------------------------------------
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(pf_abs[cap], I_cap_arr[cap], "-",  marker="o",
             label="Capacitive", linewidth=2.5, markersize=10)

    plt.plot(pf_abs[ind], I_cap_arr[ind], "--", marker="o",
             label="Inductive")

    plt.xlabel("Power factor [-]")
    plt.ylabel("Voltage [V]")
    plt.xlim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Final_results/Figures/d_Voltage_per_capacitor.pdf")

def e_Total_lifetime():

    pf_values = [1, 0, 0.9, -0.9, 0.8, -0.8, 0.7, -0.7, 0.6, -0.6,
                 0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1]

    Cap_current_dict = {}


    # --- Fill dictionary from simulations ---
    for i, pf in enumerate(pf_values, start=1):

        base_folder = Path("Final_results") / f"Simulation_{i}"


        df = pd.read_parquet(base_folder / "Dataframes" / "df.parquet")

        if pf < 0:
            key_prefix = f"pf__{abs(pf)}"   # inductive
        else:
            key_prefix = f"pf_{pf}"         # capacitive

        Cap_current_dict[f"{key_prefix}_Icap"] = df["L_tot"][0]

        del df

    # --- Add synthetic inductive values at pf = 0 and pf = 1 ---
    for pf in (1, 0):
        cap_key = f"pf_{pf}_Icap"
        ind_key = f"pf__{pf}_Icap"

        if cap_key in Cap_current_dict and ind_key not in Cap_current_dict:
            Cap_current_dict[ind_key] = Cap_current_dict[cap_key]

    # ---- Extract pf values ----
    pf_abs = []
    I_cap_list = []
    is_inductive_list = []

    for key in Cap_current_dict:
        if key.startswith("pf__"):
            pf_str = key.split("__")[1].replace("_Icap", "")
            is_inductive = True
        else:
            pf_str = key.split("pf_")[1].replace("_Icap", "")
            is_inductive = False

        pf_abs.append(float(pf_str))
        I_cap_list.append(Cap_current_dict[key])
        is_inductive_list.append(is_inductive)

    pf_abs = np.array(pf_abs)
    I_cap_arr = np.array(I_cap_list)
    is_inductive_arr = np.array(is_inductive_list)

    idx = np.argsort(pf_abs)
    pf_abs = pf_abs[idx]
    I_cap_arr = I_cap_arr[idx]
    is_inductive_arr = is_inductive_arr[idx]

    ind = is_inductive_arr
    cap = ~is_inductive_arr

    # --------------------------------------
    # MAIN PLOT
    # --------------------------------------
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(pf_abs[cap], I_cap_arr[cap], "-",  marker="o",
             label="Capacitive", linewidth=2.5, markersize=10)

    plt.plot(pf_abs[ind], I_cap_arr[ind], "--", marker="o",
             label="Inductive")

    plt.xlabel("Power factor [-]")
    plt.ylabel("Lifetime [Years]")
    plt.xlim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Final_results/Figures/e_Total_lifetime.pdf")

def f_Inverter_phase_current():

    pf_values = [1, 0, 0.9, -0.9, 0.8, -0.8, 0.7, -0.7, 0.6, -0.6,
                 0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1]

    Cap_current_dict = {}


    # --- Fill dictionary from simulations ---
    for i, pf in enumerate(pf_values, start=1):

        base_folder = Path("Final_results") / f"Simulation_{i}"


        df = pd.read_parquet(base_folder / "Dataframes" / "df.parquet")

        if pf < 0:
            key_prefix = f"pf__{abs(pf)}"   # inductive
        else:
            key_prefix = f"pf_{pf}"         # capacitive

        Cap_current_dict[f"{key_prefix}_Icap"] = float(df["Is"].mean())

        del df

    # --- Add synthetic inductive values at pf = 0 and pf = 1 ---
    for pf in (1, 0):
        cap_key = f"pf_{pf}_Icap"
        ind_key = f"pf__{pf}_Icap"

        if cap_key in Cap_current_dict and ind_key not in Cap_current_dict:
            Cap_current_dict[ind_key] = Cap_current_dict[cap_key]

    # ---- Extract pf values ----
    pf_abs = []
    I_cap_list = []
    is_inductive_list = []

    for key in Cap_current_dict:
        if key.startswith("pf__"):
            pf_str = key.split("__")[1].replace("_Icap", "")
            is_inductive = True
        else:
            pf_str = key.split("pf_")[1].replace("_Icap", "")
            is_inductive = False

        pf_abs.append(float(pf_str))
        I_cap_list.append(Cap_current_dict[key])
        is_inductive_list.append(is_inductive)

    pf_abs = np.array(pf_abs)
    I_cap_arr = np.array(I_cap_list)
    is_inductive_arr = np.array(is_inductive_list)

    idx = np.argsort(pf_abs)
    pf_abs = pf_abs[idx]
    I_cap_arr = I_cap_arr[idx]
    is_inductive_arr = is_inductive_arr[idx]

    ind = is_inductive_arr
    cap = ~is_inductive_arr

    # --------------------------------------
    # MAIN PLOT
    # --------------------------------------
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(pf_abs[cap], I_cap_arr[cap], "-",  marker="o",
             label="Capacitive", linewidth=2.5, markersize=10)

    plt.plot(pf_abs[ind], I_cap_arr[ind], "--", marker="o",
             label="Inductive")

    plt.xlabel("Power factor [-]")
    plt.ylabel("Current [A]")
    plt.xlim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Final_results/Figures/f_Inverter_phase_current.pdf")

    # --------------------------------------
    # REMOVE pf = 0 and pf = 1
    # --------------------------------------
    unique_pf = np.unique(pf_abs)
    valid_pf = unique_pf[(unique_pf > 0) & (unique_pf < 1)]

    # --------------------------------------
    # Δ CURRENT (inductive − capacitive)
    # --------------------------------------
    delta_I_cap = []

    for pf_val in valid_pf:
        mask = (pf_abs == pf_val)

        I_vals = I_cap_arr[mask]
        ind_vals = is_inductive_arr[mask]

        I_ind = I_vals[ind_vals][0]
        I_cap = I_vals[~ind_vals][0]

        delta_I_cap.append(I_ind - I_cap)

    delta_I_cap = np.array(delta_I_cap)

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(valid_pf, delta_I_cap, "-o", linewidth=2.5, markersize=10)
    plt.xlabel("Power factor [-]")
    plt.ylabel("Δ Current [A]\n(inductive - capacitive)")
    plt.grid(True)
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig("Final_results/Figures/f1_Inverter_phase_current_difference.pdf")
    plt.close()



a_Current_per_capacitor()
b_DC_ripple_RMS_Current()
c_Capacitor_core_temperature()
d_Voltage_per_capacitor()
e_Total_lifetime()
f_Inverter_phase_current()