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
    plt.savefig("Final_results/Figures/a_Current_per_capacitor.png")

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
    #plt.savefig("Final_results/Figures/aa_Capacitor_current_difference.png")
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
    plt.savefig("Final_results/Figures/b_DC_ripple_RMS_Current.png")

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
    plt.savefig("Final_results/Figures/c_Capacitor_core_temperature.png")

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
    plt.savefig("Final_results/Figures/d_Voltage_per_capacitor.png")

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
    plt.savefig("Final_results/Figures/e_Total_lifetime.png")

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
    plt.savefig("Final_results/Figures/f_Inverter_phase_current.png")

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
    #plt.savefig("Final_results/Figures/f1_Inverter_phase_current_difference.png")
    plt.close()

def g_Total_power_losses():

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

        Cap_current_dict[f"{key_prefix}_Icap"] = float(df["P_losses"].mean())

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
    plt.ylabel("Power Losses [W]")
    plt.xlim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Final_results/Figures/g1_Total_power_losses.png")

def g_Leak_power_losses():

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

        Cap_current_dict[f"{key_prefix}_Icap"] = float(df["P_leak"].mean())

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
    plt.ylabel("Power Losses [W]")
    plt.xlim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Final_results/Figures/g2_Leak_power_losses.png")

def g_Ripple_power_losses():

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

        Cap_current_dict[f"{key_prefix}_Icap"] = float(df["P_ripple"].mean())

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
    plt.ylabel("Power Losses [W]")
    plt.xlim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Final_results/Figures/g3_Ripple_power_losses.png")


#a_Current_per_capacitor()
#b_DC_ripple_RMS_Current()
#c_Capacitor_core_temperature()
#d_Voltage_per_capacitor()
#e_Total_lifetime()
#f_Inverter_phase_current()
#g_Total_power_losses()
#g_Leak_power_losses()
#g_Ripple_power_losses()

def Apparent_power():

    S_base = 1e6
    delta = 0.019150e6

    S_min = S_base - delta
    S_max = S_base + delta
    n = 86400
    block_size = 900  # 15 minutes (900 seconds)

    num_blocks = n // block_size

    # Generate one value per block
    block_values = S_base + np.random.normal(0, delta/3, num_blocks)
    block_values = np.clip(block_values, S_min, S_max)

    # Repeat each value for 900 samples
    S_in = np.repeat(block_values, block_size)

    time = np.arange(n)

    # -------- Create dataframe --------
    df = pd.DataFrame({
        "time": time,
        "S_in": S_in
    })

    df.to_parquet("Apparent_power.parquet")

    # -------- Plot --------
    plt.figure(figsize=(6.4, 4.8*0.625))
    plt.plot(time/3600, S_in, label="Apparent power")
    plt.xlabel("Time [Hours]")
    plt.ylabel("Power [W]")
    plt.xlim(min(time/3600), max(time/3600))
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig("Conference paper images/Apparent_power.pdf")
    plt.close()

#Apparent_power()





def combined_capacitor_summary_plot():

    pf_values = [1, 0, 0.9, -0.9, 0.8, -0.8, 0.7, -0.7, 0.6, -0.6,
        0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1]

    def prepare_metric_dict(column_name, agg="mean", offset=0.0):
        metric_dict = {}

        # --- Fill dictionary from simulations ---
        for i, pf in enumerate(pf_values, start=1):
            base_folder = Path("Final_results") / f"Simulation_{i}"
            df = pd.read_parquet(base_folder / "Dataframes" / "df.parquet")

            if pf < 0:
                key_prefix = f"pf__{abs(pf)}"   # inductive
            else:
                key_prefix = f"pf_{pf}"         # capacitive

            if agg == "mean":
                value = float(df[column_name].mean()) + offset
            elif agg == "first":
                value = float(df[column_name].iloc[0]) + offset
            else:
                raise ValueError(f"Unsupported aggregation: {agg}")

            metric_dict[f"{key_prefix}_val"] = value
            del df

        # --- Add synthetic inductive values at pf = 0 and pf = 1 ---
        for pf in (1, 0):
            cap_key = f"pf_{pf}_val"
            ind_key = f"pf__{pf}_val"

            if cap_key in metric_dict and ind_key not in metric_dict:
                metric_dict[ind_key] = metric_dict[cap_key]

        return metric_dict

    def extract_arrays(metric_dict):
        pf_abs = []
        values = []
        is_inductive = []

        for key in metric_dict:
            if key.startswith("pf__"):
                pf_str = key.split("__")[1].replace("_val", "")
                inductive = True
            else:
                pf_str = key.split("pf_")[1].replace("_val", "")
                inductive = False

            pf_abs.append(float(pf_str))
            values.append(metric_dict[key])
            is_inductive.append(inductive)

        pf_abs = np.array(pf_abs)
        values = np.array(values)
        is_inductive = np.array(is_inductive)

        idx = np.argsort(pf_abs)
        pf_abs = pf_abs[idx]
        values = values[idx]
        is_inductive = is_inductive[idx]

        cap = ~is_inductive
        ind = is_inductive

        return pf_abs, values, cap, ind

    # --- Prepare data for each quantity ---
    current_dict = prepare_metric_dict("I_per_cap", agg="mean")
    loss_dict = prepare_metric_dict("P_losses", agg="mean")
    temp_dict = prepare_metric_dict("T_core", agg="mean", offset=-273.15)
    life_dict = prepare_metric_dict("L_tot", agg="first")

    pf_abs, current_vals, cap, ind = extract_arrays(current_dict)
    _, loss_vals, _, _ = extract_arrays(loss_dict)
    _, temp_vals, _, _ = extract_arrays(temp_dict)
    _, life_vals, _, _ = extract_arrays(life_dict)

    # --- Create subplot figure ---
    fig, axes = plt.subplots(4, 1, figsize=(6.4, 4.8*4*0.4), sharex=True)

    # (a) Current per capacitor
    axes[0].plot(pf_abs[cap], current_vals[cap], "-", marker="o",label="Capacitive", linewidth=2.5, markersize=10)
    axes[0].plot(pf_abs[ind], current_vals[ind], "--", marker="o",label="Inductive")
    axes[0].set_title("(a) Average DC-link capacitor RMS ripple current")
    axes[0].set_ylabel("Current [A]")
    axes[0].set_xlim(0, 1)
    axes[0].grid(True)

    # (b) Total power losses per capacitor
    axes[1].plot(pf_abs[cap], loss_vals[cap], "-", marker="o",linewidth=2.5, markersize=10, color="tab:blue")
    axes[1].plot(pf_abs[ind], loss_vals[ind], "--", marker="o", color="tab:orange")
    axes[1].set_title("(b) Average DC-link capacitor total power losses")
    axes[1].set_ylabel("Power losses [W]")
    axes[1].set_xlim(0, 1)
    axes[1].grid(True)

    # (c) Core temperature per capacitor
    axes[2].plot(pf_abs[cap], temp_vals[cap], "-", marker="o",linewidth=2.5, markersize=10,color="tab:blue")
    axes[2].plot(pf_abs[ind], temp_vals[ind], "--", marker="o",color="tab:orange")
    axes[2].set_title("(c) Average DC-link capacitor core temperature")
    axes[2].set_ylabel("Temperature [°C]")
    axes[2].set_xlim(0, 1)
    axes[2].grid(True)

    # (d) Lifetime
    axes[3].plot(pf_abs[cap], life_vals[cap], "-", marker="o",linewidth=2.5, markersize=10, color="tab:blue")
    axes[3].plot(pf_abs[ind], life_vals[ind], "--", marker="o",color="tab:orange")

    axes[3].set_title("(d) DC-link capacitor lifetime")
    axes[3].set_ylabel("Lifetime [years]")
    axes[3].set_xlabel("Power factor [-]")
    axes[3].set_xlim(0, 1)
    axes[3].grid(True)

    # ---- figure legend at top ----
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=True
    )

    plt.tight_layout(rect=[0, 0, 1, 0.9575])
    plt.savefig("Final_results/Figures/capacitor_summary_4x1.pdf")
    plt.close()

combined_capacitor_summary_plot()


def temp_vs_lifetime_V_0_9():

    temp_array = np.arange(10, 60, .1)

    temp_env_vs_lifetime_dict = {}
    temp_core_vs_lifetime_dict = {}

    # --- Fill dictionary from simulations ---
    for i, temp in enumerate(temp_array, start=1):

        base_folder = Path("Final_results_temp") / f"Simulation_{i}"
        df = pd.read_parquet(base_folder / "Dataframes" / "df.parquet")

        lifetime = df["L_tot"][0]
        temp_core = df["T_core"].mean() - 273.15

        # --- Rule 1: if core temp > 84°C lifetime = 0 ---
        if temp_core > 84:
            lifetime = 0

        temp_env_vs_lifetime_dict[temp] = lifetime
        temp_core_vs_lifetime_dict[temp_core] = lifetime

    # --- Rule 2: propagate zero values ---
    for temp, lifetime in temp_env_vs_lifetime_dict.items():

        if lifetime == 0:
            temp_env_vs_lifetime_dict[temp] = 0

    # --- Convert dictionaries to sorted arrays ---
    env_temp = np.array(sorted(temp_env_vs_lifetime_dict.keys()))
    life_env = np.array([temp_env_vs_lifetime_dict[t] for t in env_temp])

    core_temp = np.array(sorted(temp_core_vs_lifetime_dict.keys()))
    life_core = np.array([temp_core_vs_lifetime_dict[t] for t in core_temp])


    # --- Find threshold locations (first point where lifetime becomes zero) ---
    env_zero_idx = np.where(life_env == 0)[0]
    core_zero_idx = np.where(life_core == 0)[0]

    env_threshold = env_temp[env_zero_idx[0]] if len(env_zero_idx) > 0 else None
    core_threshold = core_temp[core_zero_idx[0]] if len(core_zero_idx) > 0 else None

    # --- Create subplots ---
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.8), sharey=True)

    # ---- (a) Ambient temperature vs lifetime ----


    env_temp = np.concatenate((env_temp, np.arange(env_temp[-1] + 0.1, 100.1, 0.1)))
    life_env = np.concatenate((life_env, np.zeros(402)))




    axes[0].plot(env_temp, life_env, linewidth=2.5, color="tab:blue")
    axes[0].set_title("(a) Ambient temperature impact of lifetime")
    axes[0].set_xlabel("Ambient temperature [°C]")
    axes[0].set_ylabel("Lifetime [years]")
    axes[0].grid(True)
    axes[0].set_xlim(10, 100)

    if env_threshold is not None:
        axes[0].axvline(env_threshold, color="red", linewidth=2.5)
        y_mid = 0.6 * (axes[0].get_ylim()[0] + axes[0].get_ylim()[1])
        axes[0].text(env_threshold + 1, y_mid, "Threshold", color="red",
                     rotation=0, va="center", ha="left")

    # ---- (b) Core temperature vs lifetime ----
    axes[1].plot(core_temp, life_core, linewidth=2.5, color="tab:blue")
    axes[1].set_title("(b) Core temperature impact on lifetime")
    axes[1].set_xlabel("Core temperature [°C]")
    axes[1].set_ylabel("Lifetime [years]")
    axes[1].grid(True)
    axes[1].set_xlim(10, 100)

    if core_threshold is not None:
        axes[1].axvline(core_threshold, color="red", linewidth=2.5)
        y_mid = 0.6 * (axes[1].get_ylim()[0] + axes[1].get_ylim()[1])
        axes[1].text(core_threshold +1 , y_mid, "Threshold", color="red",
                     rotation=0, va="center", ha="left")

    plt.tight_layout()
    plt.savefig("Final_results_temp/Figures/temp_vs_lifetime_2x1.png")
    plt.close()


#temp_vs_lifetime_V_0_9()




def temp_vs_lifetime_V_0_7():

    temp_array = np.arange(0, 150, .1)

    temp_env_vs_lifetime_dict = {}
    temp_core_vs_lifetime_dict = {}

    # --- Fill dictionary from simulations ---
    for i, temp in enumerate(temp_array, start=1):

        base_folder = Path("Final_results_temp/V_0.7") / f"Simulation_{i}"
        df = pd.read_parquet(base_folder / "Dataframes" / "df.parquet")

        lifetime = df["L_tot"][0]
        temp_core = df["T_core"].mean() - 273.15

        # --- Rule 1: if core temp > 84°C lifetime = 0 ---
        if temp_core > 99:
            lifetime = 0

        temp_env_vs_lifetime_dict[temp] = lifetime
        temp_core_vs_lifetime_dict[temp_core] = lifetime

    # --- Rule 2: propagate zero values ---
    for temp, lifetime in temp_env_vs_lifetime_dict.items():

        if lifetime == 0:
            temp_env_vs_lifetime_dict[temp] = 0

    #print("temp_env_vs_lifetime_dict", temp_env_vs_lifetime_dict)


    #print("temp_core_vs_lifetime_dict",temp_core_vs_lifetime_dict)

    # --- Convert dictionaries to sorted arrays ---
    env_temp = np.array(sorted(temp_env_vs_lifetime_dict.keys()))
    life_env = np.array([temp_env_vs_lifetime_dict[t] for t in env_temp])

    core_temp = np.array(sorted(temp_core_vs_lifetime_dict.keys()))
    life_core = np.array([temp_core_vs_lifetime_dict[t] for t in core_temp])


    # --- Find threshold locations (first point where lifetime becomes zero) ---
    env_zero_idx = np.where(life_env == 0)[0]
    core_zero_idx = np.where(life_core == 0)[0]

    env_threshold = env_temp[env_zero_idx[0]] if len(env_zero_idx) > 0 else None
    core_threshold = core_temp[core_zero_idx[0]] if len(core_zero_idx) > 0 else None

    # --- Create subplots ---
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.8), sharey=True)

    # ---- (a) Ambient temperature vs lifetime ----

    #env_temp = np.concatenate((env_temp, np.arange(env_temp[-1] + 0.1, 100.1, 0.1)))
    #life_env = np.concatenate((life_env, np.zeros(402)))

    axes[0].plot(env_temp, life_env, linewidth=2.5, color="tab:blue")
    axes[0].set_title("(a) Ambient temperature impact of lifetime")
    axes[0].set_xlabel("Ambient temperature [°C]")
    axes[0].set_ylabel("Lifetime [years]")
    axes[0].grid(True)
    axes[0].set_xlim(0, 120)

    if env_threshold is not None:
        axes[0].axvline(env_threshold, color="red", linewidth=2.5)
        y_mid = 0.525 * (axes[0].get_ylim()[0] + axes[0].get_ylim()[1])
        axes[0].text(env_threshold + 1, y_mid, "Capacitor\nfailure", color="red",
                     rotation=0, va="center", ha="left", linespacing=1.2)

    # ---- (b) Core temperature vs lifetime ----
    axes[1].plot(core_temp, life_core, linewidth=2.5, color="tab:blue")
    axes[1].set_title("(b) Core temperature impact on lifetime")
    axes[1].set_xlabel("Core temperature [°C]")
    axes[1].set_ylabel("Lifetime [years]")
    axes[1].grid(True)
    axes[1].set_xlim(0, 120)

    if core_threshold is not None:
        axes[1].axvline(core_threshold, color="red", linewidth=2.5)
        y_mid = 0.525 * (axes[1].get_ylim()[0] + axes[1].get_ylim()[1])
        axes[1].text(core_threshold + 1, y_mid, "Capacitor\nfailure", color="red",
                     rotation=0, va="center", ha="left", linespacing=1.2)


    plt.tight_layout()
    plt.savefig("Final_results_temp/Figures/temp_vs_lifetime_2x1_V_0_7.pdf")
    plt.close()


#temp_vs_lifetime_V_0_7()




def capacitor_electro_thermal_timeseries():

    df = pd.read_parquet(f"Final_results/Simulation_1/Dataframes/df.parquet")
    I_per_cap = df["I_per_cap"].to_numpy()
    P_losses = df["P_losses"].to_numpy()
    T_core = df["T_core"].to_numpy() - 273.15
    time = np.arange(len(I_per_cap))

    fig, axs = plt.subplots(3, 1, figsize=(6.4, 4.8*0.5*3), sharex=True)

    # --- Current ---
    axs[0].plot(time/3600, I_per_cap, linewidth=2)
    axs[0].set_title("(a) RMS ripple current per DC-link capacitor")
    axs[0].set_ylabel("Current [A]")
    axs[0].grid(True)
    axs[0].set_xlim(min(time/3600), max(time/3600))

    # --- Losses ---
    axs[1].plot(time/3600, P_losses, linewidth=2)
    axs[1].set_title("(b) DC-link capacitor total power losses")
    axs[1].set_ylabel("Power [W]")
    axs[1].grid(True)
    axs[1].set_xlim(min(time/3600), max(time/3600))

    # --- Temperature ---
    axs[2].plot(time/3600, T_core, linewidth=2)
    axs[2].set_title("(c) DC-link capacitor core temperature")
    axs[2].set_ylabel("Temperature [°C]")
    axs[2].set_xlabel("Time [Hours]")
    axs[2].grid(True)
    axs[2].set_xlim(min(time/3600), max(time/3600))

    plt.tight_layout()
    plt.savefig("Final_results/Figures/capacitor_electro_thermal_timeseries.pdf")
    plt.close()

#capacitor_electro_thermal_timeseries()




