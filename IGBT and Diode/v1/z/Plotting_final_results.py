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

def IGBT_and_Diode_Current():

    pf_values = [1, 0, 0.9, -0.9, 0.8, -0.8, 0.7, -0.7, 0.6, -0.6,
                 0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1]

    Power_losses_dict = {}

    # --- Fill dictionary from simulations ---
    for i, pf in enumerate(pf_values, start=1):
        folder = f"z/Final_results/Simulation_{i}/df_electrical_loss"

        df = Calculation_functions.read_datafames(df_dir=folder)

        if pf < 0:
            key_prefix = f"pf__{abs(pf)}"   # inductive (negative)
        else:
            key_prefix = f"pf_{pf}"         # capacitive (positive or 0)

        Power_losses_dict[f"{key_prefix}_P_I"] = float(df["is_I"].mean())
        Power_losses_dict[f"{key_prefix}_P_D"] = float(df["is_D"].mean())

        del df  # free memory

    # --- Add synthetic inductive values at pf = 0 and pf = 1 ---
    for pf in (1, 0):
        cap_prefix = f"pf_{pf}"   # capacitive key
        ind_prefix = f"pf__{pf}"  # inductive key

        cap_key_I = f"{cap_prefix}_P_I"
        cap_key_D = f"{cap_prefix}_P_D"
        ind_key_I = f"{ind_prefix}_P_I"
        ind_key_D = f"{ind_prefix}_P_D"

        # If we have capacitive data but no inductive for this pf, copy it
        if cap_key_I in Power_losses_dict and ind_key_I not in Power_losses_dict:
            Power_losses_dict[ind_key_I] = Power_losses_dict[cap_key_I]
            Power_losses_dict[ind_key_D] = Power_losses_dict[cap_key_D]


    # ---- Extract pf values and losses for plotting ----
    pf_abs = []
    P_I_list = []
    P_D_list = []
    is_inductive_list = []  # True if pf__ (inductive), False if pf_ (capacitive)

    # Work only on P_I keys to avoid duplication
    for key in Power_losses_dict:
        if key.endswith("_P_I"):
            if key.startswith("pf__"):
                # negative (inductive)
                pf_str = key.split("__")[1].replace("_P_I", "")
                is_inductive = True
            else:
                # positive or zero (capacitive)
                pf_str = key.split("pf_")[1].replace("_P_I", "")
                is_inductive = False

            pf_val_abs = float(pf_str)
            pf_abs.append(pf_val_abs)
            is_inductive_list.append(is_inductive)

            # IGBT
            P_I_list.append(Power_losses_dict[key])

            # Diode
            diode_key = key.replace("_P_I", "_P_D")
            P_D_list.append(Power_losses_dict[diode_key])

    # Convert to numpy arrays and sort by |pf|
    pf_abs = np.array(pf_abs)
    P_I_arr = np.array(P_I_list)
    P_D_arr = np.array(P_D_list)
    is_inductive_arr = np.array(is_inductive_list)

    idx = np.argsort(pf_abs)
    pf_abs = pf_abs[idx]
    P_I_arr = P_I_arr[idx]
    P_D_arr = P_D_arr[idx]
    is_inductive_arr = is_inductive_arr[idx]

    # ---- Split by inductive vs capacitive (by key prefix, not numeric sign) ----
    ind = is_inductive_arr
    cap = ~is_inductive_arr

    plt.figure(figsize=(6.4, 4.8))

    # IGBT
    plt.plot(pf_abs[cap], P_I_arr[cap], "-",  marker="o", color="blue",   label="IGBT (capacitive)", linewidth=2.5, markersize=10)
    plt.plot(pf_abs[ind], P_I_arr[ind], "--", marker="o", color="orange", label="IGBT (inductive)")

    # Diode
    plt.plot(pf_abs[cap], P_D_arr[cap], "-",  marker="s", color="green",  label="Diode (capacitive)", linewidth=2.5, markersize=10)
    plt.plot(pf_abs[ind], P_D_arr[ind], "--", marker="s", color="red",    label="Diode (inductive)")


    plt.xlabel("Power factor [-]")
    plt.ylabel("Current [A]")
    #plt.title("Average power losses vs power factor")
    plt.xlim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Final_results/Figures/IGBT_and_Diode_current.pdf")

    # --------------------------------------
    # REMOVE pf = 0 and pf = 1
    # --------------------------------------
    unique_pf = np.unique(pf_abs)

    # Keep only pf between 0 and 1 (exclusive)
    valid_mask = (unique_pf > 0) & (unique_pf < 1)
    unique_pf_filtered = unique_pf[valid_mask]

    # --------------------------------------
    # Combined subplot figure for Δ currents
    # --------------------------------------

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 4.8 * 2), sharex=True)

    # --------------------------------------
    # Δ CURRENT FOR IGBT
    # --------------------------------------
    delta_I_igbt = []

    for pf_val in unique_pf_filtered:
        mask = (pf_abs == pf_val)

        I_vals = P_I_arr[mask]
        ind_vals = is_inductive_arr[mask]

        I_ind = I_vals[ind_vals][0]
        I_cap = I_vals[~ind_vals][0]

        delta_I_igbt.append(I_ind - I_cap)

    delta_I_igbt = np.array(delta_I_igbt)

    ax1.plot(unique_pf_filtered, delta_I_igbt, "-o", linewidth=2.5, markersize=10)
    ax1.set_ylabel("IGBT Δ Current [A]\n(inductive - capacitive)")
    ax1.grid(True)
    ax1.set_xlim(0, 1)
    #ax1.set_title("Difference in IGBT and Diode Current\nInductive – Capacitive")

    # --------------------------------------
    # Δ CURRENT FOR DIODE
    # --------------------------------------
    delta_I_diode = []

    for pf_val in unique_pf_filtered:
        mask = (pf_abs == pf_val)

        I_vals_D = P_D_arr[mask]
        ind_vals = is_inductive_arr[mask]

        I_ind_D = I_vals_D[ind_vals][0]
        I_cap_D = I_vals_D[~ind_vals][0]

        delta_I_diode.append(I_ind_D - I_cap_D)

    delta_I_diode = np.array(delta_I_diode)

    ax2.plot(unique_pf_filtered, delta_I_diode, "-o", linewidth=2.5, markersize=10)
    ax2.set_xlabel("Power factor [-]")
    ax2.set_ylabel("Diode Δ Current [A]\n(inductive - capacitive)")
    ax2.grid(True)
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig("Final_results/Figures/IGBT_Diode_current_difference_subplots.pdf")
    plt.close()

# -------------------------------------------------
# Power losses
# -------------------------------------------------

def Total_power_losses():

    pf_values = [1, 0, 0.9, -0.9, 0.8, -0.8, 0.7, -0.7, 0.6, -0.6,
                 0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1]

    Power_losses_dict = {}

    # --- Fill dictionary from simulations ---
    for i, pf in enumerate(pf_values, start=1):
        folder = f"z/Final_results/Simulation_{i}/df_electrical_loss"

        df = Calculation_functions.read_datafames(df_dir=folder)

        if pf < 0:
            key_prefix = f"pf__{abs(pf)}"   # inductive (negative)
        else:
            key_prefix = f"pf_{pf}"         # capacitive (positive or 0)

        Power_losses_dict[f"{key_prefix}_P_I"] = float(df["P_I"].mean())
        Power_losses_dict[f"{key_prefix}_P_D"] = float(df["P_D"].mean())

        del df  # free memory

    # --- Add synthetic inductive values at pf = 0 and pf = 1 ---
    for pf in (1, 0):
        cap_prefix = f"pf_{pf}"   # capacitive key
        ind_prefix = f"pf__{pf}"  # inductive key

        cap_key_I = f"{cap_prefix}_P_I"
        cap_key_D = f"{cap_prefix}_P_D"
        ind_key_I = f"{ind_prefix}_P_I"
        ind_key_D = f"{ind_prefix}_P_D"

        # If we have capacitive data but no inductive for this pf, copy it
        if cap_key_I in Power_losses_dict and ind_key_I not in Power_losses_dict:
            Power_losses_dict[ind_key_I] = Power_losses_dict[cap_key_I]
            Power_losses_dict[ind_key_D] = Power_losses_dict[cap_key_D]


    # ---- Extract pf values and losses for plotting ----
    pf_abs = []
    P_I_list = []
    P_D_list = []
    is_inductive_list = []  # True if pf__ (inductive), False if pf_ (capacitive)

    # Work only on P_I keys to avoid duplication
    for key in Power_losses_dict:
        if key.endswith("_P_I"):
            if key.startswith("pf__"):
                # negative (inductive)
                pf_str = key.split("__")[1].replace("_P_I", "")
                is_inductive = True
            else:
                # positive or zero (capacitive)
                pf_str = key.split("pf_")[1].replace("_P_I", "")
                is_inductive = False

            pf_val_abs = float(pf_str)
            pf_abs.append(pf_val_abs)
            is_inductive_list.append(is_inductive)

            # IGBT
            P_I_list.append(Power_losses_dict[key])

            # Diode
            diode_key = key.replace("_P_I", "_P_D")
            P_D_list.append(Power_losses_dict[diode_key])

    # Convert to numpy arrays and sort by |pf|
    pf_abs = np.array(pf_abs)
    P_I_arr = np.array(P_I_list)
    P_D_arr = np.array(P_D_list)
    is_inductive_arr = np.array(is_inductive_list)

    idx = np.argsort(pf_abs)
    pf_abs = pf_abs[idx]
    P_I_arr = P_I_arr[idx]
    P_D_arr = P_D_arr[idx]
    is_inductive_arr = is_inductive_arr[idx]

    # ---- Split by inductive vs capacitive (by key prefix, not numeric sign) ----
    ind = is_inductive_arr
    cap = ~is_inductive_arr

    plt.figure(figsize=(6.4, 4.8))

    # IGBT
    plt.plot(pf_abs[cap], P_I_arr[cap], "-",  marker="o", color="blue",   label="IGBT (capacitive)", linewidth=2.5, markersize=10)
    plt.plot(pf_abs[ind], P_I_arr[ind], "--", marker="o", color="orange", label="IGBT (inductive)")

    # Diode
    plt.plot(pf_abs[cap], P_D_arr[cap], "-",  marker="s", color="green",  label="Diode (capacitive)", linewidth=2.5, markersize=10)
    plt.plot(pf_abs[ind], P_D_arr[ind], "--", marker="s", color="red",    label="Diode (inductive)")


    plt.xlabel("Power factor [-]")
    plt.ylabel("Power losses [W]")
    #plt.title("Average power losses vs power factor")
    plt.xlim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Final_results/Figures/Total_power_losses.pdf")

def Switching_power_losses():

    pf_values = [1, 0, 0.9, -0.9, 0.8, -0.8, 0.7, -0.7, 0.6, -0.6,
                 0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1]

    Power_losses_dict = {}

    # --- Fill dictionary from simulations ---
    for i, pf in enumerate(pf_values, start=1):
        folder = f"z/Final_results/Simulation_{i}/df_electrical_loss"

        df = Calculation_functions.read_datafames(df_dir=folder)

        if pf < 0:
            key_prefix = f"pf__{abs(pf)}"  # inductive (negative)
        else:
            key_prefix = f"pf_{pf}"  # capacitive (positive or 0)

        Power_losses_dict[f"{key_prefix}_P_I"] = float(df["P_sw_I"].mean())
        Power_losses_dict[f"{key_prefix}_P_D"] = float(df["P_sw_D"].mean())

        del df  # free memory

    # --- Add synthetic inductive values at pf = 0 and pf = 1 ---
    for pf in (1, 0):
        cap_prefix = f"pf_{pf}"  # capacitive key
        ind_prefix = f"pf__{pf}"  # inductive key

        cap_key_I = f"{cap_prefix}_P_I"
        cap_key_D = f"{cap_prefix}_P_D"
        ind_key_I = f"{ind_prefix}_P_I"
        ind_key_D = f"{ind_prefix}_P_D"

        # If we have capacitive data but no inductive for this pf, copy it
        if cap_key_I in Power_losses_dict and ind_key_I not in Power_losses_dict:
            Power_losses_dict[ind_key_I] = Power_losses_dict[cap_key_I]
            Power_losses_dict[ind_key_D] = Power_losses_dict[cap_key_D]

    # ---- Extract pf values and losses for plotting ----
    pf_abs = []
    P_I_list = []
    P_D_list = []
    is_inductive_list = []  # True if pf__ (inductive), False if pf_ (capacitive)

    # Work only on P_I keys to avoid duplication
    for key in Power_losses_dict:
        if key.endswith("_P_I"):
            if key.startswith("pf__"):
                # negative (inductive)
                pf_str = key.split("__")[1].replace("_P_I", "")
                is_inductive = True
            else:
                # positive or zero (capacitive)
                pf_str = key.split("pf_")[1].replace("_P_I", "")
                is_inductive = False

            pf_val_abs = float(pf_str)
            pf_abs.append(pf_val_abs)
            is_inductive_list.append(is_inductive)

            # IGBT
            P_I_list.append(Power_losses_dict[key])

            # Diode
            diode_key = key.replace("_P_I", "_P_D")
            P_D_list.append(Power_losses_dict[diode_key])

    # Convert to numpy arrays and sort by |pf|
    pf_abs = np.array(pf_abs)
    P_I_arr = np.array(P_I_list)
    P_D_arr = np.array(P_D_list)
    is_inductive_arr = np.array(is_inductive_list)

    idx = np.argsort(pf_abs)
    pf_abs = pf_abs[idx]
    P_I_arr = P_I_arr[idx]
    P_D_arr = P_D_arr[idx]
    is_inductive_arr = is_inductive_arr[idx]

    # ---- Split by inductive vs capacitive (by key prefix, not numeric sign) ----
    ind = is_inductive_arr
    cap = ~is_inductive_arr

    plt.figure(figsize=(6.4, 4.8))

    # IGBT
    plt.plot(pf_abs[cap], P_I_arr[cap], "-", marker="o", color="blue", label="IGBT (capacitive)", linewidth=2.5,
             markersize=10)
    plt.plot(pf_abs[ind], P_I_arr[ind], "--", marker="o", color="orange", label="IGBT (inductive)")

    # Diode
    plt.plot(pf_abs[cap], P_D_arr[cap], "-", marker="s", color="green", label="Diode (capacitive)", linewidth=2.5,
             markersize=10)
    plt.plot(pf_abs[ind], P_D_arr[ind], "--", marker="s", color="red", label="Diode (inductive)")

    plt.xlabel("Power factor [-]")
    plt.ylabel("Power losses [W]")
    #plt.title("Average switching losses vs power factor")
    plt.xlim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Final_results/Figures/Switching_power_losses.pdf")

def Conduction_power_losses():

    pf_values = [1, 0, 0.9, -0.9, 0.8, -0.8, 0.7, -0.7, 0.6, -0.6,
                 0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1]

    Power_losses_dict = {}

    # --- Fill dictionary from simulations ---
    for i, pf in enumerate(pf_values, start=1):
        folder = f"z/Final_results/Simulation_{i}/df_electrical_loss"

        df = Calculation_functions.read_datafames(df_dir=folder)

        if pf < 0:
            key_prefix = f"pf__{abs(pf)}"  # inductive (negative)
        else:
            key_prefix = f"pf_{pf}"  # capacitive (positive or 0)

        Power_losses_dict[f"{key_prefix}_P_I"] = float(df["P_con_I"].mean())
        Power_losses_dict[f"{key_prefix}_P_D"] = float(df["P_con_D"].mean())

        del df  # free memory

    # --- Add synthetic inductive values at pf = 0 and pf = 1 ---
    for pf in (1, 0):
        cap_prefix = f"pf_{pf}"  # capacitive key
        ind_prefix = f"pf__{pf}"  # inductive key

        cap_key_I = f"{cap_prefix}_P_I"
        cap_key_D = f"{cap_prefix}_P_D"
        ind_key_I = f"{ind_prefix}_P_I"
        ind_key_D = f"{ind_prefix}_P_D"

        # If we have capacitive data but no inductive for this pf, copy it
        if cap_key_I in Power_losses_dict and ind_key_I not in Power_losses_dict:
            Power_losses_dict[ind_key_I] = Power_losses_dict[cap_key_I]
            Power_losses_dict[ind_key_D] = Power_losses_dict[cap_key_D]

    # ---- Extract pf values and losses for plotting ----
    pf_abs = []
    P_I_list = []
    P_D_list = []
    is_inductive_list = []  # True if pf__ (inductive), False if pf_ (capacitive)

    # Work only on P_I keys to avoid duplication
    for key in Power_losses_dict:
        if key.endswith("_P_I"):
            if key.startswith("pf__"):
                # negative (inductive)
                pf_str = key.split("__")[1].replace("_P_I", "")
                is_inductive = True
            else:
                # positive or zero (capacitive)
                pf_str = key.split("pf_")[1].replace("_P_I", "")
                is_inductive = False

            pf_val_abs = float(pf_str)
            pf_abs.append(pf_val_abs)
            is_inductive_list.append(is_inductive)

            # IGBT
            P_I_list.append(Power_losses_dict[key])

            # Diode
            diode_key = key.replace("_P_I", "_P_D")
            P_D_list.append(Power_losses_dict[diode_key])

    # Convert to numpy arrays and sort by |pf|
    pf_abs = np.array(pf_abs)
    P_I_arr = np.array(P_I_list)
    P_D_arr = np.array(P_D_list)
    is_inductive_arr = np.array(is_inductive_list)

    idx = np.argsort(pf_abs)
    pf_abs = pf_abs[idx]
    P_I_arr = P_I_arr[idx]
    P_D_arr = P_D_arr[idx]
    is_inductive_arr = is_inductive_arr[idx]

    # ---- Split by inductive vs capacitive (by key prefix, not numeric sign) ----
    ind = is_inductive_arr
    cap = ~is_inductive_arr

    plt.figure(figsize=(6.4, 4.8))

    # IGBT
    plt.plot(pf_abs[cap], P_I_arr[cap], "-", marker="o", color="blue", label="IGBT (capacitive)", linewidth=2.5,
             markersize=10)
    plt.plot(pf_abs[ind], P_I_arr[ind], "--", marker="o", color="orange", label="IGBT (inductive)")

    # Diode
    plt.plot(pf_abs[cap], P_D_arr[cap], "-", marker="s", color="green", label="Diode (capacitive)", linewidth=2.5,
             markersize=10)
    plt.plot(pf_abs[ind], P_D_arr[ind], "--", marker="s", color="red", label="Diode (inductive)")

    plt.xlabel("Power factor [-]")
    plt.ylabel("Power losses [W]")
    #plt.title("Average Conduction losses vs power factor")

    plt.xlim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Final_results/Figures/Conduction_power_losses.pdf")

# -------------------------------------------------
# Temperature values
# -------------------------------------------------

def Temperature_igbt_diode_values():

    pf_values = [1, 0, 0.9, -0.9, 0.8, -0.8, 0.7, -0.7, 0.6, -0.6,
                 0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1]

    Temp_dict = {}

    # --- Fill dictionary from simulations ---
    for i, pf in enumerate(pf_values, start=1):
        folder = f"z/Final_results/Simulation_{i}/df_thermal"

        df = Calculation_functions.read_datafames(df_dir=folder)

        if pf < 0:
            key_prefix = f"pf__{abs(pf)}"  # inductive (negative)
        else:
            key_prefix = f"pf_{pf}"        # capacitive (positive or 0)

        # store average junction temperatures
        Temp_dict[f"{key_prefix}_T_I"] = float((df["Tj_igbt"]-273.15).mean())
        Temp_dict[f"{key_prefix}_T_D"] = float((df["Tj_diode"]-273.15).mean())

        del df  # free memory

    # --- Add synthetic inductive values at pf = 0 and pf = 1 ---
    for pf in (1, 0):
        cap_prefix = f"pf_{pf}"
        ind_prefix = f"pf__{pf}"

        cap_key_I = f"{cap_prefix}_T_I"
        cap_key_D = f"{cap_prefix}_T_D"
        ind_key_I = f"{ind_prefix}_T_I"
        ind_key_D = f"{ind_prefix}_T_D"

        if cap_key_I in Temp_dict and ind_key_I not in Temp_dict:
            Temp_dict[ind_key_I] = Temp_dict[cap_key_I]
            Temp_dict[ind_key_D] = Temp_dict[cap_key_D]

    # ---- Extract pf values and temperatures for plotting ----
    pf_abs = []
    T_I_list = []
    T_D_list = []
    is_inductive_list = []

    for key in Temp_dict:
        if key.endswith("_T_I"):
            if key.startswith("pf__"):
                pf_str = key.split("__")[1].replace("_T_I", "")
                is_inductive = True
            else:
                pf_str = key.split("pf_")[1].replace("_T_I", "")
                is_inductive = False

            pf_val_abs = float(pf_str)
            pf_abs.append(pf_val_abs)
            is_inductive_list.append(is_inductive)

            # IGBT temp
            T_I_list.append(Temp_dict[key])

            # Diode temp
            diode_key = key.replace("_T_I", "_T_D")
            T_D_list.append(Temp_dict[diode_key])

    # Convert to numpy arrays and sort by |pf|
    pf_abs = np.array(pf_abs)
    T_I_arr = np.array(T_I_list)
    T_D_arr = np.array(T_D_list)
    is_inductive_arr = np.array(is_inductive_list)

    idx = np.argsort(pf_abs)
    pf_abs = pf_abs[idx]
    T_I_arr = T_I_arr[idx]
    T_D_arr = T_D_arr[idx]
    is_inductive_arr = is_inductive_arr[idx]

    # ---- Split by inductive vs capacitive ----
    ind = is_inductive_arr
    cap = ~is_inductive_arr

    plt.figure(figsize=(6.4, 4.8))

    # IGBT temperatures
    plt.plot(pf_abs[cap], T_I_arr[cap], "-", marker="o", color="blue",
             label="IGBT (capacitive)", linewidth=2.5, markersize=10)
    plt.plot(pf_abs[ind], T_I_arr[ind], "--", marker="o", color="orange",
             label="IGBT (inductive)")

    # Diode temperatures
    plt.plot(pf_abs[cap], T_D_arr[cap], "-", marker="s", color="green",
             label="Diode (capacitive)", linewidth=2.5, markersize=10)
    plt.plot(pf_abs[ind], T_D_arr[ind], "--", marker="s", color="red",
             label="Diode (inductive)")

    plt.xlabel("Power factor [-]")
    plt.ylabel("Junction temperature [°C]")
    #plt.title("Average junction temperature vs power factor")

    plt.xlim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Final_results/Figures/Junction_temperature_vs_pf.pdf")

def Temperature_pad_sink_values():

    pf_values = [1, 0, 0.9, -0.9, 0.8, -0.8, 0.7, -0.7, 0.6, -0.6,
                 0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1]

    Temp_dict = {}

    # ===============================
    #   LOAD THERMAL DATA
    # ===============================
    for i, pf in enumerate(pf_values, start=1):

        folder = f"z/Final_results/Simulation_{i}/df_thermal"
        df = Calculation_functions.read_datafames(df_dir=folder)

        # PF prefix for capacitive or inductive
        if pf < 0:
            key_prefix = f"pf__{abs(pf)}"     # inductive (negative)
        else:
            key_prefix = f"pf_{pf}"           # capacitive (positive or 0)

        # Convert to Celsius
        T_case_C = float((df["T_case"] - 273.15).mean())
        T_sink_C = float((df["T_sink"] - 273.15).mean())

        Temp_dict[f"{key_prefix}_T_pad"] = T_case_C
        Temp_dict[f"{key_prefix}_T_sink"] = T_sink_C

        del df  # free memory

    # ========================================================
    #   SYNTHETIC INDUCTIVE POINTS AT pf = 0 AND pf = 1
    # ========================================================
    for pf in (1, 0):
        cap_prefix = f"pf_{pf}"
        ind_prefix = f"pf__{pf}"

        cap_pad = f"{cap_prefix}_T_pad"
        cap_sink = f"{cap_prefix}_T_sink"
        ind_pad = f"{ind_prefix}_T_pad"
        ind_sink = f"{ind_prefix}_T_sink"

        if cap_pad in Temp_dict and ind_pad not in Temp_dict:
            Temp_dict[ind_pad] = Temp_dict[cap_pad]
            Temp_dict[ind_sink] = Temp_dict[cap_sink]

    # ========================================================
    #   EXTRACT DATA FOR PLOTTING
    # ========================================================
    pf_abs = []
    T_pad_list = []
    T_sink_list = []
    inductive_flags = []

    for key in Temp_dict:
        if key.endswith("_T_pad"):

            # Determine pf + capacitive/inductive
            if key.startswith("pf__"):
                pf_val = -float(key.split("__")[1].replace("_T_pad", ""))
                inductive = True
            else:
                pf_val = float(key.split("pf_")[1].replace("_T_pad", ""))
                inductive = False

            pf_abs.append(abs(pf_val))
            inductive_flags.append(inductive)

            # Load temperatures
            T_pad_list.append(Temp_dict[key])
            T_sink_list.append(Temp_dict[key.replace("_T_pad", "_T_sink")])

    # Convert to numpy and sort by pf
    pf_abs = np.array(pf_abs)
    T_pad_arr = np.array(T_pad_list)
    T_sink_arr = np.array(T_sink_list)
    inductive_flags = np.array(inductive_flags)

    sort_idx = np.argsort(pf_abs)
    pf_abs = pf_abs[sort_idx]
    T_pad_arr = T_pad_arr[sort_idx]
    T_sink_arr = T_sink_arr[sort_idx]
    inductive_flags = inductive_flags[sort_idx]

    cap = ~inductive_flags
    ind = inductive_flags

    # ========================================================
    #   PLOT TEMPERATURES (TWO SUBPLOTS)
    # ========================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 4.8 * 2), sharex=True)

    # ---------- PAD TEMPERATURE (T_case) ----------
    ax1.plot(pf_abs[cap], T_pad_arr[cap], "-", marker="o", markersize=10,
             color="blue", linewidth=2.5, label="Pad (capacitive)")
    ax1.plot(pf_abs[ind], T_pad_arr[ind], "--", marker="o", markersize=6,
             color="orange", label="Pad (inductive)")

    ax1.set_ylabel("Pad Temp [°C]")
    ax1.set_title("Average pad and sink temperatures vs power factor")
    ax1.grid(True)
    ax1.legend()

    # ---------- SINK TEMPERATURE (T_sink) ----------
    ax2.plot(pf_abs[cap], T_sink_arr[cap], "-", marker="s", markersize=10,
             color="green", linewidth=2.5, label="Sink (capacitive)")
    ax2.plot(pf_abs[ind], T_sink_arr[ind], "--", marker="s", markersize=6,
             color="red", label="Sink (inductive)")

    ax2.set_xlabel("Power factor [-]")
    ax2.set_ylabel("Sink Temp [°C]")
    ax2.grid(True)
    ax2.set_xlim(0, 1)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("Final_results/Figures/Thermal_pad_and_sink_vs_pf.pdf")
    plt.close()

def Temperature_delta_t_values():
    """
    Plot average temperature swing ΔT for IGBT and Diode vs |power factor|,
    distinguishing capacitive and inductive operation.
    Data source:
      - Final_results/Simulation_i/df_lifetime_IGBT/df_IGBT_final.parquet
      - Final_results/Simulation_i/df_lifetime_Diode/df_Diode_final.parquet
    """

    pf_values = [1, 0, 0.9, -0.9, 0.8, -0.8, 0.7, -0.7, 0.6, -0.6,
                 0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1]

    DeltaT_dict = {}

    # ===============================
    #   LOAD LIFETIME ΔT DATA
    # ===============================
    for i, pf in enumerate(pf_values, start=1):

        base_folder = Path("Final_results") / f"Simulation_{i}"

        igbt_file = base_folder / "df_lifetime_IGBT" / "df_IGBT_final.parquet"
        diode_file = base_folder / "df_lifetime_Diode" / "df_Diode_final.parquet"

        df_igbt = pd.read_parquet(igbt_file)
        df_diode = pd.read_parquet(diode_file)

        # PF prefix for capacitive or inductive
        if pf < 0:
            key_prefix = f"pf__{abs(pf)}"     # inductive (negative)
        else:
            key_prefix = f"pf_{pf}"           # capacitive (positive or 0)

        # Mean ΔT for IGBT and Diode
        dT_igbt = float(df_igbt["deltaT_igbt"].mean())
        dT_diode = float(df_diode["deltaT_diode"].mean())

        DeltaT_dict[f"{key_prefix}_dT_I"] = dT_igbt
        DeltaT_dict[f"{key_prefix}_dT_D"] = dT_diode

        del df_igbt, df_diode  # free memory

    # ========================================================
    #   SYNTHETIC INDUCTIVE POINTS AT pf = 0 AND pf = 1
    # ========================================================
    for pf in (1, 0):
        cap_prefix = f"pf_{pf}"
        ind_prefix = f"pf__{pf}"

        cap_I = f"{cap_prefix}_dT_I"
        cap_D = f"{cap_prefix}_dT_D"
        ind_I = f"{ind_prefix}_dT_I"
        ind_D = f"{ind_prefix}_dT_D"

        if cap_I in DeltaT_dict and ind_I not in DeltaT_dict:
            DeltaT_dict[ind_I] = DeltaT_dict[cap_I]
            DeltaT_dict[ind_D] = DeltaT_dict[cap_D]

    # ========================================================
    #   EXTRACT DATA FOR PLOTTING
    # ========================================================
    pf_abs = []
    dT_I_list = []
    dT_D_list = []
    inductive_flags = []

    for key in DeltaT_dict:
        if key.endswith("_dT_I"):

            # Determine pf + capacitive/inductive
            if key.startswith("pf__"):
                pf_val = -float(key.split("__")[1].replace("_dT_I", ""))
                inductive = True
            else:
                pf_val = float(key.split("pf_")[1].replace("_dT_I", ""))
                inductive = False

            pf_abs.append(abs(pf_val))
            inductive_flags.append(inductive)

            # ΔT IGBT / Diode
            dT_I_list.append(DeltaT_dict[key])
            dT_D_list.append(DeltaT_dict[key.replace("_dT_I", "_dT_D")])

    # Convert to numpy and sort by |pf|
    pf_abs = np.array(pf_abs)
    dT_I_arr = np.array(dT_I_list)
    dT_D_arr = np.array(dT_D_list)
    inductive_flags = np.array(inductive_flags)

    sort_idx = np.argsort(pf_abs)
    pf_abs = pf_abs[sort_idx]
    dT_I_arr = dT_I_arr[sort_idx]
    dT_D_arr = dT_D_arr[sort_idx]
    inductive_flags = inductive_flags[sort_idx]

    cap = ~inductive_flags
    ind = inductive_flags

    # ========================================================
    #   PLOT ΔT ON ONE GRAPH
    # ========================================================
    plt.figure(figsize=(6.4, 4.8))

    # ---------- IGBT ΔT ----------
    plt.plot(pf_abs[cap], dT_I_arr[cap], "-", marker="o", markersize=10,
             color="blue", linewidth=2.5, label="IGBT (capacitive)")
    plt.plot(pf_abs[ind], dT_I_arr[ind], "--", marker="o", markersize=6,
             color="orange", label="IGBT (inductive)")

    # ---------- DIODE ΔT ----------
    plt.plot(pf_abs[cap], dT_D_arr[cap], "-", marker="s", markersize=10,
             color="green", linewidth=2.5, label="Diode (capacitive)")
    plt.plot(pf_abs[ind], dT_D_arr[ind], "--", marker="s", markersize=6,
             color="red", label="Diode (inductive)")

    # Labels and styling
    plt.xlabel("Power factor [-]")
    plt.ylabel("Temperature [°C]")
    #plt.title("Average temperature swing ΔT vs power factor")

    plt.xlim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig("Final_results/Figures/DeltaT_IGBT_Diode_vs_pf.pdf")
    plt.close()

def Temperature_t_mean_values():
    """
    Plot average temperature swing ΔT for IGBT and Diode vs |power factor|,
    distinguishing capacitive and inductive operation.
    Data source:
      - Final_results/Simulation_i/df_lifetime_IGBT/df_IGBT_final.parquet
      - Final_results/Simulation_i/df_lifetime_Diode/df_Diode_final.parquet
    """

    pf_values = [1, 0, 0.9, -0.9, 0.8, -0.8, 0.7, -0.7, 0.6, -0.6,
                 0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1]

    DeltaT_dict = {}

    # ===============================
    #   LOAD LIFETIME ΔT DATA
    # ===============================
    for i, pf in enumerate(pf_values, start=1):

        base_folder = Path("Final_results") / f"Simulation_{i}"

        igbt_file = base_folder / "df_lifetime_IGBT" / "df_IGBT_final.parquet"
        diode_file = base_folder / "df_lifetime_Diode" / "df_Diode_final.parquet"

        df_igbt = pd.read_parquet(igbt_file)
        df_diode = pd.read_parquet(diode_file)

        # PF prefix for capacitive or inductive
        if pf < 0:
            key_prefix = f"pf__{abs(pf)}"     # inductive (negative)
        else:
            key_prefix = f"pf_{pf}"           # capacitive (positive or 0)

        # Mean ΔT for IGBT and Diode
        dT_igbt = float((df_igbt["Tmean_igbt"]-273.15).mean())
        dT_diode = float((df_diode["Tmean_diode"]-273.15).mean())

        DeltaT_dict[f"{key_prefix}_dT_I"] = dT_igbt
        DeltaT_dict[f"{key_prefix}_dT_D"] = dT_diode

        del df_igbt, df_diode  # free memory

    # ========================================================
    #   SYNTHETIC INDUCTIVE POINTS AT pf = 0 AND pf = 1
    # ========================================================
    for pf in (1, 0):
        cap_prefix = f"pf_{pf}"
        ind_prefix = f"pf__{pf}"

        cap_I = f"{cap_prefix}_dT_I"
        cap_D = f"{cap_prefix}_dT_D"
        ind_I = f"{ind_prefix}_dT_I"
        ind_D = f"{ind_prefix}_dT_D"

        if cap_I in DeltaT_dict and ind_I not in DeltaT_dict:
            DeltaT_dict[ind_I] = DeltaT_dict[cap_I]
            DeltaT_dict[ind_D] = DeltaT_dict[cap_D]

    # ========================================================
    #   EXTRACT DATA FOR PLOTTING
    # ========================================================
    pf_abs = []
    dT_I_list = []
    dT_D_list = []
    inductive_flags = []

    for key in DeltaT_dict:
        if key.endswith("_dT_I"):

            # Determine pf + capacitive/inductive
            if key.startswith("pf__"):
                pf_val = -float(key.split("__")[1].replace("_dT_I", ""))
                inductive = True
            else:
                pf_val = float(key.split("pf_")[1].replace("_dT_I", ""))
                inductive = False

            pf_abs.append(abs(pf_val))
            inductive_flags.append(inductive)

            # ΔT IGBT / Diode
            dT_I_list.append(DeltaT_dict[key])
            dT_D_list.append(DeltaT_dict[key.replace("_dT_I", "_dT_D")])

    # Convert to numpy and sort by |pf|
    pf_abs = np.array(pf_abs)
    dT_I_arr = np.array(dT_I_list)
    dT_D_arr = np.array(dT_D_list)
    inductive_flags = np.array(inductive_flags)

    sort_idx = np.argsort(pf_abs)
    pf_abs = pf_abs[sort_idx]
    dT_I_arr = dT_I_arr[sort_idx]
    dT_D_arr = dT_D_arr[sort_idx]
    inductive_flags = inductive_flags[sort_idx]

    cap = ~inductive_flags
    ind = inductive_flags

    # ========================================================
    #   PLOT ΔT ON ONE GRAPH
    # ========================================================
    plt.figure(figsize=(6.4, 4.8))

    # ---------- IGBT ΔT ----------
    plt.plot(pf_abs[cap], dT_I_arr[cap], "-", marker="o", markersize=10,
             color="blue", linewidth=2.5, label="IGBT (capacitive)")
    plt.plot(pf_abs[ind], dT_I_arr[ind], "--", marker="o", markersize=6,
             color="orange", label="IGBT (inductive)")

    # ---------- DIODE ΔT ----------
    plt.plot(pf_abs[cap], dT_D_arr[cap], "-", marker="s", markersize=10,
             color="green", linewidth=2.5, label="Diode (capacitive)")
    plt.plot(pf_abs[ind], dT_D_arr[ind], "--", marker="s", markersize=6,
             color="red", label="Diode (inductive)")

    # Labels and styling
    plt.xlabel("Power factor [-]")
    plt.ylabel("Temperature [°C]")
    #plt.title("Mean temperature vs power factor")

    plt.xlim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig("Final_results/Figures/T_mean_IGBT_Diode_vs_pf.pdf")
    plt.close()

def Temperature_thermal_period_values():
    """
    Plot average temperature swing ΔT for IGBT and Diode vs |power factor|,
    distinguishing capacitive and inductive operation.
    Data source:
      - Final_results/Simulation_i/df_lifetime_IGBT/df_IGBT_final.parquet
      - Final_results/Simulation_i/df_lifetime_Diode/df_Diode_final.parquet
    """

    pf_values = [1, 0, 0.9, -0.9, 0.8, -0.8, 0.7, -0.7, 0.6, -0.6,
                 0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1]

    DeltaT_dict = {}

    # ===============================
    #   LOAD LIFETIME ΔT DATA
    # ===============================
    for i, pf in enumerate(pf_values, start=1):

        base_folder = Path("Final_results") / f"Simulation_{i}"

        igbt_file = base_folder / "df_lifetime_IGBT" / "df_IGBT_final.parquet"
        diode_file = base_folder / "df_lifetime_Diode" / "df_Diode_final.parquet"

        df_igbt = pd.read_parquet(igbt_file)
        df_diode = pd.read_parquet(diode_file)

        # PF prefix for capacitive or inductive
        if pf < 0:
            key_prefix = f"pf__{abs(pf)}"     # inductive (negative)
        else:
            key_prefix = f"pf_{pf}"           # capacitive (positive or 0)

        # Mean ΔT for IGBT and Diode
        dT_igbt = float((df_igbt["thermal_cycle_period_igbt"]).mean())
        dT_diode = float((df_diode["thermal_cycle_period_diode"]).mean())

        DeltaT_dict[f"{key_prefix}_dT_I"] = dT_igbt
        DeltaT_dict[f"{key_prefix}_dT_D"] = dT_diode

        del df_igbt, df_diode  # free memory

    # ========================================================
    #   SYNTHETIC INDUCTIVE POINTS AT pf = 0 AND pf = 1
    # ========================================================
    for pf in (1, 0):
        cap_prefix = f"pf_{pf}"
        ind_prefix = f"pf__{pf}"

        cap_I = f"{cap_prefix}_dT_I"
        cap_D = f"{cap_prefix}_dT_D"
        ind_I = f"{ind_prefix}_dT_I"
        ind_D = f"{ind_prefix}_dT_D"

        if cap_I in DeltaT_dict and ind_I not in DeltaT_dict:
            DeltaT_dict[ind_I] = DeltaT_dict[cap_I]
            DeltaT_dict[ind_D] = DeltaT_dict[cap_D]

    # ========================================================
    #   EXTRACT DATA FOR PLOTTING
    # ========================================================
    pf_abs = []
    dT_I_list = []
    dT_D_list = []
    inductive_flags = []

    for key in DeltaT_dict:
        if key.endswith("_dT_I"):

            # Determine pf + capacitive/inductive
            if key.startswith("pf__"):
                pf_val = -float(key.split("__")[1].replace("_dT_I", ""))
                inductive = True
            else:
                pf_val = float(key.split("pf_")[1].replace("_dT_I", ""))
                inductive = False

            pf_abs.append(abs(pf_val))
            inductive_flags.append(inductive)

            # ΔT IGBT / Diode
            dT_I_list.append(DeltaT_dict[key])
            dT_D_list.append(DeltaT_dict[key.replace("_dT_I", "_dT_D")])

    # Convert to numpy and sort by |pf|
    pf_abs = np.array(pf_abs)
    dT_I_arr = np.array(dT_I_list)
    dT_D_arr = np.array(dT_D_list)
    inductive_flags = np.array(inductive_flags)

    sort_idx = np.argsort(pf_abs)
    pf_abs = pf_abs[sort_idx]
    dT_I_arr = dT_I_arr[sort_idx]
    dT_D_arr = dT_D_arr[sort_idx]
    inductive_flags = inductive_flags[sort_idx]

    cap = ~inductive_flags
    ind = inductive_flags

    # ========================================================
    #   PLOT ΔT ON ONE GRAPH
    # ========================================================
    plt.figure(figsize=(6.4, 4.8))

    # ---------- IGBT ΔT ----------
    plt.plot(pf_abs[cap], dT_I_arr[cap], "-", marker="o", markersize=10,
             color="blue", linewidth=2.5, label="IGBT (capacitive)")
    plt.plot(pf_abs[ind], dT_I_arr[ind], "--", marker="o", markersize=6,
             color="orange", label="IGBT (inductive)")

    # ---------- DIODE ΔT ----------
    plt.plot(pf_abs[cap], dT_D_arr[cap], "-", marker="s", markersize=10,
             color="green", linewidth=2.5, label="Diode (capacitive)")
    plt.plot(pf_abs[ind], dT_D_arr[ind], "--", marker="s", markersize=6,
             color="red", label="Diode (inductive)")

    # Labels and styling
    plt.xlabel("Power factor [-]")
    plt.ylabel("Seconds [s]")
    #plt.title("Average thermal period vs power factor")

    plt.xlim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig("Final_results/Figures/Thermal_period_IGBT_Diode_vs_pf.pdf")
    plt.close()

# -------------------------------------------------
# Lifetime values
# -------------------------------------------------

def Lifetime_values():


    pf_values = [1, 0, 0.9, -0.9, 0.8, -0.8, 0.7, -0.7, 0.6, -0.6,
                 0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1]

    lifetime_dict = {}

    # ===============================
    #   LOAD LIFETIME DATA
    # ===============================
    for i, pf in enumerate(pf_values, start=1):

        base_folder = Path("Final_results") / f"Simulation_{i}"

        igbt_file = base_folder / "df_lifetime_IGBT" / "df_IGBT_final.parquet"
        diode_file = base_folder / "df_lifetime_Diode" / "df_Diode_final.parquet"

        df_igbt = pd.read_parquet(igbt_file)
        df_diode = pd.read_parquet(diode_file)

        # PF prefix for capacitive or inductive
        if pf < 0:
            key_prefix = f"pf__{abs(pf)}"     # inductive (negative)
        else:
            key_prefix = f"pf_{pf}"           # capacitive (positive or 0)

        # Lifetime (years) for IGBT and Diode
        life_igbt = float(df_igbt["lifetime_years_igbt_actual"].iloc[0])
        life_diode = float(df_diode["lifetime_years_diode_actual"].iloc[0])

        lifetime_dict[f"{key_prefix}_L_I"] = life_igbt
        lifetime_dict[f"{key_prefix}_L_D"] = life_diode

        del df_igbt, df_diode  # free memory

    # ========================================================
    #   SYNTHETIC INDUCTIVE POINTS AT pf = 0 AND pf = 1
    # ========================================================
    for pf in (1, 0):
        cap_prefix = f"pf_{pf}"
        ind_prefix = f"pf__{pf}"

        cap_I = f"{cap_prefix}_L_I"
        cap_D = f"{cap_prefix}_L_D"
        ind_I = f"{ind_prefix}_L_I"
        ind_D = f"{ind_prefix}_L_D"

        if cap_I in lifetime_dict and ind_I not in lifetime_dict:
            lifetime_dict[ind_I] = lifetime_dict[cap_I]
            lifetime_dict[ind_D] = lifetime_dict[cap_D]

    # ========================================================
    #   EXTRACT DATA FOR PLOTTING
    # ========================================================
    pf_abs = []
    L_I_list = []
    L_D_list = []
    inductive_flags = []

    for key in lifetime_dict:
        if key.endswith("_L_I"):

            # Determine pf + capacitive/inductive
            if key.startswith("pf__"):
                pf_val = -float(key.split("__")[1].replace("_L_I", ""))
                inductive = True
            else:
                pf_val = float(key.split("pf_")[1].replace("_L_I", ""))
                inductive = False

            pf_abs.append(abs(pf_val))
            inductive_flags.append(inductive)

            # Lifetime IGBT / Diode
            L_I_list.append(lifetime_dict[key])
            L_D_list.append(lifetime_dict[key.replace("_L_I", "_L_D")])

    # Convert to numpy and sort by |pf|
    pf_abs = np.array(pf_abs)
    L_I_arr = np.array(L_I_list)
    L_D_arr = np.array(L_D_list)
    inductive_flags = np.array(inductive_flags)

    sort_idx = np.argsort(pf_abs)
    pf_abs = pf_abs[sort_idx]
    L_I_arr = L_I_arr[sort_idx]
    L_D_arr = L_D_arr[sort_idx]
    inductive_flags = inductive_flags[sort_idx]

    cap = ~inductive_flags
    ind = inductive_flags

    # ========================================================
    #   PLOT LIFETIME ON ONE GRAPH
    # ========================================================
    plt.figure(figsize=(6.4, 4.8))

    # ---------- IGBT lifetime ----------
    plt.plot(pf_abs[cap], L_I_arr[cap], "-", marker="o", markersize=10,
             color="blue", linewidth=2.5, label="IGBT (capacitive)")
    plt.plot(pf_abs[ind], L_I_arr[ind], "--", marker="o", markersize=6,
             color="orange", label="IGBT (inductive)")

    # ---------- Diode lifetime ----------
    plt.plot(pf_abs[cap], L_D_arr[cap], "-", marker="s", markersize=10,
             color="green", linewidth=2.5, label="Diode (capacitive)")
    plt.plot(pf_abs[ind], L_D_arr[ind], "--", marker="s", markersize=6,
             color="red", label="Diode (inductive)")

    # Labels and styling
    plt.xlabel("Power factor [-]")
    plt.ylabel("Lifetime [years]")
    #plt.title("Estimated lifetime vs power factor")
    #plt.yscale("log")  # <<< Make y-axis logarithmic

    plt.xlim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig("Final_results/Figures/Lifetime_IGBT_Diode_vs_pf.pdf")
    plt.close()

    # ========================================================
    #   SWITCH BOTTLENECK LIFETIME (NEW PLOT)
    # ========================================================

    # Element-wise min: bottleneck lifetime of the switch
    L_switch_arr = np.minimum(L_I_arr, L_D_arr)

    # Optional: which device is limiting (not required for plot,
    # but useful if you want to inspect later)
    bottleneck_device = np.where(L_I_arr <= L_D_arr, "IGBT", "Diode")

    # New figure for switch lifetime
    plt.figure(figsize=(6.4, 4.8))

    # Capacitive and inductive shown separately if you like
    plt.plot(pf_abs[cap], L_switch_arr[cap], "-o", linewidth=2.5,
             label="Switch lifetime (capacitive)")
    plt.plot(pf_abs[ind], L_switch_arr[ind], "--o", linewidth=2.0,
             label="Switch lifetime (inductive)")

    plt.xlabel("Power factor [-]")
    plt.ylabel("Lifetime [years]")
    #plt.title("Switch bottleneck lifetime vs power factor")
    plt.xlim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Final_results/Figures/Lifetime_Switch_vs_pf.pdf")
    plt.close()

def plot_weibull_lifetime_pdfs_MC():
    # All pf values you simulated (same order as your original loop)
    pf_values = [1, 0, 0.9, -0.9, 0.8, -0.8, 0.7, -0.7, 0.6, -0.6,
                 0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1]

    # Inductive pf values only
    pf_plot_values = [-0.1, -0.5, -0.9]

    lifetimes_IGBT = {}
    lifetimes_Diode = {}

    # ==========================
    # Load lifetime samples
    # ==========================
    for i, pf in enumerate(pf_values, start=1):
        base_folder = Path("Final_results") / f"Simulation_{i}"
        igbt_file = base_folder / "df_lifetime_IGBT_MC" / "df.parquet"
        diode_file = base_folder / "df_lifetime_Diode_MC" / "df.parquet"

        df_igbt = pd.read_parquet(igbt_file)
        df_diode = pd.read_parquet(diode_file)

        lifetimes_IGBT[pf] = df_igbt["Lifetime_igbt_MC_actual"].to_numpy()
        lifetimes_Diode[pf] = df_diode["Lifetime_diode_MC_actual"].to_numpy()

        del df_igbt, df_diode

    # Weibull fit helper
    def fit_weibull(samples):
        samples = np.asarray(samples)
        samples = samples[samples > 0]
        c, loc, scale = weibull_min.fit(samples, floc=0)
        return c, loc, scale

    # Combined x-range
    all_samples = np.concatenate(
        [lifetimes_IGBT[pf] for pf in pf_plot_values] +
        [lifetimes_Diode[pf] for pf in pf_plot_values]
    )
    x = np.linspace(0, np.max(all_samples) * 1.05, 500)

    # ==========================
    # One Figure
    # ==========================
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    # Assign one color per pf value
    # (Matplotlib will cycle these in order)
    colors = plt.cm.tab10(np.linspace(0, 1, len(pf_plot_values)))
    pf_to_color = {pf: color for pf, color in zip(pf_plot_values, colors)}

    # Plot IGBT (solid)
    for pf in pf_plot_values:
        samples = lifetimes_IGBT[pf]
        c, loc, scale = fit_weibull(samples)
        pdf = weibull_min.pdf(x, c, loc=loc, scale=scale)

        pf_label = abs(pf)  # remove minus sign

        ax.plot(
            x,
            pdf,
            linestyle='-',
            linewidth=2,
            color=pf_to_color[pf],
            label=f"IGBT pf={pf_label}"
        )

    # Plot Diode (dashed)
    for pf in pf_plot_values:
        samples = lifetimes_Diode[pf]
        c, loc, scale = fit_weibull(samples)
        pdf = weibull_min.pdf(x, c, loc=loc, scale=scale)

        pf_label = abs(pf)

        ax.plot(x,pdf,linestyle='--',linewidth=2,color=pf_to_color[pf],label=f"Diode pf={pf_label}")

    ax.set_title("Weibull PDF – IGBT (solid) and Diode (dashed)")
    ax.set_xlabel("Lifetime [years]")
    ax.set_ylabel("PDF [-]")
    ax.grid(True)

    # Set x-limit
    ax.set_xlim(0, x.max())

    ax.legend()

    plt.tight_layout()
    out_dir = Path("Final_results") / "Figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_dir / "Weibull_PDF_IGBT_Diode_inductive_single_plot.pdf",dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ======================================================
    #  SWITCH BOTTLENECK LIFETIME (NEW: min(IGBT, Diode))
    # ======================================================

    # Compute switch lifetimes per sample per pf
    lifetimes_Switch = {}
    for pf in pf_plot_values:
        lifetimes_Switch[pf] = np.minimum(
            lifetimes_IGBT[pf],
            lifetimes_Diode[pf]
        )

    # x-range for switch (can reuse x since min <= components)
    # but you could recompute if you want:
    # all_switch = np.concatenate([lifetimes_Switch[pf] for pf in pf_plot_values])
    # x_switch = np.linspace(0, np.max(all_switch) * 1.05, 500)
    x_switch = x

    fig_sw, ax_sw = plt.subplots(figsize=(6.4, 4.8))

    for pf in pf_plot_values:
        samples = lifetimes_Switch[pf]
        c, loc, scale = fit_weibull(samples)
        pdf = weibull_min.pdf(x_switch, c, loc=loc, scale=scale)

        pf_label = abs(pf)

        ax_sw.plot(
            x_switch,
            pdf,
            linewidth=2,
            color=pf_to_color[pf],
            label=f"Switch pf={pf_label}"
        )

    ax_sw.set_title("Weibull PDF – Switch bottleneck lifetime (min of IGBT, Diode)")
    ax_sw.set_xlabel("Lifetime [years]")
    ax_sw.set_ylabel("PDF [-]")
    ax_sw.grid(True)
    ax_sw.set_xlim(0, x_switch.max())
    ax_sw.legend()

    plt.tight_layout()
    fig_sw.savefig(
        out_dir / "Weibull_PDF_Switch_bottleneck_inductive.pdf",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig_sw)

IGBT_and_Diode_Current()
Total_power_losses()
Switching_power_losses()
Conduction_power_losses()

Temperature_igbt_diode_values()
Temperature_pad_sink_values()
Temperature_delta_t_values()
Temperature_t_mean_values()
Temperature_thermal_period_values()
Lifetime_values()
plot_weibull_lifetime_pdfs_MC()