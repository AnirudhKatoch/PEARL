
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


    plt.xlabel("Current [A]")
    plt.ylabel("Power losses [W]")
    #plt.title("Average power losses vs power factor")
    plt.xlim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Final_results/Figures/IGBT_and_Diode_current.pdf")