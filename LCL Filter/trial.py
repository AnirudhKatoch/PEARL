
df_4_L2 = pd.DataFrame(
    {
        # --- per-second ---
        "V_L2_RMS":            np.atleast_1d(V_L2_RMS),       # [V]
        "I_L2_RMS":            np.atleast_1d(I_L2_RMS),       # [A]
        "P_c_L2":              np.atleast_1d(P_c_L2),         # [W] core loss
        "P_w_L2":              np.atleast_1d(P_w_L2),         # [W] winding loss
        "P_total_L2":          np.atleast_1d(P_total_L2),     # [W]
        "T_inductor_L2":       np.atleast_1d(T_inductor_L2),  # [K]
        # --- scalars: last row only ---
        "A_surface_L2":        Calculation_functions.last_of_column(A_surface_L2,Profile_size),           # [m²]
        "Ae_L2":               Calculation_functions.last_of_column(Ae_L2,Profile_size),                  # [m²]
        "le_L2":               Calculation_functions.last_of_column(le_L2,Profile_size),                  # [m]
        "Ve_L2":               Calculation_functions.last_of_column(Ve_L2,Profile_size),                  # [m³]
        "N_L2":                Calculation_functions.last_of_column(N_L2,Profile_size),                   # [-]
        "lg_L2":               Calculation_functions.last_of_column(lg_L2,Profile_size),                  # [m]
        "B_peak_L2":           Calculation_functions.last_of_column(B_peak_L2,Profile_size),              # [T]
        "N_parallel_wire_L2":  Calculation_functions.last_of_column(N_parallel_wire_L2,Profile_size),     # [-]
        "A_wire_actual_L2":    Calculation_functions.last_of_column(A_wire_actual_L2,Profile_size),       # [m²]
        "l_turn_L2":           Calculation_functions.last_of_column(l_turn_L2,Profile_size),              # [m]
        "Rdc_L2":              Calculation_functions.last_of_column(Rdc_L2,Profile_size),                 # [Ω]
        "R_th_L2":             Calculation_functions.last_of_column(R_th_L2,Profile_size),                # [K/W]
        "Lifetime_L2":         Calculation_functions.last_of_column(Lifetime_L2,Profile_size),            # [yr]
        "Lifetime_consumed_L2":Calculation_functions.last_of_column(Lifetime_consumed_L2,Profile_size),   # [%]
    })
df_4_L2.to_parquet("Results/df_4_L2.parquet")