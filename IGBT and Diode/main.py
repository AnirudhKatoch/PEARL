from Input_parameters import Input_parameters_class
import numpy as np
from Calculation_functions import Calculation_functions_class
import matplotlib.pyplot as plt
from Electrical_model import compute_IGBT_and_Diode_power_losses
import time
from Thermal_model import simulate_igbt_diode_cauer
import pandas as pd
from Plotting_results import Plotting_lifetime,Plotting_electrical,Plotting_electrical_loss,Plotting_thermal

Calculation_functions = Calculation_functions_class()

start_time = time.time()

# ----------------------------------------#
# Input parameters
# ----------------------------------------#

params = Input_parameters_class()


dt = params.dt; chunk_seconds = params.chunk_seconds; saving_dataframes = params.saving_dataframes; plotting_values = params.plotting_values

A0 = params.A0; A1 = params.A1; T0_K =  params.T0_K; lambda_K = params.lambda_K; alpha = params.alpha; Ea_J = params.Ea_J; kB_J_per_K = params.kB_J_per_K; C = params.C; gamma = params.gamma; k_thickness = params.k_thickness

max_IGBT_temperature = params.max_IGBT_temperature; max_Diode_temperature = params.max_Diode_temperature

f = params.f; omega = params.omega; T0_init = params.T0_init

Cauer_model_accuracy = params.Cauer_model_accuracy; deltaT_min = params.deltaT_min; T_env = params.T_env
r_I =params.r_I; cap_I = params.cap_I; r_D = params.r_D; cap_D = params.cap_D; r_paste = params.r_paste; cap_paste = params.cap_paste; r_sink = params.r_sink; cap_sink = params.cap_sink

f_sw = params.f_sw; t_on = params.t_on; t_off = params.t_off; I_ref = params.I_ref; V_ref = params.V_ref; Err_D = params.Err_D
R_IGBT = params.R_IGBT; V_0_IGBT = params.V_0_IGBT; R_D = params.R_D; V_0_D = params.V_0_D

S = params.S; P = params.P; Q = params.Q; pf = params.pf; Vs = params.Vs; Is = params.Is; V_dc = params.V_dc; phi = params.phi; M = params.M

sim_dir, df_electrical_loss_dir, df_thermal_dir, df_lifetime_IGBT_dir, df_lifetime_Diode_dir, df_electrical_dir, Figures_dir = Calculation_functions.create_simulation_folders()

# ----------------------------------------#
# Chunking setup
# ----------------------------------------#

chunk_seconds = int(chunk_seconds)
samples_per_sec = int(1 / dt)     # e.g. 1000 for dt = 0.001
N_sec = len(Is)                   # length of Is, phi, V_dc, pf in seconds
n_chunks = int(np.ceil(N_sec / chunk_seconds))

for chunk_idx in range(n_chunks):
    sec_start = chunk_idx * chunk_seconds
    sec_end   = min((chunk_idx + 1) * chunk_seconds, N_sec)

    if sec_start >= sec_end:
        break  # safety

    print(f"\n--- Chunk {chunk_idx+1}/{n_chunks} ---")

    Is_chunk   = Is[sec_start:sec_end]
    phi_chunk = phi[sec_start:sec_end]
    V_dc_chunk = V_dc[sec_start:sec_end]
    pf_chunk = pf[sec_start:sec_end]
    T_env_chunk = T_env[sec_start:sec_end]

    # ----------------------------------------#
    # Electrical calculations
    # ----------------------------------------#

    # Temporary containers for 1-second segments of power losses and electrical outputs
    P_I_list     = []; P_D_list     = []; is_I_list    = []; is_D_list    = []
    P_sw_I_list  = []; P_sw_D_list  = []; P_con_I_list = []; P_con_D_list = []

    for i,(Is_i, phi_i, V_dc_i, pf_i) in enumerate(zip(Is_chunk, phi_chunk, V_dc_chunk, pf_chunk)):

        P_I_sec, P_D_sec, is_I_sec, is_D_sec, P_sw_I_sec, P_sw_D_sec, P_con_I_sec, P_con_D_sec = compute_IGBT_and_Diode_power_losses(Is=Is_i, phi=phi_i, V_dc=V_dc_i, pf=pf_i, dt=dt,
                                                                                                                                     M=M, omega=omega, t_on=t_on, t_off=t_off, f_sw=f_sw, I_ref=I_ref, V_ref=V_ref, Err_D=Err_D,
                                                                                                                                     R_IGBT=R_IGBT, V_0_IGBT=V_0_IGBT, R_D=R_D, V_0_D=V_0_D)
        # Append each 1-second result of power losses and electrical outputs
        P_I_list.append(P_I_sec); P_D_list.append(P_D_sec); is_I_list.append(is_I_sec); is_D_list.append(is_D_sec)
        P_sw_I_list.append(P_sw_I_sec); P_sw_D_list.append(P_sw_D_sec); P_con_I_list.append(P_con_I_sec); P_con_D_list.append(P_con_D_sec)

    # Concatenate into final full length arrays of power losses and electrical outputs
    P_I = np.concatenate(P_I_list); P_D = np.concatenate(P_D_list); is_I = np.concatenate(is_I_list); is_D     = np.concatenate(is_D_list)
    P_sw_I = np.concatenate(P_sw_I_list); P_sw_D = np.concatenate(P_sw_D_list); P_con_I = np.concatenate(P_con_I_list); P_con_D = np.concatenate(P_con_D_list)

    # ----------------------------------------#
    # Thermal calculations
    # ----------------------------------------#

    time_local, T_i, T_d, T_p, T_s = simulate_igbt_diode_cauer(r_I=r_I, cap_I=cap_I, r_D = r_D, cap_D=cap_D,
                                                                     r_paste=r_paste, cap_paste=cap_paste, r_sink=r_sink, cap_sink=cap_sink,
                                                                     P_I=P_I, P_D=P_D, T_env=np.repeat(T_env_chunk, int(1 / dt)), dt=dt,
                                                                    method="BDF", rtol=Cauer_model_accuracy, atol=1e-6,debug=False,T0_init=T0_init)

    # prepare T0_init for next chunk: last state of this chunk
    T_i_last = T_i[:, -1]
    T_d_last = T_d[:, -1]
    T_p_last = T_p[:, -1]
    T_s_last = T_s[:, -1]

    T0_init = np.concatenate([T_i_last, T_d_last, T_p_last, T_s_last])

    time_global = time_local + sec_start

    Tj_igbt = T_i[0, :]
    Tj_diode = T_d[0, :]
    T_case = T_p[0, :]
    T_sink = T_s[0, :]

    if Cauer_model_accuracy<= 1e-3:
        Calculation_functions.check_igbt_diode_temp_limits(Tj_igbt=Tj_igbt,Tj_diode=Tj_diode,max_IGBT_temperature=max_IGBT_temperature,max_Diode_temperature=max_Diode_temperature)

    df_electrical_loss_chunk = pd.DataFrame({"time": time_global, "P_I": P_I, "P_D": P_D, "is_I": is_I, "is_D": is_D, "P_sw_I": P_sw_I, "P_sw_D": P_sw_D, "P_con_I": P_con_I, "P_con_D": P_con_D})
    df_electrical_loss_chunk.to_parquet(df_electrical_loss_dir / f"df_{chunk_idx + 1}.parquet", index=False,engine="pyarrow")

    df_thermal_chunk = pd.DataFrame({"time": time_global, "Tj_igbt": Tj_igbt, "Tj_diode": Tj_diode, "T_case": T_case, "T_sink": T_sink, })
    df_thermal_chunk.to_parquet(df_thermal_dir / f"df_{chunk_idx + 1}.parquet", index=False,engine="pyarrow")

    # Delete all large electrical arrays
    del P_I, P_D, is_I, is_D, P_sw_I, P_sw_D, P_con_I, P_con_D
    del P_I_list , P_D_list , is_I_list , is_D_list , P_sw_I_list , P_sw_D_list , P_con_I_list , P_con_D_list

    # Delete thermal arrays
    del T_i, T_d, T_p, T_s
    del T_case, T_sink
    del T_i_last, T_d_last, T_p_last, T_s_last

    # Delete dataframes
    del df_electrical_loss_chunk
    del df_thermal_chunk

    # Delete input chunks
    del Is_chunk, phi_chunk, V_dc_chunk, pf_chunk, T_env_chunk

    # Delete time arrays
    del time_local, time_global

    deltaT_igbt, Tmean_igbt, thermal_cycle_period_igbt,count_igbt = Calculation_functions.rainflow_algorithm(Tj_igbt,dt)
    deltaT_diode, Tmean_diode, thermal_cycle_period_diode, count_diode = Calculation_functions.rainflow_algorithm(Tj_diode, dt)

    #deltaT_igbt = np.clip(deltaT_igbt, 20,200)
    #deltaT_diode = np.clip(deltaT_diode, 20,200)

    # Delete thermal arrays
    del Tj_igbt,Tj_diode

    Nf_igbt = Calculation_functions.cycles_to_failure_lesit(deltaT=deltaT_igbt, Tmean=Tmean_igbt,
                                                                  thermal_cycle_period=thermal_cycle_period_igbt, A0=A0,
                                                                  A1=A1, T0_K=T0_K, lambda_K=lambda_K, alpha=alpha,
                                                                  Ea_J=Ea_J, kB_J_per_K=kB_J_per_K, C=C, gamma=gamma,
                                                                  k_thickness=k_thickness["IGBT"])

    Nf_diode = Calculation_functions.cycles_to_failure_lesit(deltaT=deltaT_diode, Tmean=Tmean_diode,
                                                                  thermal_cycle_period=thermal_cycle_period_diode, A0=A0,
                                                                  A1=A1, T0_K=T0_K, lambda_K=lambda_K, alpha=alpha,
                                                                  Ea_J=Ea_J, kB_J_per_K=kB_J_per_K, C=C, gamma=gamma,
                                                                  k_thickness=k_thickness["Diode"])

    df_lifetime_IGBT_chunk = pd.DataFrame({"deltaT_igbt": deltaT_igbt, "Tmean_igbt": Tmean_igbt, "thermal_cycle_period_igbt": thermal_cycle_period_igbt, "count_igbt":count_igbt, "Nf_igbt": Nf_igbt})
    df_lifetime_IGBT_chunk.to_parquet(df_lifetime_IGBT_dir / f"df_{chunk_idx + 1}.parquet", index=False, engine="pyarrow")

    del deltaT_igbt, Tmean_igbt, thermal_cycle_period_igbt, count_igbt, Nf_igbt, df_lifetime_IGBT_chunk

    df_lifetime_Diode_chunk = pd.DataFrame({"deltaT_diode": deltaT_diode, "Tmean_diode": Tmean_diode, "thermal_cycle_period_diode": thermal_cycle_period_diode,"count_diode":count_diode,"Nf_diode": Nf_diode})
    df_lifetime_Diode_chunk.to_parquet(df_lifetime_Diode_dir / f"df_{chunk_idx + 1}.parquet", index=False,engine="pyarrow")

    del deltaT_diode, Tmean_diode, thermal_cycle_period_diode, Nf_diode, count_diode, df_lifetime_Diode_chunk

Plotting_electrical(S=S,P=P,Q=Q,Vs=Vs,Is=Is,V_dc=V_dc,pf=pf,phi=phi,T_env=T_env,Figures_dir=Figures_dir)

df_electrical = pd.DataFrame({ "S":S, "P":P, "Q":Q, "pf":pf, "Vs":Vs, "Is": Is, "V_dc":V_dc, "phi":phi, "T_env":T_env})
df_electrical.to_parquet(df_electrical_dir / f"df.parquet", index=False,engine="pyarrow")
del S, P, Q, pf, Vs , V_dc, phi, T_env, df_electrical

df_IGBT = Calculation_functions.read_datafames(df_dir=df_lifetime_IGBT_dir)
Nf_igbt = df_IGBT["Nf_igbt"].to_numpy()
count_igbt = df_IGBT["count_igbt"].to_numpy()
Nf_igbt_eq, lifetime_years_igbt = Calculation_functions.miners_rule(Nf=Nf_igbt, count=count_igbt, Is=Is)
print("lifetime_years_igbt",lifetime_years_igbt)

df_Diode = Calculation_functions.read_datafames(df_dir=df_lifetime_Diode_dir)
Nf_diode = df_Diode["Nf_diode"].to_numpy()
count_diode = df_Diode["count_diode"].to_numpy()
Nf_diode_eq, lifetime_years_diode = Calculation_functions.miners_rule(Nf=Nf_diode, count=count_diode, Is=Is)
print("lifetime_years_diode",lifetime_years_diode)
del Is

Plotting_lifetime(df_IGBT=df_IGBT, df_Diode=df_Diode, Nf_igbt_eq=Nf_igbt_eq, lifetime_years_igbt=lifetime_years_igbt, Nf_diode_eq=Nf_diode_eq, lifetime_years_diode=lifetime_years_diode,Figures_dir=Figures_dir)
del lifetime_years_igbt,lifetime_years_diode,Nf_igbt_eq,Nf_diode_eq,df_IGBT,df_Diode,Nf_igbt,count_igbt,Nf_diode,count_diode

df_electrical_loss = Calculation_functions.read_datafames(df_dir=df_electrical_loss_dir)
Plotting_electrical_loss(df_electrical_loss=df_electrical_loss, Figures_dir=Figures_dir)
del df_electrical_loss

df_thermal = Calculation_functions.read_datafames(df_dir=df_thermal_dir)
Plotting_thermal(df_thermal=df_thermal,Figures_dir=Figures_dir)
del df_thermal









end_time = time.time()
print("Execution time all code:", end_time - start_time, "seconds")