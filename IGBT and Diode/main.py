from Input_parameters import Input_parameters_class
import numpy as np
from Calculation_functions import Calculation_functions_class
import matplotlib.pyplot as plt
from Electrical_model import compute_IGBT_and_Diode_power_losses
import time
from Thermal_model import simulate_igbt_diode_cauer
import pandas as pd

start_time = time.time()

params = Input_parameters_class()     # create instance
locals().update(params.__dict__)      # inject variables

sim_dir, df_electrical_dir, df_thermal_dir = Calculation_functions_class.create_simulation_folders()

# ----------------------------------------#
# Chunking setup
# ----------------------------------------#

chunk_seconds = 30                # 1 hour per chunk
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
                                                                    method="BDF", rtol=Cauer_model_accuracy, atol=1e-6,debug=True,T0_init=T0_init)
    Tj_igbt = T_i[0, :]
    Tj_diode = T_d[0, :]
    T_case = T_p[0, :]
    T_sink = T_s[0, :]

    time_global = time_local + sec_start

    df_electrical_chunk = pd.DataFrame({"time": time_global, "P_I": P_I, "P_D": P_D, "is_I": is_I, "is_D": is_D, "P_sw_I": P_sw_I, "P_sw_D": P_sw_D, "P_con_I": P_con_I, "P_con_D": P_con_D})
    df_thermal_chunk = pd.DataFrame({"time": time_global, "Tj_igbt": Tj_igbt, "Tj_diode": Tj_diode, "T_case": T_case, "T_sink": T_sink, })

    df_electrical_chunk.to_parquet(df_electrical_dir / f"df_{chunk_idx + 1}.parquet", index=False,engine="pyarrow")
    df_thermal_chunk.to_parquet(df_thermal_dir / f"df_{chunk_idx + 1}.parquet", index=False,engine="pyarrow")

    # prepare T0_init for next chunk: last state of this chunk
    T_i_last = T_i[:, -1]
    T_d_last = T_d[:, -1]
    T_p_last = T_p[:, -1]
    T_s_last = T_s[:, -1]

    T0_init = np.concatenate([T_i_last, T_d_last, T_p_last, T_s_last])








plt.figure(figsize=(12, 6))

plt.plot(time_global[:40], Tj_igbt[:40]-273.15,  label="IGBT Junction")
plt.plot(time_global[:40], Tj_diode[:40]-273.15, label="Diode Junction")
plt.plot(time_global[:40], T_case[:40]-273.15,   label="Case")
plt.plot(time_global[:40], T_sink[:40]-273.15,   label="Heat Sink")

plt.xlabel("Time [s]")
plt.ylabel("Temperature [°C]")
plt.title("Thermal Response of IGBT + Diode Cauer Network")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig("Figures/neu_6.png")

end_time = time.time()
print("Execution time all code:", end_time - start_time, "seconds")
