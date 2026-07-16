import numpy as np
from Input_parameters import Input_parameters_class
from Calculation_functions import Calculation_functions_class
from LCL_filter_design import LCL_filter_design_function
from Plotting_function import Plotting_functions_class
import pandas as pd
from functools import lru_cache
from pathlib import Path
import multiprocessing as mp
from matplotlib import pyplot as plt

def trial_function(pf_new, S_RMS_new, sim_dir):

    params = Input_parameters_class()
    Calculation_functions = Calculation_functions_class()
    Plotting_function = Plotting_functions_class()

    Vdc_rated = params.Vdc_rated; Vo_rated = params.Vo_rated; inverter_phases = params.inverter_phases; M_rated = params.M_rated; single_phase_inverter_topology = params.single_phase_inverter_topology; waveform_voltage_definition = params.waveform_voltage_definition; modulation_scheme = params.modulation_scheme; f = params.f; fsw = params.fsw; T = params.T; Tsw = params.Tsw; omega = params.omega
    Profile_size = params.Profile_size; Vdc_RMS = params.Vdc_RMS; M = params.M; Vo = params.Vo; Vg_RMS = params.Vg_RMS; S_RMS = params.S_RMS; pf = params.pf; P_RMS = params.P_RMS; Q_RMS = params.Q_RMS; Ig_RMS = params.Ig_RMS
    T_amb = params.T_amb; heat_transfer_coefficient = params.heat_transfer_coefficient
    resolution_per_cycle = params.resolution_per_cycle; dt = params.dt; samples_per_switching_period = params.samples_per_switching_period; Minimum_required_samples_per_switching_period = params.Minimum_required_samples_per_switching_period; seconds_per_sample = params.seconds_per_sample
    L1_specs = params.L1_specs
    C_specs = params.C_specs
    L2_specs = params.L2_specs
    Vg_ll_RMS = params.Vg_ll_RMS; S_rated = params.S_rated; I_rated_RMS = params.I_rated_RMS; I_rated_peak = params.I_rated_peak; current_ripple_limit = params.current_ripple_limit; delta = params.delta; omega_sw = params.omega_sw

    _, dataframes_dir, figures_dir = Calculation_functions.create_simulation_folders(base="Results")

    #sim_dir = "Results/Simulation_1"
    #dataframes_dir = "Results/Simulation_1/Dataframes"
    #figures_dir = "Results/Simulation_1/Figures"

    pf = np.full(len(pf),pf_new)
    S_RMS = np.full(len(S_RMS), S_RMS_new)
    print(sim_dir, pf_new, S_RMS[0])

    # ----------------------------------------#
    # Input validation and safety checks
    # ----------------------------------------#

    # Check that all inverter, grid, PWM, and mission-profile inputs are physically consistent before simulation.
    Calculation_functions.validate_ac_rms_voltage_limit(Vdc_RMS=Vdc_RMS, M=M, inverter_phases=inverter_phases, modulation_scheme=modulation_scheme, single_phase_inverter_topology=single_phase_inverter_topology, Vg_RMS=Vg_RMS)
    Calculation_functions.validate_pwm_pulse_amplitude(Vdc_rated=Vdc_rated, inverter_phases=inverter_phases, single_phase_inverter_topology=single_phase_inverter_topology,waveform_voltage_definition=waveform_voltage_definition, Vo=Vo)
    Calculation_functions.validate_simulation_resolution(samples_per_switching_period=samples_per_switching_period, Minimum_required_samples_per_switching_period=Minimum_required_samples_per_switching_period)
    Calculation_functions.validate_grid_phase_voltage_matches_line_to_line_voltage(Vg_RMS=Vg_RMS,Vg_ll_RMS=Vg_ll_RMS,tolerance_percent=1.0)
    Calculation_functions.validate_mission_profile_lengths(Profile_size=Profile_size, Vdc_RMS=Vdc_RMS, M=M, Vo=Vo, Vg_RMS=Vg_RMS, S_RMS=S_RMS, pf=pf)
    Calculation_functions.validate_power_factor_profile(pf=pf)
    Calculation_functions.validate_apparent_power_limit(S_RMS=S_RMS, S_rated=S_rated)
    Calculation_functions.validate_pwm_pulse_amplitude_profile(Vdc_RMS=Vdc_RMS, inverter_phases=inverter_phases, single_phase_inverter_topology=single_phase_inverter_topology, waveform_voltage_definition=waveform_voltage_definition, Vo=Vo )

    # ----------------------------------------#
    # LCL filter component selection
    # ----------------------------------------#

    L1_optimum, L2_optimum, C_optimum, R3_optimum = LCL_filter_design_function(Vg_ll_RMS=Vg_ll_RMS, S_rated=S_rated, I_rated=I_rated_RMS, fsw=fsw, omega_sw=omega_sw, fo=f, Udc_rated=Vdc_rated, M_rated=M_rated, inverter_phases=inverter_phases, modulation_scheme=modulation_scheme, print_values = False, current_ripple_limit=0.30, delta=0.19, num_C_values=10)
    Calculation_functions.check_within_tolerance({"L1": (L1_specs["L1"], L1_optimum), "L2": (L2_specs["L2"], L2_optimum), "C":  (C_specs["C"],   C_optimum), "R3": (C_specs["R3"],  R3_optimum), }, tol=0.125)

    # ----------------------------------------#
    # LCL filter middle branch [C]
    # ----------------------------------------#

    if C_specs["Thermal_resistance_C"] == None:
        C_specs["Thermal_resistance_C"] = Calculation_functions.calculate_capacitor_thermal_resistance(method="surface_area", case_shape="cylinder", D_case=C_specs["D_case"], H_case=C_specs["H_case"], heat_transfer_coefficient=heat_transfer_coefficient)
        #C_specs["Thermal_resistance_C"] = Calculation_functions.calculate_capacitor_thermal_resistance(method="surface_area", case_shape="box", W_case=C_specs["W_case"], H_case=C_specs["H_case"], L_case=C_specs["L_case"],heat_transfer_coefficient=heat_transfer_coefficient)
    if C_specs["tan_delta_0"] == None:
        C_specs["tan_delta_0"] = Calculation_functions.calculate_tan_delta_0(tan_delta_measured = C_specs["tan_delta_measured"],  Rs = C_specs["Rs"],C = C_specs["C"], f_measured = C_specs["f_measured_for_tan_delta"])

    # ----------------------------------------#
    # LCL filter inverter side [L1]
    # ----------------------------------------#

    # Core Geometry

    A_surface_L1 = Calculation_functions.calculate_core_surface_area(A_core = L1_specs["A_core"], B_core = L1_specs["B_core"], D_core = L1_specs["D_core"], F_core = L1_specs["F_core"], G_core = L1_specs["G_core"])
    Ae_L1 = Calculation_functions.calculate_Ae(method = "geometry", kf=L1_specs["kf"], D_core=L1_specs["D_core"], E_core=L1_specs["E_core"]) # Effective cross-sectional area of the core [m²].
    le_L1 = Calculation_functions.calculate_le(method="geometry", A_core=L1_specs["A_core"], B_core=L1_specs["B_core"], F_core=L1_specs["F_core"], G_core=L1_specs["G_core"])  # [m]   Effective magnetic path length; the average distance the flux travels around the core loop
    Ve_L1 = Calculation_functions.calculate_core_volume(Ae=Ae_L1, le=le_L1)                                                    # [m³]  Effective core volume; used to compute total core losses from volumetric loss density
    N_L1 = Calculation_functions.calculate_turns(L=L1_specs["L1"], I_peak=I_rated_peak, B_max=L1_specs["B_max"], Ae = Ae_L1)   # [-] Minimum number of turns required.
    lg_L1 = Calculation_functions.calculate_air_gap(mu_0=L1_specs["mu_0"], N=N_L1, Ae=Ae_L1, L=L1_specs["L1"], le=le_L1, mu_r=L1_specs["mu_r"])  # [m] Required air gap length.
    B_peak_L1 = Calculation_functions.calculate_B_peak(mu_0=L1_specs["mu_0"], N=N_L1, I_peak=I_rated_peak, lg=lg_L1, le=le_L1, mu_r=L1_specs["mu_r"]) # [T] Peak flux density in the core.
    Calculation_functions.safety_checks(B_peak=B_peak_L1, B_max=L1_specs["B_max"], Bsat=L1_specs["Bsat"], lg=lg_L1, le=le_L1)

    # Winding

    A_wire_L1_minimum, d_wire_L1_minimum = Calculation_functions.calculate_minimum_required_wire_area(I_RMS_rated = I_rated_RMS, J_max = L1_specs["J_max"]) # [m²] Minimum copper cross-section required to carry the rated current
    N_parallel_wire_L1 = Calculation_functions.calculate_parallel_strands(A_wire_minimum = A_wire_L1_minimum, A_strand = L1_specs["A_strand"])             # [-] The number of parallel strands required to achieve the minimum copper cross-section from individual strand area
    A_wire_actual_L1 = Calculation_functions.calculate_actual_wire_area(N_parallel = N_parallel_wire_L1, A_strand = L1_specs["A_strand"])                  # [m²] Actual total copper cross-section after rounding up
    Calculation_functions.check_window_fill(N_turns = N_L1, N_parallel = N_parallel_wire_L1, A_wire_bare = L1_specs["A_strand"], F_core = L1_specs["F_core"], G_core = L1_specs["G_core"], kf_window_max= 0.5) # Check whether the winding physically fits inside the core window.
    l_turn_L1 = Calculation_functions.calculate_l_turn(D_core=L1_specs["D_core"], E_core=L1_specs["E_core"])                                               # [m] Estimate mean length of one turn for a rectangular toroidal core.
    Rdc_L1 = Calculation_functions.calculate_Rdc(rho=L1_specs["rho"], N=N_L1, l_turn=l_turn_L1, A_wire=A_wire_actual_L1)                                   # [ohm] float  DC winding resistance # Assumed  no Skin or Proximity Effect
    R_th_L1 = Calculation_functions.calculate_inductor_thermal_resistance(method="surface_area", A_surface=A_surface_L1,heat_transfer_coefficient=heat_transfer_coefficient)      # [K/W] Thermal resistance from  to ambient.

    # ----------------------------------------#
    # LCL filter grid side [L2]
    # ----------------------------------------#

    # Core Geometry

    A_surface_L2 = Calculation_functions.calculate_core_surface_area(A_core = L2_specs["A_core"], B_core = L2_specs["B_core"], D_core = L2_specs["D_core"], F_core = L2_specs["F_core"], G_core = L2_specs["G_core"])
    Ae_L2 = Calculation_functions.calculate_Ae(method = "geometry", kf=L2_specs["kf"], D_core=L2_specs["D_core"], E_core=L2_specs["E_core"]) # Effective cross-sectional area of the core [m²].
    le_L2 = Calculation_functions.calculate_le(method="geometry", A_core=L2_specs["A_core"], B_core=L2_specs["B_core"], F_core=L2_specs["F_core"], G_core=L2_specs["G_core"])  # [m]   Effective magnetic path length; the average distance the flux travels around the core loop
    Ve_L2 = Calculation_functions.calculate_core_volume(Ae=Ae_L2, le=le_L2)                                                    # [m³]  Effective core volume; used to compute total core losses from volumetric loss density
    N_L2 = Calculation_functions.calculate_turns(L=L2_specs["L2"], I_peak=I_rated_peak, B_max=L2_specs["B_max"], Ae = Ae_L2)   # [-] Minimum number of turns required.
    lg_L2 = Calculation_functions.calculate_air_gap(mu_0=L2_specs["mu_0"], N=N_L2, Ae=Ae_L2, L=L2_specs["L2"], le=le_L2, mu_r=L2_specs["mu_r"])  # [m] Required air gap length.
    B_peak_L2 = Calculation_functions.calculate_B_peak(mu_0=L2_specs["mu_0"], N=N_L2, I_peak=I_rated_peak, lg=lg_L2, le=le_L2, mu_r=L2_specs["mu_r"]) # [T] Peak flux density in the core.
    Calculation_functions.safety_checks(B_peak=B_peak_L2, B_max=L2_specs["B_max"], Bsat=L2_specs["Bsat"], lg=lg_L2, le=le_L2)

    # Winding

    A_wire_L2_minimum, d_wire_L2_minimum = Calculation_functions.calculate_minimum_required_wire_area(I_RMS_rated = I_rated_RMS, J_max = L2_specs["J_max"]) # [m²] Minimum copper cross-section required to carry the rated current
    N_parallel_wire_L2 = Calculation_functions.calculate_parallel_strands(A_wire_minimum = A_wire_L2_minimum, A_strand = L2_specs["A_strand"])             # [-] The number of parallel strands required to achieve the minimum copper cross-section from individual strand area
    A_wire_actual_L2 = Calculation_functions.calculate_actual_wire_area(N_parallel = N_parallel_wire_L2, A_strand = L2_specs["A_strand"])                  # [m²] Actual total copper cross-section after rounding up
    Calculation_functions.check_window_fill(N_turns = N_L2, N_parallel = N_parallel_wire_L2, A_wire_bare = L2_specs["A_strand"], F_core = L2_specs["F_core"], G_core = L2_specs["G_core"], kf_window_max= 0.5) # Check whether the winding physically fits inside the core window.
    l_turn_L2 = Calculation_functions.calculate_l_turn(D_core=L2_specs["D_core"], E_core=L2_specs["E_core"])                                               # [m] Estimate mean length of one turn for a rectangular toroidal core.
    Rdc_L2 = Calculation_functions.calculate_Rdc(rho=L2_specs["rho"], N=N_L2, l_turn=l_turn_L2, A_wire=A_wire_actual_L2)                                                         # [ohm] float  DC winding resistance # Assumed  no Skin or Proximity Effect
    R_th_L2 = Calculation_functions.calculate_inductor_thermal_resistance(method = "surface_area", A_surface = A_surface_L2 , heat_transfer_coefficient=heat_transfer_coefficient)   # [K/W] Thermal resistance from  to ambient .

    # ----------------------------------------#
    # Start of lru_cache
    # ----------------------------------------#

    @lru_cache(maxsize=1500)
    def solve_setpoint(Vdc_RMS_i, M_i, Vo_i, Vg_RMS_i, S_RMS_i, pf_i, P_RMS_i, Q_RMS_i, Ig_RMS_i, T_amb_i):

        t_one = np.arange(0, 1, dt)  # time vector for a single second

        # ----------------------------------------#
        # Electrical model
        # ----------------------------------------#

        # grid voltage waveform (one second)
        Vg = np.sqrt(2) * Vg_RMS_i * np.sin(omega * t_one)

        pf_inst = np.full(resolution_per_cycle * f, pf_i)
        phi = np.arccos(np.clip(np.abs(pf_inst), 0.0, 1.0))
        sign = np.where(pf_inst >= 0, 1.0, -1.0)  # 0 -> +1, so the 90° shift survives
        phase_shift = sign * phi

        Ig_ref = np.sqrt(2) * Ig_RMS_i * np.sin(omega * t_one + phase_shift)

        # required inverter voltage (pass single-element arrays, Profile_size=1)
        Vs_ref = Calculation_functions.compute_Vs_ref_phasor(t=t_one, f=f, Ig_RMS=np.array([Ig_RMS_i]), Vg_RMS=np.array([Vg_RMS_i]), phase_shift=phase_shift, L1=L1_specs["L1"], L2=L2_specs["L2"], C=C_specs["C"],R1=L1_specs["R1"], R2=L2_specs["R2"], R3=C_specs["R3"],Profile_size=1,samples_per_second=resolution_per_cycle * f)

        # Vs_ref = Calculation_functions.distort_Vs_ref(Vs_ref=Vs_ref, t=t_one, omega=omega,harmonics={5: 0.08, 7: 0.05, 11: 0.04, 13: 0.03}, noise_level=0.02, seed=0)


        Calculation_functions.validate_required_inverter_voltage(Vs_ref=Vs_ref, Vo_available=Vo_rated)

        # --- Optional Vs_ref refinement loop -------------------------------------
        # Iterate Vs_ref until the delivered I_L2 fundamental RMS matches the
        # commanded Ig_ref RMS. Set refine_max_iter = 1 to disable (single pass).
        refine_max_iter = 5  # 1 -> refinement off
        refine_tol = 1e-4  # relative tolerance on fundamental RMS (0.1%)
        n_cycles_meas = 10  # cycles used to measure the delivered fundamental

        target_rms = Ig_RMS_i  # commanded grid-current RMS (the goal)
        spc = int(round(resolution_per_cycle))
        win = n_cycles_meas * spc

        best = None
        converged = False
        for it in range(refine_max_iter):
            # switching output (Vo passed as single-element array, Profile_size=1)
            Vs = Calculation_functions.Three_phase_switching_output(t=t_one, Vs_ref=Vs_ref, Vo=np.array([Vo_i]),Tsw=Tsw, f=f, Profile_size=1)



#            _ = Calculation_functions.check_Vs_quality(t=t_one, Vs=Vs, Vs_ref=Vs_ref, f=f, fsw=fsw, Profile_size=1,raise_on_fail=True)

            # LCL filter response
            V_L1, I_L1, V_C, I_C, V_L2, I_L2 = Calculation_functions.LCL_Filter_Grid_Connected(t=t_one, Vs=Vs, Vg=Vg,
                                                                                               L1=L1_specs["L1"],
                                                                                               L2=L2_specs["L2"],
                                                                                               C=C_specs["C"],
                                                                                               R1=L1_specs["R1"],
                                                                                               R2=L2_specs["R2"],
                                                                                               R3=C_specs["R3"])

            # delivered I_L2 TOTAL RMS (ripple included)
            #x = np.asarray(I_L2, float)[-win:]

            # Total
            #IL2_total_rms = np.sqrt(np.mean(x ** 2))

            # delivered I_L2 FUNDAMENTAL RMS (single-bin DFT, leakage-free)
            x = np.asarray(I_L2, float)[-win:]
            N = len(x)
            k = int(round(N / spc))  # cycles in window = fundamental bin
            X = np.dot(x, np.exp(-1j * 2 * np.pi * k * np.arange(N) / N))
            IL2_fund_rms = (2.0 * np.abs(X) / N) / np.sqrt(2)

            rel_err = abs(IL2_fund_rms - target_rms) / target_rms

            if best is None or rel_err < best[0]:
                best = (rel_err, Vs_ref.copy(),(V_L1, I_L1, V_C, I_C, V_L2, I_L2))

            if rel_err < refine_tol:
                converged = True
                break
            Vs_ref = Vs_ref * (target_rms / IL2_fund_rms)

        if not converged:
            rel_err, Vs_ref, (V_L1, I_L1, V_C, I_C, V_L2, I_L2) = best
            print(f"[warn] refinement did not converge; using best iteration, rel_err={rel_err:.2e}")

        # ----------------------------------------#
        # LCL filter middle branch [C]
        # ----------------------------------------#

        # ----- Capacitor branch [C] -----
        I_C_RMS_harmonics = Calculation_functions.compute_I_C_RMS_per_harmonic_for_capacitor(I_C=I_C, f=f, fsw=fsw,resolution_per_cycle=resolution_per_cycle, Profile_size=1)

        I_C_RMS = Calculation_functions.Singal_RMS(Signal=I_C, resolution_per_cycle=resolution_per_cycle, f=f)

        # Power losses
        P_total_C = Calculation_functions.Capacitor_total_power_losses(fsw=fsw, f=f, I_C_RMS_harmonics=I_C_RMS_harmonics, tan_delta_0=C_specs["tan_delta_0"], C=C_specs["C"], Rs=C_specs["Rs"])

        # Temperature  (T_amb is THIS operating point's ambient)
        T_C = Calculation_functions.Capacitor_hotspot_temperature(T_amb=T_amb_i, P_total_C=P_total_C,Thermal_resistance_C=C_specs["Thermal_resistance_C"])

        V_C_RMS = Calculation_functions.Singal_RMS(Signal=V_C, resolution_per_cycle=resolution_per_cycle, f=f)

        Calculation_functions.validate_capacitor_operating_limits(T_C=T_C, V_C_RMS=V_C_RMS, V_C=V_C,T_C_Rated=C_specs["T_C_Rated"],V_C_RMS_Rated=C_specs["V_C_RMS_Rated"],V_C_Peak_Rated=C_specs["V_C_Peak_Rated"],V_RMS_overvoltage_factor=1.5, V_peak_overvoltage_factor=1.5)

        # ----------------------------------------#
        # LCL filter inverter side [L1]
        # ----------------------------------------#

        # ----- Inverter-side inductor [L1] (operating-point dependent) -----
        V_L1_RMS = Calculation_functions.Singal_RMS(Signal=V_L1, resolution_per_cycle=resolution_per_cycle, f=f)
        I_L1_RMS = Calculation_functions.Singal_RMS(Signal=I_L1, resolution_per_cycle=resolution_per_cycle, f=f)

        I_L1_peak_harmonics, harmonic_orders_L1, harmonic_freqs_L1 = Calculation_functions.compute_I_L_peak_per_harmonic_for_inductor(I_L=I_L1, f=f, resolution_per_cycle=resolution_per_cycle, Profile_size=1)

        P_c_L1, _ = Calculation_functions.calculate_inductor_core_losses(I_peak_harmonics=I_L1_peak_harmonics, harmonic_freqs=harmonic_freqs_L1,
                                                                         mu_0=L1_specs["mu_0"], N=N_L1, lg=lg_L1, le=le_L1, mu_r=L1_specs["mu_r"],
                                                                         k=L1_specs["k"], a=L1_specs["a"], b=L1_specs["b"], Ve=Ve_L1)

        P_w_L1, _ = Calculation_functions.calculate_winding_losses(I_L=I_L1, Rdc=Rdc_L1, resolution_per_cycle=resolution_per_cycle, f=f, Profile_size=1)

        P_total_L1 = P_c_L1 + P_w_L1

        T_inductor_L1 = Calculation_functions.calculate_inductor_temperature(T_amb=T_amb_i, R_th=R_th_L1, P_total=P_total_L1)

        Calculation_functions.check_insulation_voltage_stress(V_L=V_L1, N_turns=N_L1, V_bd=L1_specs['V_bd'],resolution_per_cycle=resolution_per_cycle, f=f, Profile_size=1)

        # ----------------------------------------#
        # LCL filter grid side [L2]
        # ----------------------------------------#

        # ----- Grid-side inductor [L2] (operating-point dependent) -----
        V_L2_RMS = Calculation_functions.Singal_RMS(Signal=V_L2, resolution_per_cycle=resolution_per_cycle, f=f)
        I_L2_RMS = Calculation_functions.Singal_RMS(Signal=I_L2, resolution_per_cycle=resolution_per_cycle, f=f)

        I_L2_peak_harmonics, harmonic_orders_L2, harmonic_freqs_L2 = Calculation_functions.compute_I_L_peak_per_harmonic_for_inductor(I_L=I_L2, f=f, resolution_per_cycle=resolution_per_cycle, Profile_size=1)

        P_c_L2, _ = Calculation_functions.calculate_inductor_core_losses(I_peak_harmonics=I_L2_peak_harmonics, harmonic_freqs=harmonic_freqs_L2,mu_0=L2_specs["mu_0"], N=N_L2, lg=lg_L2, le=le_L2, mu_r=L2_specs["mu_r"],k=L2_specs["k"], a=L2_specs["a"], b=L2_specs["b"], Ve=Ve_L2)

        P_w_L2, _ = Calculation_functions.calculate_winding_losses(I_L=I_L2, Rdc=Rdc_L2, resolution_per_cycle=resolution_per_cycle, f=f, Profile_size=1)

        P_total_L2 = P_c_L2 + P_w_L2

        T_inductor_L2 = Calculation_functions.calculate_inductor_temperature(T_amb=T_amb_i, R_th=R_th_L2, P_total=P_total_L2)

        Calculation_functions.check_insulation_voltage_stress(V_L=V_L2, N_turns=N_L2, V_bd=L2_specs['V_bd'],resolution_per_cycle=resolution_per_cycle, f=f, Profile_size=1)

        return (Vg, pf_inst, phi, phase_shift, Ig_ref, Vs_ref, Vs,
                V_L1, I_L1, V_C, I_C, V_L2, I_L2,
                I_C_RMS_harmonics, I_C_RMS, P_total_C, T_C, V_C_RMS,
                V_L1_RMS, I_L1_RMS, P_c_L1, P_w_L1, P_total_L1, T_inductor_L1,
                V_L2_RMS, I_L2_RMS, P_c_L2, P_w_L2, P_total_L2, T_inductor_L2)

    # collect one tuple per second
    results = [solve_setpoint(round(float(Vdc_RMS[i]), 3), round(float(M[i]), 4), round(float(Vo[i]), 3),
                              round(float(Vg_RMS[i]), 3), round(float(S_RMS[i]), 1), round(float(pf[i]), 4),
                              round(float(P_RMS[i]), 1), round(float(Q_RMS[i]), 1), round(float(Ig_RMS[i]), 3),
                              round(float(T_amb[i]), 2),)
        for i in range(Profile_size)]

    # transpose: turn a list-of-tuples into a tuple-of-lists, then concatenate each
    (Vg, pf_inst, phi, phase_shift, Ig_ref, Vs_ref, Vs,
    V_L1, I_L1, V_C, I_C, V_L2, I_L2,
    I_C_RMS_harmonics, I_C_RMS, P_total_C, T_C, V_C_RMS,
    V_L1_RMS, I_L1_RMS, P_c_L1, P_w_L1, P_total_L1, T_inductor_L1,
    V_L2_RMS, I_L2_RMS, P_c_L2, P_w_L2, P_total_L2, T_inductor_L2) = (np.concatenate(col) for col in zip(*results))

    res = Calculation_functions.spectral_split_IL2_Ig(I_L2, Ig_ref, resolution_per_cycle, n_cycles=f)
    THD_percent_I_L2 = res["THD_IL2_self"]

    def compare_IL2_to_Ig_ref(I_L2, Ig_ref, resolution_per_cycle, n_cycles=1, printing=True):
        """
        Compare the delivered grid current against the commanded reference at the
        fundamental, in magnitude and phase.

        Both signals are analysed over the last n_cycles fundamental periods. The
        fundamental phasor of each is extracted with a single-bin DFT, the bin index
        being derived from the actual window length so the extraction stays
        leakage-free.

        Parameters
        ----------
        I_L2, Ig_ref : array
            Delivered grid-side current and commanded reference current [A].
        resolution_per_cycle : int
            Samples per fundamental cycle [-].
        n_cycles : int, optional
            Number of trailing fundamental cycles to analyse. Default 1.
        printing : bool, optional
            Print the comparison summary. Default True.

        Returns
        -------
        dict with keys:
            IL2_fund_rms, Ig_fund_rms   : fundamental RMS of each signal [A]
            magnitude_error_percent     : 100 * (IL2 - Ig) / Ig [%]
            IL2_phase_deg, Ig_phase_deg : fundamental phase of each signal [deg]
            phase_error_deg             : IL2 phase minus Ig phase, wrapped to +/-180 [deg]
        """
        spc = int(round(resolution_per_cycle))
        win = n_cycles * spc

        def _phasor(a):
            a = np.asarray(a, dtype=float)
            a = a[np.isfinite(a)]
            if win > len(a):
                raise ValueError(
                    f"Window of {win} samples exceeds signal length {len(a)}.")
            a = a[-win:]
            N = len(a)

            k = int(round(N / spc))  # cycles in window = fund bin
            n = np.arange(N)
            X = np.dot(a, np.exp(-1j * 2 * np.pi * k * n / N))

            rms = (2.0 * np.abs(X) / N) / np.sqrt(2)
            phase_deg = np.degrees(np.angle(X))
            return rms, phase_deg

        IL2_rms, IL2_phase = _phasor(I_L2)
        Ig_rms, Ig_phase = _phasor(Ig_ref)

        magnitude_error_percent = 100.0 * (IL2_rms - Ig_rms) / Ig_rms
        phase_error_deg = (IL2_phase - Ig_phase + 180.0) % 360.0 - 180.0

        if printing:
            print("=" * 52)
            print(f"  Ig_ref fundamental RMS : {Ig_rms:>12.4f}  A")
            print(f"  I_L2   fundamental RMS : {IL2_rms:>12.4f}  A")
            print(f"  Magnitude error        : {magnitude_error_percent:>12.4f}  %")
            print("-" * 52)
            print(f"  Ig_ref phase           : {Ig_phase:>12.4f}  deg")
            print(f"  I_L2   phase           : {IL2_phase:>12.4f}  deg")
            print(f"  Phase error            : {phase_error_deg:>12.4f}  deg")
            print("=" * 52)

        return {
            "IL2_fund_rms": IL2_rms,
            "Ig_fund_rms": Ig_rms,
            "magnitude_error_percent": magnitude_error_percent,
            "IL2_phase_deg": IL2_phase,
            "Ig_phase_deg": Ig_phase,
            "phase_error_deg": phase_error_deg,
        }

    compare_IL2_to_Ig_ref(I_L2=I_L2, Ig_ref=Ig_ref, resolution_per_cycle=resolution_per_cycle, n_cycles=1, printing=True)

    # ----------------------------------------#
    # LCL filter middle branch [C]
    # ----------------------------------------#

    # Lifetime Calculations
    if C_specs["Lifetime_calculations"] == "Graphical":
        Lifetime_C_series = Calculation_functions.Capacitor_lifetime_graphical(T_C= T_C, V_C_RMS=V_C_RMS, V_C_RMS_Rated=C_specs["V_C_RMS_Rated"], lifetime_curves=C_specs["lifetime_curves"])
    elif C_specs["Lifetime_calculations"] == "Analytical":
        Lifetime_C_series = Calculation_functions.calculate_capacitor_lifetime_analytical(T_operating = T_C, V_C_RMS = V_C_RMS, V_C_RMS_Rated = C_specs["V_C_RMS_Rated"], t1 = C_specs["Lifetime_Rated"], T1 = C_specs["Temperature_Rated"], A = C_specs["A"], n = C_specs["n"])
    Lifetime_C,  Lifetime_consumed_C  = Calculation_functions.miners_rule_modified(L_per_second=Lifetime_C_series,  seconds_per_sample=seconds_per_sample)

    # ----------------------------------------#
    # LCL filter inverter side [L1]
    # ----------------------------------------#

    Lifetime_L1_series = Calculation_functions.calculate_inductor_lifetime(T_operating = T_inductor_L1, T_rated = L1_specs["T_insulation_rated"], L_rated = L1_specs["L_insulation_rated"], Ea = L1_specs["Ea_insulation"], kb = L1_specs["kb"], L_max_years = L1_specs["L_max_years"])   # [Years]  Predicted winding insulation lifetime at each second of the mission profile
    Lifetime_L1, Lifetime_consumed_L1 = Calculation_functions.miners_rule_modified(L_per_second=Lifetime_L1_series, seconds_per_sample=seconds_per_sample)

    # ----------------------------------------#
    # LCL filter grid side [L2]
    # ----------------------------------------#

    Lifetime_L2_series = Calculation_functions.calculate_inductor_lifetime(T_operating = T_inductor_L2, T_rated = L2_specs["T_insulation_rated"], L_rated = L2_specs["L_insulation_rated"], Ea = L2_specs["Ea_insulation"], kb = L2_specs["kb"], L_max_years = L2_specs["L_max_years"])   # [Years]  Predicted winding insulation lifetime at each second of the mission profile
    Lifetime_L2, Lifetime_consumed_L2 = Calculation_functions.miners_rule_modified(L_per_second=Lifetime_L2_series, seconds_per_sample=seconds_per_sample)

    C_report = dict(C=C_specs["C"], R_th=C_specs["Thermal_resistance_C"], V_RMS=V_C_RMS, V_RMS_rated=C_specs["V_C_RMS_Rated"], I_RMS=I_C_RMS, P_total=P_total_C, T=T_C, T_rated=C_specs["T_C_Rated"], Lifetime=Lifetime_C,Lifetime_consumed_C=Lifetime_consumed_C)
    L1_report = dict(L=L1_specs["L1"], N=N_L1, lg=lg_L1, B_peak=B_peak_L1, B_max=L1_specs["B_max"], Bsat=L1_specs["Bsat"], Ae=Ae_L1, le=le_L1, Ve=Ve_L1, A_surface=A_surface_L1, I_RMS=I_L1_RMS, V_RMS=V_L1_RMS, Rdc=Rdc_L1, N_parallel=N_parallel_wire_L1, A_wire=A_wire_actual_L1, l_turn=l_turn_L1, P_core=P_c_L1, P_winding=P_w_L1, P_total=P_total_L1, R_th=R_th_L1, T=T_inductor_L1, T_rated=L1_specs["T_insulation_rated"], Lifetime=Lifetime_L1, Lifetime_consumed=Lifetime_consumed_L1,)
    L2_report = dict(L=L2_specs["L2"], N=N_L2, lg=lg_L2, B_peak=B_peak_L2, B_max=L2_specs["B_max"], Bsat=L2_specs["Bsat"], Ae=Ae_L2, le=le_L2, Ve=Ve_L2, A_surface=A_surface_L2, I_RMS=I_L2_RMS, V_RMS=V_L2_RMS, Rdc=Rdc_L2, N_parallel=N_parallel_wire_L2, A_wire=A_wire_actual_L2, l_turn=l_turn_L2, P_core=P_c_L2, P_winding=P_w_L2, P_total=P_total_L2, R_th=R_th_L2, T=T_inductor_L2, T_rated=L2_specs["T_insulation_rated"], Lifetime=Lifetime_L2, Lifetime_consumed=Lifetime_consumed_L2, )
    #Calculation_functions.compare_components(C_report, L1_report, L2_report)

    #blabla = False
    blabla = True

    if blabla == True:

        df_1_power_flow_RMS = pd.DataFrame(
            {
                "Vdc_RMS": Vdc_RMS,
                "Vg_RMS": Vg_RMS,
                "S_RMS": S_RMS,
                "pf": pf,
                "P_RMS": P_RMS,
                "Q_RMS": Q_RMS,
                "Ig_RMS": Ig_RMS,
                "T_amb":T_amb
            })
        df_1_power_flow_RMS.to_parquet(f"{dataframes_dir}/df_1_power_flow_RMS.parquet")
        #Plotting_function.plot_df_1_power_flow_RMS(df_1_power_flow_RMS=df_1_power_flow_RMS, figures_dir=figures_dir,xlabel = "Time [day]")
        del df_1_power_flow_RMS


        n = int(resolution_per_cycle * f)

        df_2_power_flow_inst = pd.DataFrame(
            {
                "pf_inst": pf_inst[-n:],
                "phi": phi[-n:],
                "Vg":Vg[-n:],
                "Ig_ref": Ig_ref[-n:],
                "Vs_ref": Vs_ref[-n:],
                "Vs": Vs[-n:],
                "V_L1": V_L1[-n:],
                "I_L1": I_L1[-n:],
                "V_C": V_C[-n:],
                "I_C": I_C[-n:],
                "V_L2": V_L2[-n:],
                "I_L2": I_L2[-n:],
                "THD_percent_I_L2": np.where(np.arange(n) == n - 1, THD_percent_I_L2, np.nan)
            })
        df_2_power_flow_inst.to_parquet(f"{dataframes_dir}/df_2_power_flow_inst.parquet")
        #Plotting_function.plot_df_2_power_flow_inst(df_2_power_flow_inst=df_2_power_flow_inst, figures_dir=figures_dir, resolution_per_cycle=resolution_per_cycle)
        #Plotting_function.plot_Ig_ref_vs_I_L2(df_2_power_flow_inst, figures_dir, resolution_per_cycle, f=50, t=None, y_margin=0.05,xlabel="Time [s]")
        #Plotting_function.plot_six_waveforms(df_2_power_flow_inst, figures_dir, resolution_per_cycle, f=50, t=None, y_margin=0.05,xlabel="Time [s]")
        del df_2_power_flow_inst

        df_3_C = pd.DataFrame(
            {
                "V_C_RMS" : np.atleast_1d(V_C_RMS),
                "I_C_RMS" : np.atleast_1d(I_C_RMS),
                "P_total_C" : np.atleast_1d(P_total_C),
                "T_C" : np.atleast_1d(T_C),
                "Lifetime_C" : Calculation_functions.last_of_column(Lifetime_C,Profile_size),
                "Lifetime_consumed_C": Calculation_functions.last_of_column(Lifetime_consumed_C,Profile_size),
            })
        df_3_C.to_parquet(f"{dataframes_dir}/df_3_C.parquet")

        df_4_L1 = pd.DataFrame(
            {
                # --- per-second ---
                "V_L1_RMS":            np.atleast_1d(V_L1_RMS),       # [V]
                "I_L1_RMS":            np.atleast_1d(I_L1_RMS),       # [A]
                "P_c_L1":              np.atleast_1d(P_c_L1),         # [W] core loss
                "P_w_L1":              np.atleast_1d(P_w_L1),         # [W] winding loss
                "P_total_L1":          np.atleast_1d(P_total_L1),     # [W]
                "T_inductor_L1":       np.atleast_1d(T_inductor_L1),  # [K]
                # --- scalars: last row only ---
                "A_surface_L1":        Calculation_functions.last_of_column(A_surface_L1,Profile_size),           # [m²]
                "Ae_L1":               Calculation_functions.last_of_column(Ae_L1,Profile_size),                  # [m²]
                "le_L1":               Calculation_functions.last_of_column(le_L1,Profile_size),                  # [m]
                "Ve_L1":               Calculation_functions.last_of_column(Ve_L1,Profile_size),                  # [m³]
                "N_L1":                Calculation_functions.last_of_column(N_L1,Profile_size),                   # [-]
                "lg_L1":               Calculation_functions.last_of_column(lg_L1,Profile_size),                  # [m]
                "B_peak_L1":           Calculation_functions.last_of_column(B_peak_L1,Profile_size),              # [T]
                "N_parallel_wire_L1":  Calculation_functions.last_of_column(N_parallel_wire_L1,Profile_size),     # [-]
                "A_wire_actual_L1":    Calculation_functions.last_of_column(A_wire_actual_L1,Profile_size),       # [m²]
                "l_turn_L1":           Calculation_functions.last_of_column(l_turn_L1,Profile_size),              # [m]
                "Rdc_L1":              Calculation_functions.last_of_column(Rdc_L1,Profile_size),                 # [Ω]
                "R_th_L1":             Calculation_functions.last_of_column(R_th_L1,Profile_size),                # [K/W]
                "Lifetime_L1":         Calculation_functions.last_of_column(Lifetime_L1,Profile_size),            # [yr]
                "Lifetime_consumed_L1":Calculation_functions.last_of_column(Lifetime_consumed_L1,Profile_size),   # [%]
            })
        df_4_L1.to_parquet(f"{dataframes_dir}/df_4_L1.parquet")

        df_5_L2 = pd.DataFrame(
            {
                # --- per-second ---
                "V_L2_RMS"            : np.atleast_1d(V_L2_RMS),       # [V]
                "I_L2_RMS"            : np.atleast_1d(I_L2_RMS),       # [A]
                "P_c_L2"              : np.atleast_1d(P_c_L2),         # [W] core loss
                "P_w_L2"              : np.atleast_1d(P_w_L2),         # [W] winding loss
                "P_total_L2"          : np.atleast_1d(P_total_L2),     # [W]
                "T_inductor_L2"       : np.atleast_1d(T_inductor_L2),  # [K]
                # --- scalars: last row only ---
                "A_surface_L2"        : Calculation_functions.last_of_column(A_surface_L2,Profile_size),           # [m²]
                "Ae_L2"               : Calculation_functions.last_of_column(Ae_L2,Profile_size),                  # [m²]
                "le_L2"               : Calculation_functions.last_of_column(le_L2,Profile_size),                  # [m]
                "Ve_L2"               : Calculation_functions.last_of_column(Ve_L2,Profile_size),                  # [m³]
                "N_L2"                : Calculation_functions.last_of_column(N_L2,Profile_size),                   # [-]
                "lg_L2"               : Calculation_functions.last_of_column(lg_L2,Profile_size),                  # [m]
                "B_peak_L2"           : Calculation_functions.last_of_column(B_peak_L2,Profile_size),              # [T]
                "N_parallel_wire_L2"  : Calculation_functions.last_of_column(N_parallel_wire_L2,Profile_size),     # [-]
                "A_wire_actual_L2"    : Calculation_functions.last_of_column(A_wire_actual_L2,Profile_size),       # [m²]
                "l_turn_L2"           : Calculation_functions.last_of_column(l_turn_L2,Profile_size),              # [m]
                "Rdc_L2"              : Calculation_functions.last_of_column(Rdc_L2,Profile_size),                 # [Ω]
                "R_th_L2"             : Calculation_functions.last_of_column(R_th_L2,Profile_size),                # [K/W]
                "Lifetime_L2"         : Calculation_functions.last_of_column(Lifetime_L2,Profile_size),            # [yr]
                "Lifetime_consumed_L2": Calculation_functions.last_of_column(Lifetime_consumed_L2,Profile_size),   # [%]
            })
        df_5_L2.to_parquet(f"{dataframes_dir}/df_5_L2.parquet")
        #Plotting_function.plot_df_components(df_3_C=df_3_C, df_4_L1=df_4_L1, df_5_L2=df_5_L2, figures_dir=figures_dir,xlabel = "Time [day]")
        del df_3_C, df_4_L1, df_5_L2

    # ----------------------------------------#
    # Monte Carlo Simulations
    # ----------------------------------------#

    # Inputs

    number_of_samples = 1000
    normal_distribution = 0.01
    rng = np.random.default_rng(42)

    # Capacitor

    if C_specs["Lifetime_calculations"] == "Graphical":

        T_C_samples = Calculation_functions.normal_distribution_function(np.mean(T_C), normal_distribution, number_of_samples, rng)
        V_C_samples = Calculation_functions.normal_distribution_function(np.mean(V_C_RMS), normal_distribution, number_of_samples, rng)
        V_C_RMS_Rated_samples = Calculation_functions.normal_distribution_function(C_specs["V_C_RMS_Rated"], normal_distribution,number_of_samples, rng)
        lifetime_curves_samples = Calculation_functions.build_lifetime_curves_samples(C_specs["lifetime_curves"], normal_distribution, number_of_samples, rng)

        Lifetime_C_MC = np.empty(number_of_samples)
        for i in range(number_of_samples):
            Lifetime_C_MC[i] = Calculation_functions.Capacitor_lifetime_graphical(T_C=T_C_samples[i], V_C_RMS=V_C_samples[i], V_C_RMS_Rated=V_C_RMS_Rated_samples[i], lifetime_curves=lifetime_curves_samples[i],)

    elif C_specs["Lifetime_calculations"] == "Analytical":

        T_eq_C = Calculation_functions.equivalent_temperature_capacitor(L_eq_years=Lifetime_C,  V_C_RMS=np.mean(V_C_RMS), V_C_RMS_Rated=C_specs["V_C_RMS_Rated"],
                                                  t1=C_specs["Lifetime_Rated"], T1=C_specs["Temperature_Rated"], A=C_specs["A"], n=C_specs["n"])

        T_C_samples   = Calculation_functions.normal_distribution_function(T_eq_C, normal_distribution, number_of_samples, rng)
        V_C_samples   = Calculation_functions.normal_distribution_function(np.mean(V_C_RMS), normal_distribution, number_of_samples, rng)
        V_C_RMS_Rated_samples = Calculation_functions.normal_distribution_function(C_specs["V_C_RMS_Rated"], normal_distribution, number_of_samples, rng)
        t1_samples   = Calculation_functions.normal_distribution_function(C_specs["Lifetime_Rated"], normal_distribution, number_of_samples, rng)
        T1_samples   = Calculation_functions.normal_distribution_function(C_specs["Temperature_Rated"], normal_distribution, number_of_samples, rng)
        A_samples     = Calculation_functions.normal_distribution_function(C_specs["A"],    normal_distribution, number_of_samples, rng)
        n_samples     = Calculation_functions.normal_distribution_function(C_specs["n"],    normal_distribution, number_of_samples, rng)

        Lifetime_C_MC = Calculation_functions.calculate_capacitor_lifetime_analytical(T_operating = T_C_samples,V_C_RMS = V_C_samples,
                                                                                      V_C_RMS_Rated = V_C_RMS_Rated_samples,t1 = t1_samples,
                                                                                      T1 = T1_samples, A = A_samples,n = n_samples,)

    L_eq_L1_years = Lifetime_L1
    L_eq_L2_years = Lifetime_L2

    T_eq_L1 = Calculation_functions.equivalent_temperature(L_eq_L1_years, L1_specs["T_insulation_rated"],L1_specs["L_insulation_rated"], L1_specs["Ea_insulation"],L1_specs["kb"])
    T_eq_L2 = Calculation_functions.equivalent_temperature(L_eq_L2_years, L2_specs["T_insulation_rated"],L2_specs["L_insulation_rated"], L2_specs["Ea_insulation"],L2_specs["kb"])

    # Inductor L1

    T_L1_samples      = Calculation_functions.normal_distribution_function(T_eq_L1,    normal_distribution, number_of_samples, rng)
    T_rated_L1_samples = Calculation_functions.normal_distribution_function(L1_specs["T_insulation_rated"], normal_distribution, number_of_samples, rng)
    L_rated_L1_samples = Calculation_functions.normal_distribution_function(L1_specs["L_insulation_rated"], normal_distribution, number_of_samples, rng)
    Ea_L1_samples     = Calculation_functions.normal_distribution_function(L1_specs["Ea_insulation"], normal_distribution, number_of_samples, rng)

    Lifetime_L1_MC = Calculation_functions.calculate_inductor_lifetime(T_operating = T_L1_samples,T_rated = T_rated_L1_samples,
                                                                       L_rated = L_rated_L1_samples, Ea = Ea_L1_samples,
                                                                       kb = L1_specs["kb"], L_max_years = L1_specs["L_max_years"],)

    # Inductor 2

    T_L2_samples      = Calculation_functions.normal_distribution_function(T_eq_L2, normal_distribution, number_of_samples, rng)
    T_rated_L2_samples = Calculation_functions.normal_distribution_function(L2_specs["T_insulation_rated"], normal_distribution, number_of_samples, rng)
    L_rated_L2_samples = Calculation_functions.normal_distribution_function(L2_specs["L_insulation_rated"], normal_distribution, number_of_samples, rng)
    Ea_L2_samples = Calculation_functions.normal_distribution_function(L2_specs["Ea_insulation"], normal_distribution, number_of_samples, rng)

    Lifetime_L2_MC = Calculation_functions.calculate_inductor_lifetime(T_operating = T_L2_samples,T_rated = T_rated_L2_samples,
                                                                       L_rated = L_rated_L2_samples, Ea = Ea_L2_samples,
                                                                       kb = L2_specs["kb"], L_max_years = L2_specs["L_max_years"],)

    Lifetime_LCL_MC = np.minimum.reduce([Lifetime_C_MC, Lifetime_L1_MC, Lifetime_L2_MC, ])


    # B10: 10th percentile of each MC lifetime distribution [years]
    B10_C = float(np.percentile(Lifetime_C_MC, 10))
    B10_L1 = float(np.percentile(Lifetime_L1_MC, 10))
    B10_L2 = float(np.percentile(Lifetime_L2_MC, 10))
    B10_LCL = float(np.percentile(Lifetime_LCL_MC, 10))

    df_6_MC = pd.DataFrame(
        {
            "Lifetime_C_MC"   : Lifetime_C_MC,    # [Years]
            "Lifetime_L1_MC"  : Lifetime_L1_MC,   # [Years]
            "Lifetime_L2_MC"  : Lifetime_L2_MC,   # [Years]
            "Lifetime_LCL_MC" : Lifetime_LCL_MC,  # [Years]
            "B10_C"           : B10_C,            # [-]
            "B10_L1"          : B10_L1,           # [-]
            "B10_L2"          : B10_L2,           # [-]
            "B10_LCL"         : B10_LCL,          # [-]
        })
    df_6_MC.to_parquet(f"{dataframes_dir}/df_6_MC.parquet")
    #Plotting_function.plot_lifetime_monte_carlo(Lifetime_C_MC=Lifetime_C_MC, Lifetime_L1_MC=Lifetime_L1_MC, Lifetime_L2_MC=Lifetime_L2_MC,
                                                #Lifetime_LCL_MC=Lifetime_LCL_MC,figures_dir=figures_dir,B10_C=B10_C, B10_L1=B10_L1,
                                                #B10_L2=B10_L2, B10_LCL=B10_LCL, plot_type="histogram", bins=50)


schedule = [
    (+0.866, 1e6,      "Simulation_1"),
    (+0.5,   1e6,      "Simulation_2"),
    (-0.866, 1e6,      "Simulation_3"),
    (-0.5,   1e6,      "Simulation_4"),
    (1.0,    1e6,      "Simulation_5"),
    (1.0,    1e6*0.75, "Simulation_6"),
    (1.0,    1e6*0.50, "Simulation_7"),
    (1.0,    1e6*0.25, "Simulation_8"),
]


def run_one(pf_new, S_RMS_new, name):
    sim_dir = str(Path("Results") / name)
    trial_function(pf_new, S_RMS_new, sim_dir)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    for pf_new, S_RMS_new, name in schedule:
        p = mp.Process(target=run_one, args=(pf_new, S_RMS_new, name))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"{name} failed with exit code {p.exitcode}")