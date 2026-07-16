import numpy as np
from Input_parameters import Input_parameters_class
from Calculation_functions import Calculation_functions_class
from LCL_filter_design import LCL_filter_design_function
from Plotting_function import Plotting_functions_class
import pandas as pd

params = Input_parameters_class()
Calculation_functions = Calculation_functions_class()
Plotting_function = Plotting_functions_class()

Vdc_rated = params.Vdc_rated; Vo_rated = params.Vo_rated; inverter_phases = params.inverter_phases; M_rated = params.M_rated; single_phase_inverter_topology = params.single_phase_inverter_topology; waveform_voltage_definition = params.waveform_voltage_definition; modulation_scheme = params.modulation_scheme; f = params.f; fsw = params.fsw; T = params.T; Tsw = params.Tsw; omega = params.omega
Profile_size = params.Profile_size; Vdc_RMS = params.Vdc_RMS; M = params.M; Vo = params.Vo; Vg_RMS = params.Vg_RMS; S_RMS = params.S_RMS; pf = params.pf; P_RMS = params.P_RMS; Q_RMS = params.Q_RMS; Ig_RMS = params.Ig_RMS
T_amb = params.T_amb
resolution_per_cycle = params.resolution_per_cycle; dt = params.dt; samples_per_switching_period = params.samples_per_switching_period; Minimum_required_samples_per_switching_period = params.Minimum_required_samples_per_switching_period
L1_specs = params.L1_specs
C_specs = params.C_specs
L2_specs = params.L2_specs
Vg_ll_RMS = params.Vg_ll_RMS; S_rated = params.S_rated; I_rated_RMS = params.I_rated_RMS; I_rated_peak = params.I_rated_peak; current_ripple_limit = params.current_ripple_limit; delta = params.delta; omega_sw = params.omega_sw

#sim_dir, dataframes_dir, figures_dir = Calculation_functions.create_simulation_folders(base="Results")

sim_dir = "Results/Simulation_1"
dataframes_dir = "Results/Simulation_1/Dataframes"
figures_dir = "Results/Simulation_1/Figures"


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
# Time profile
# ----------------------------------------#

t = np.arange(0, Profile_size, dt)                                                                                      # Create the simulation time vector from 0 to Profile_size seconds using the fixed time step dt
samples_per_second = resolution_per_cycle * f                                                                           # Number of simulation samples in one second, used to expand 1-second mission-profile values into time-domain waveforms

########################################################################################################################
# Electrical model
########################################################################################################################


Vg = np.sqrt(2) * np.repeat(Vg_RMS, samples_per_second) * np.sin(omega * t)                                             # Generate the instantaneous grid voltage waveform from the RMS grid-voltage profile

pf_inst = np.repeat(pf, samples_per_second)                                                                             # Expand the 1-second power-factor profile so it has the same length as the time vector
phi = np.arccos(np.abs(pf_inst))                                                                                        # Compute the power-factor phase angle from the magnitude of the power factor
phase_shift = np.sign(pf_inst) * phi                                                                                    # Apply the sign of the power factor to determine whether the current leads or lags the voltage
Ig_ref = np.sqrt(2) * np.repeat(Ig_RMS, samples_per_second) * np.sin(omega * t + phase_shift)                           # Generate the instantaneous current reference that the inverter should inject into the grid

Vs_ref = Calculation_functions.compute_Vs_ref_phasor(t=t, f=f, Ig_RMS=Ig_RMS, Vg_RMS=Vg_RMS, phase_shift=phase_shift, L1=L1_specs["L1"], L2=L2_specs["L2"], C=C_specs["C"], R1=L1_specs["R1"], R2=L2_specs["R2"], R3=C_specs["R3"], Profile_size=Profile_size, samples_per_second=samples_per_second)

Calculation_functions.validate_required_inverter_voltage(Vs_ref=Vs_ref, Vo_available=Vo_rated)         # Update validate_required_inverter_voltage

Vs = Calculation_functions.Three_phase_switching_output(t=t, Vs_ref=Vs_ref, Vo=Vo, Tsw=Tsw, f=f, Profile_size=Profile_size)         # Producing the switching Vs from semiconductors#
_ = Calculation_functions.check_Vs_quality( t=t, Vs=Vs, Vs_ref=Vs_ref, f=f, fsw=fsw, Profile_size=Profile_size, raise_on_fail=True) # Checking the quality of Vs produced

V_L1, I_L1, V_C, I_C, V_L2, I_L2 = Calculation_functions.LCL_Filter_Grid_Connected(t=t, Vs=Vs, Vg=Vg, L1=L1_specs["L1"], L2=L2_specs["L2"], C=C_specs["C"], R1=L1_specs["R1"], R2=L2_specs["R2"], R3=C_specs["R3"]) # Simulate the LCL filter response and obtain voltages and currents of L1, C, and L2

THD_percent = Calculation_functions.compute_THD_I_L2(t=t, I_L2=I_L2, Ig_ref=Ig_ref, f=f, dt=dt, resolution_per_cycle=resolution_per_cycle,save_path=f"Figures/Current_comparing_{pf[-1]}.png", plot = False,printing=False)


########################################################################################################################

# ----------------------------------------#
# LCL filter middle branch [C]
# ----------------------------------------#

if C_specs["Thermal_resistance_C"] == None:
    C_specs["Thermal_resistance_C"] = Calculation_functions.calculate_capacitor_thermal_resistance(method="surface_area", D_case=C_specs["D_case"], H_case=C_specs["H_case"], heat_transfer_coefficient=10)  # [W/m²K] natural convection
if C_specs["tan_delta_0"] == None:
    C_specs["tan_delta_0"] = Calculation_functions.calculate_tan_delta_0(tan_delta_measured = C_specs["tan_delta_measured"],  Rs = C_specs["Rs"],C = C_specs["C"], f_measured = C_specs["f_measured_for_tan_delta"])

I_C_RMS_harmonics = Calculation_functions.compute_I_C_RMS_per_harmonic_for_capacitor(I_C=I_C, f=f, fsw=fsw, resolution_per_cycle=resolution_per_cycle, Profile_size=Profile_size)
I_C_RMS = Calculation_functions.Singal_RMS(Signal=I_C, resolution_per_cycle=resolution_per_cycle, f=f)

# Power Losses

P_total_C = Calculation_functions.Capacitor_total_power_losses(fsw=fsw,f=f, I_C_RMS_harmonics=I_C_RMS_harmonics,tan_delta_0=C_specs["tan_delta_0"],C=C_specs["C"],Rs=C_specs["Rs"])

# Temperature Calculations

T_C = Calculation_functions.Capacitor_hotspot_temperature(T_amb=T_amb, P_total_C=P_total_C, Thermal_resistance_C=C_specs["Thermal_resistance_C"])
V_C_RMS = Calculation_functions.Singal_RMS(Signal=V_C, resolution_per_cycle=resolution_per_cycle, f=f)
Calculation_functions.validate_capacitor_operating_limits(T_C=T_C, V_C_RMS=V_C_RMS, V_C=V_C, T_C_Rated=C_specs["T_C_Rated"], V_C_RMS_Rated=C_specs["V_C_RMS_Rated"], V_C_Peak_Rated=C_specs["V_C_Peak_Rated"],V_RMS_overvoltage_factor=1.5, V_peak_overvoltage_factor=1.5)



# Lifetime Calculations

if C_specs["Lifetime_calculations"] == "Graphical":
    Lifetime_C_series = Calculation_functions.Capacitor_lifetime_graphical(T_C= T_C, V_C_RMS=V_C_RMS, V_C_RMS_Rated=C_specs["V_C_RMS_Rated"], lifetime_curves=C_specs["lifetime_curves"])
elif C_specs["Lifetime_calculations"] == "Analytical":
    Lifetime_C_series = Calculation_functions.calculate_capacitor_lifetime_analytical(T_operating = T_C, V_C_RMS = V_C_RMS, V_C_RMS_Rated = C_specs["V_C_RMS_Rated"], t1 = C_specs["Lifetime_Rated"], T1 = C_specs["Temperature_Rated"], A = C_specs["A"], n = C_specs["n"])

Lifetime_C, Lifetime_consumed_C = Calculation_functions.miners_rule(L_per_second =Lifetime_C_series)

#########################################################################################################################

# ----------------------------------------#
# LCL filter inverter side [L1]
# ----------------------------------------#

# RMS

V_L1_RMS = Calculation_functions.Singal_RMS(Signal=V_L1, resolution_per_cycle=resolution_per_cycle, f=f)
I_L1_RMS = Calculation_functions.Singal_RMS(Signal=I_L1, resolution_per_cycle=resolution_per_cycle, f=f)

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
Calculation_functions.check_window_fill(N_turns = N_L1, N_parallel = N_parallel_wire_L1, A_wire_bare = L1_specs["A_strand"], F_core = L1_specs["F_core"], G_core = L1_specs["G_core"], kf_window_max= 0.4) # Check whether the winding physically fits inside the core window.
l_turn_L1 = Calculation_functions.calculate_l_turn(D_core=L1_specs["D_core"], E_core=L1_specs["E_core"])                                               # [m] Estimate mean length of one turn for a rectangular toroidal core.
Rdc_L1 = Calculation_functions.calculate_Rdc(rho=L1_specs["rho"], N=N_L1, l_turn=l_turn_L1, A_wire=A_wire_actual_L1)                                                         # [ohm] float  DC winding resistance # Assumed  no Skin or Proximity Effect

# Power Losses

I_L1_peak_harmonics, harmonic_orders_L1, harmonic_freqs_L1  = Calculation_functions.compute_I_L_peak_per_harmonic_for_inductor(I_L=I_L1, f=f, resolution_per_cycle=resolution_per_cycle, Profile_size=Profile_size) # [A] Peak amplitudes at each harmonic of the fundamental frequency
P_c_L1, _ = Calculation_functions.calculate_inductor_core_losses(I_peak_harmonics=I_L1_peak_harmonics, harmonic_freqs=harmonic_freqs_L1, mu_0=L1_specs["mu_0"], N=N_L1, lg=lg_L1, le=le_L1, mu_r=L1_specs["mu_r"], k=L1_specs["k"], a=L1_specs["a"], b=L1_specs["b"], Ve=Ve_L1) # [W] Total core loss in the inductor L1
P_w_L1, _ = Calculation_functions.calculate_winding_losses(I_L = I_L1, Rdc = Rdc_L1, resolution_per_cycle = resolution_per_cycle, f = f, Profile_size = Profile_size) # [W] Total winding  loss in the inductor L1
P_total_L1 = P_c_L1 + P_w_L1   # [W] Total inductor losses

# Temperature Calculations

R_th_L1 = Calculation_functions.calculate_inductor_thermal_resistance(method = "surface_area", A_surface = A_surface_L1 , heat_transfer_coefficient = 10)   # [K/W] Thermal resistance from  to ambient .
T_inductor_L1 = Calculation_functions.calculate_inductor_temperature(T_amb = T_amb, R_th = R_th_L1, P_total = P_total_L1)                     # [K] Inductor temperature

# Lifetime Calculations

Calculation_functions.check_insulation_voltage_stress(V_L = V_L1, N_turns = N_L1, V_bd = L1_specs['V_bd'], resolution_per_cycle = resolution_per_cycle, f = f, Profile_size = Profile_size)
Lifetime_L1_series = Calculation_functions.calculate_inductor_lifetime(T_operating = T_inductor_L1, T_rated = L1_specs["T_insulation_rated"], L_rated = L1_specs["L_insulation_rated"], Ea = L1_specs["Ea_insulation"], kb = L1_specs["kb"] )   # [Years]  Predicted winding insulation lifetime at each second of the mission profile
Lifetime_L1, Lifetime_consumed_L1 = Calculation_functions.miners_rule(L_per_second = Lifetime_L1_series)



#########################################################################################################################

# ----------------------------------------#
# LCL filter grid side [L2]
# ----------------------------------------#

# RMS

V_L2_RMS = Calculation_functions.Singal_RMS(Signal=V_L2, resolution_per_cycle=resolution_per_cycle, f=f)
I_L2_RMS = Calculation_functions.Singal_RMS(Signal=I_L2, resolution_per_cycle=resolution_per_cycle, f=f)

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
Calculation_functions.check_window_fill(N_turns = N_L2, N_parallel = N_parallel_wire_L2, A_wire_bare = L2_specs["A_strand"], F_core = L2_specs["F_core"], G_core = L2_specs["G_core"], kf_window_max= 0.45) # Check whether the winding physically fits inside the core window.
l_turn_L2 = Calculation_functions.calculate_l_turn(D_core=L2_specs["D_core"], E_core=L2_specs["E_core"])                                               # [m] Estimate mean length of one turn for a rectangular toroidal core.
Rdc_L2 = Calculation_functions.calculate_Rdc(rho=L2_specs["rho"], N=N_L2, l_turn=l_turn_L2, A_wire=A_wire_actual_L2)                                                         # [ohm] float  DC winding resistance # Assumed  no Skin or Proximity Effect

# Power Losses

I_L2_peak_harmonics, harmonic_orders_L2, harmonic_freqs_L2  = Calculation_functions.compute_I_L_peak_per_harmonic_for_inductor(I_L=I_L2, f=f, resolution_per_cycle=resolution_per_cycle, Profile_size=Profile_size) # [A] Peak amplitudes at each harmonic of the fundamental frequency
P_c_L2, _ = Calculation_functions.calculate_inductor_core_losses(I_peak_harmonics=I_L2_peak_harmonics, harmonic_freqs=harmonic_freqs_L2, mu_0=L2_specs["mu_0"], N=N_L2, lg=lg_L2, le=le_L2, mu_r=L2_specs["mu_r"], k=L2_specs["k"], a=L2_specs["a"], b=L2_specs["b"], Ve=Ve_L2) # [W] Total core loss in the inductor L1
P_w_L2, _ = Calculation_functions.calculate_winding_losses(I_L = I_L2, Rdc = Rdc_L2, resolution_per_cycle = resolution_per_cycle, f = f, Profile_size = Profile_size) # [W] Total winding  loss in the inductor L1
P_total_L2 = P_c_L2 + P_w_L2   # [W] Total inductor losses

# Temperature Calculations

R_th_L2 = Calculation_functions.calculate_inductor_thermal_resistance(method = "surface_area", A_surface = A_surface_L2 , heat_transfer_coefficient = 10)   # [K/W] Thermal resistance from  to ambient .
T_inductor_L2 = Calculation_functions.calculate_inductor_temperature(T_amb = T_amb, R_th = R_th_L2, P_total = P_total_L2)                     # [K] Inductor temperature

# Lifetime Calculations

Calculation_functions.check_insulation_voltage_stress(V_L = V_L2, N_turns = N_L2, V_bd = L2_specs['V_bd'], resolution_per_cycle = resolution_per_cycle, f = f, Profile_size = Profile_size)
Lifetime_L2_series = Calculation_functions.calculate_inductor_lifetime(T_operating = T_inductor_L2, T_rated = L2_specs["T_insulation_rated"], L_rated = L2_specs["L_insulation_rated"], Ea = L2_specs["Ea_insulation"], kb = L2_specs["kb"] )   # [Years]  Predicted winding insulation lifetime at each second of the mission profile
Lifetime_L2, Lifetime_consumed_L2 = Calculation_functions.miners_rule(L_per_second = Lifetime_L2_series)


df_1_power_flow_RMS = pd.DataFrame(
    {
        "Vdc_RMS": Vdc_RMS,
        "Vg_RMS": Vg_RMS,
        "S_RMS": S_RMS,
        "pf": pf,
        "P_RMS": P_RMS,
        "Q_RMS": Q_RMS,
        "Ig_RMS": Ig_RMS,
    })
#df_1_power_flow_RMS.to_parquet(f"{dataframes_dir}/df_1_power_flow_RMS.parquet")
#Plotting_function.plot_df_1_power_flow_RMS(df_1_power_flow_RMS=df_1_power_flow_RMS, figures_dir=figures_dir)


df_2_power_flow_inst = pd.DataFrame(
    {
        "pf_inst": pf_inst,
        "phi": phi,
        "Ig_ref": Ig_ref,
        "Vs_ref": Vs_ref,
        "Vs": Vs,
        "V_L1": V_L1,
        "I_L1": I_L1,
        "V_C": V_C,
        "I_C": I_C,
        "V_L2": V_L2,
        "I_L2": I_L2,
        "THD_percent": np.where(np.arange(len(t)) == len(t) - 1, THD_percent, np.nan),
    })
#df_2_power_flow_inst.to_parquet(f"{dataframes_dir}/df_2_power_flow_inst.parquet")
Plotting_function.plot_df_2_power_flow_inst(df_2_power_flow_inst=df_2_power_flow_inst, figures_dir=figures_dir, resolution_per_cycle=resolution_per_cycle)


df_3_C = pd.DataFrame(
    {
        "V_C_RMS" : np.atleast_1d(V_C_RMS),
        "I_C_RMS" : np.atleast_1d(I_C_RMS),
        "P_total_C" : np.atleast_1d(P_total_C),
        "T_C" : np.atleast_1d(T_C),
        "Lifetime_C" : Calculation_functions.last_of_column(Lifetime_C,Profile_size),
        "Lifetime_consumed_C": Calculation_functions.last_of_column(Lifetime_consumed_C,Profile_size),
    })
#df_3_C.to_parquet(f"{dataframes_dir}/df_3_C.parquet")

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
#df_4_L1.to_parquet(f"{dataframes_dir}/df_4_L1.parquet")

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
#df_5_L2.to_parquet(f"{dataframes_dir}/df_5_L2.parquet")

Plotting_function.plot_df_components(df_3_C, df_4_L1, df_5_L2, figures_dir)
























########################################################################################################################


'''
def compare_components(C, L1, L2, profile_index=-1, K_to_C=273.15):
    """
    Print a side-by-side comparison of the capacitor (C) and the two inductors (L1, L2).

    Each argument is a dict of already-computed results. Values may be scalars or
    per-second arrays; arrays are reduced to the second given by `profile_index`
    (default -1 = last second of the mission profile).

    Expected keys
    -------------
    C  : C, R_th, V_RMS, V_RMS_rated, I_RMS, P_total, T, T_rated, Lifetime
    L1 / L2 : L, N, lg, B_peak, B_max, Bsat, Ae, le, Ve, A_surface,
              I_RMS, V_RMS, Rdc, N_parallel, A_wire, l_turn,
              P_core, P_winding, P_total, R_th, T, T_rated,
              Lifetime, Lifetime_consumed

    Missing keys print as a dash, so the function still works on partially-filled dicts.
    """

    LBL, UNT, COL, RAT = 24, 8, 15, 10
    line = "-" * (LBL + UNT + 2 * COL + RAT)

    def v(d, key):
        return _val(d.get(key), profile_index)

    def row(label, unit, a, b, scale=1.0, offset=0.0, nfmt="{:.4f}", ratio=False):
        s1 = nfmt.format(a * scale + offset) if a is not None else "-"
        s2 = nfmt.format(b * scale + offset) if b is not None else "-"
        if ratio and a not in (None, 0) and b not in (None, 0):
            rr = "{:.2f}".format(a / b)
        else:
            rr = ""
        print(f"{label:<{LBL}}{unit:<{UNT}}{s1:>{COL}}{s2:>{COL}}{rr:>{RAT}}")

    # ------------------------------------------------------------------ #
    # Header
    # ------------------------------------------------------------------ #
    print("=" * len(line))
    print(f"COMPONENT COMPARISON   (mission-profile second index = {profile_index})")
    print("=" * len(line))

    # ------------------------------------------------------------------ #
    # Capacitor (standalone block)
    # ------------------------------------------------------------------ #
    print("\nCAPACITOR")
    print(line)
    C_C = v(C, "C")
    vrms, vrated = v(C, "V_RMS"), v(C, "V_RMS_rated")
    tc, trated = v(C, "T"), v(C, "T_rated")
    print(f"  Capacitance        : {C_C * 1e6:.4f} µF" if C_C is not None else "  Capacitance        : -")
    if v(C, "R_th") is not None:
        print(f"  Thermal resistance : {v(C, 'R_th'):.4f} K/W")
    if vrms is not None:
        ratio_txt = f"   (rated {vrated:.0f} V, ratio {vrms / vrated:.2f})" if vrated else ""
        print(f"  V_RMS              : {vrms:.2f} V{ratio_txt}")
    if v(C, "I_RMS") is not None:
        print(f"  I_RMS              : {v(C, 'I_RMS'):.2f} A")
    if v(C, "P_total") is not None:
        print(f"  P_total            : {v(C, 'P_total'):.4f} W")
    if tc is not None:
        margin = f"   (rated {trated - K_to_C:.0f} °C)" if trated else ""
        print(f"  T_hotspot          : {tc - K_to_C:.2f} °C{margin}")
    if v(C, "Lifetime") is not None:
        print(f"  Lifetime           : {v(C, 'Lifetime'):.2f} years")

    # ------------------------------------------------------------------ #
    # Inductors L1 vs L2 (aligned columns)
    # ------------------------------------------------------------------ #
    print("\nINDUCTORS  —  L1 (inverter side)  vs  L2 (grid side)")
    print(line)
    print(f"{'Quantity':<{LBL}}{'Unit':<{UNT}}{'L1':>{COL}}{'L2':>{COL}}{'L1/L2':>{RAT}}")
    print(line)

    print("[ Core geometry ]")
    row("Inductance",     "[µH]",  v(L1, "L"),  v(L2, "L"),  scale=1e6, nfmt="{:.3f}", ratio=True)
    row("Turns N",        "[-]",   v(L1, "N"),  v(L2, "N"),  nfmt="{:.0f}", ratio=True)
    row("Air gap",        "[mm]",  v(L1, "lg"), v(L2, "lg"), scale=1e3, nfmt="{:.3f}", ratio=True)
    row("B_peak",         "[T]",   v(L1, "B_peak"), v(L2, "B_peak"), nfmt="{:.4f}", ratio=True)
    # B_peak / Bsat as a percentage
    bp1 = v(L1, "B_peak"); bs1 = v(L1, "Bsat")
    bp2 = v(L2, "B_peak"); bs2 = v(L2, "Bsat")
    pct1 = (bp1 / bs1) if (bp1 is not None and bs1) else None
    pct2 = (bp2 / bs2) if (bp2 is not None and bs2) else None
    row("B_peak / Bsat",  "[%]",   pct1, pct2, scale=100.0, nfmt="{:.2f}")
    row("Ae (eff. area)", "[mm²]", v(L1, "Ae"), v(L2, "Ae"), scale=1e6, nfmt="{:.2f}", ratio=True)
    row("le (path len)",  "[mm]",  v(L1, "le"), v(L2, "le"), scale=1e3, nfmt="{:.2f}", ratio=True)
    row("Ve (volume)",    "[cm³]", v(L1, "Ve"), v(L2, "Ve"), scale=1e6, nfmt="{:.2f}", ratio=True)
    row("A_surface",      "[cm²]", v(L1, "A_surface"), v(L2, "A_surface"), scale=1e4, nfmt="{:.2f}", ratio=True)

    print("[ Winding ]")
    row("I_RMS",          "[A]",   v(L1, "I_RMS"), v(L2, "I_RMS"), nfmt="{:.2f}", ratio=True)
    row("V_RMS",          "[V]",   v(L1, "V_RMS"), v(L2, "V_RMS"), nfmt="{:.4f}", ratio=True)
    row("Rdc",            "[mΩ]",  v(L1, "Rdc"), v(L2, "Rdc"), scale=1e3, nfmt="{:.4f}", ratio=True)
    row("Parallel strands", "[-]", v(L1, "N_parallel"), v(L2, "N_parallel"), nfmt="{:.0f}", ratio=True)
    row("A_wire actual",  "[mm²]", v(L1, "A_wire"), v(L2, "A_wire"), scale=1e6, nfmt="{:.2f}", ratio=True)
    row("Mean turn len",  "[mm]",  v(L1, "l_turn"), v(L2, "l_turn"), scale=1e3, nfmt="{:.2f}", ratio=True)

    print("[ Power losses ]")
    row("Core loss  P_c",    "[W]", v(L1, "P_core"),    v(L2, "P_core"),    nfmt="{:.4f}", ratio=True)
    row("Winding loss P_w",  "[W]", v(L1, "P_winding"), v(L2, "P_winding"), nfmt="{:.4f}", ratio=True)
    row("Total loss P_tot",  "[W]", v(L1, "P_total"),   v(L2, "P_total"),   nfmt="{:.4f}", ratio=True)

    print("[ Thermal ]")
    row("Thermal R_th",   "[K/W]", v(L1, "R_th"), v(L2, "R_th"), nfmt="{:.4f}", ratio=True)
    row("Temperature",    "[°C]",  v(L1, "T"), v(L2, "T"), offset=-K_to_C, nfmt="{:.2f}")
    row("Rated temp",     "[°C]",  v(L1, "T_rated"), v(L2, "T_rated"), offset=-K_to_C, nfmt="{:.2f}")
    # Margin = T_rated - T_operating
    m1 = (v(L1, "T_rated") - v(L1, "T")) if (v(L1, "T_rated") is not None and v(L1, "T") is not None) else None
    m2 = (v(L2, "T_rated") - v(L2, "T")) if (v(L2, "T_rated") is not None and v(L2, "T") is not None) else None
    row("Margin below rated", "[K]", m1, m2, nfmt="{:.2f}")

    print("[ Lifetime ]")
    row("Lifetime",          "[yr]", v(L1, "Lifetime"), v(L2, "Lifetime"), nfmt="{:.4f}", ratio=True)
    row("Lifetime consumed", "[%]",  v(L1, "Lifetime_consumed"), v(L2, "Lifetime_consumed"), nfmt="{:.3e}")

    print("=" * len(line))
C_report = dict(C=C_specs["C"], R_th=C_specs["Thermal_resistance_C"], V_RMS=V_C_RMS, V_RMS_rated=C_specs["V_C_RMS_Rated"], I_RMS=I_C_RMS, P_total=P_total_C, T=T_C, T_rated=C_specs["T_C_Rated"], Lifetime=Lifetime_C,)
L1_report = dict(L=L1_specs["L1"], N=N_L1, lg=lg_L1, B_peak=B_peak_L1, B_max=L1_specs["B_max"], Bsat=L1_specs["Bsat"], Ae=Ae_L1, le=le_L1, Ve=Ve_L1, A_surface=A_surface_L1, I_RMS=I_L1_RMS, V_RMS=V_L1_RMS, Rdc=Rdc_L1, N_parallel=N_parallel_wire_L1, A_wire=A_wire_actual_L1, l_turn=l_turn_L1, P_core=P_c_L1, P_winding=P_w_L1, P_total=P_total_L1, R_th=R_th_L1, T=T_inductor_L1, T_rated=L1_specs["T_insulation_rated"], Lifetime=Lifetime_L1, Lifetime_consumed=Lifetime_consumed_L1,)
L2_report = dict(L=L2_specs["L2"], N=N_L2, lg=lg_L2, B_peak=B_peak_L2, B_max=L2_specs["B_max"], Bsat=L2_specs["Bsat"], Ae=Ae_L2, le=le_L2, Ve=Ve_L2, A_surface=A_surface_L2, I_RMS=I_L2_RMS, V_RMS=V_L2_RMS, Rdc=Rdc_L2, N_parallel=N_parallel_wire_L2, A_wire=A_wire_actual_L2, l_turn=l_turn_L2, P_core=P_c_L2, P_winding=P_w_L2, P_total=P_total_L2, R_th=R_th_L2, T=T_inductor_L2, T_rated=L2_specs["T_insulation_rated"], Lifetime=Lifetime_L2, Lifetime_consumed=Lifetime_consumed_L2, )
compare_components(C_report, L1_report, L2_report)
'''

