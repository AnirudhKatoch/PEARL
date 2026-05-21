import numpy as np
from matplotlib import pyplot as plt
from Input_parameters import Input_parameters_class
from Calculation_functions import Calculation_functions_class
from LCL_filter_design import LCL_filter_design_function

params = Input_parameters_class()
Calculation_functions = Calculation_functions_class()

Vdc_rated = params.Vdc_rated; Vo_rated = params.Vo_rated; inverter_phases = params.inverter_phases; M_rated = params.M_rated; single_phase_inverter_topology = params.single_phase_inverter_topology; waveform_voltage_definition = params.waveform_voltage_definition; modulation_scheme = params.modulation_scheme; f = params.f; fsw = params.fsw; T = params.T; Tsw = params.Tsw; omega = params.omega
Profile_size = params.Profile_size; Vdc_RMS = params.Vdc_RMS; M = params.M; Vo = params.Vo; Vg_RMS = params.Vg_RMS; S_RMS = params.S_RMS; pf = params.pf; P_RMS = params.P_RMS; Q_RMS = params.Q_RMS
T_amb = params.T_amb
resolution_per_cycle = params.resolution_per_cycle; dt = params.dt; samples_per_switching_period = params.samples_per_switching_period; Minimum_required_samples_per_switching_period = params.Minimum_required_samples_per_switching_period
L1 = params.L1; R1 = params.R1
C = params.C;  I_C_RMS_rated = params.I_C_RMS_rated; Thermal_resistance_C = params.Thermal_resistance_C;  tan_delta_0 = params.tan_delta_0; Rs = params.Rs; R3 = params.R3; T_C_Rated = params.T_C_Rated; V_C_RMS_Rated = params.V_C_RMS_Rated; V_C_Peak_Rated = params.V_C_Peak_Rated; lifetime_curves_capacitor = params.lifetime_curves_capacitor
L2 = params.L2; R2 = params.R2
Vg_ll_RMS = params.Vg_ll_RMS; S_rated = params.S_rated; I_rated = params.I_rated; current_ripple_limit = params.current_ripple_limit; delta = params.delta; omega_sw = params.omega_sw


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
# L1, L2, C, and R3 can be provided directly by the user.
# If not provided, this function calculates recommended values from the design constraints.
# R1 and R2 are parasitic inductor resistances; ideally they are zero, but small measured/estimated values can be used.
L1_optimum, L2_optimum, C_optimum, R3_optimum = LCL_filter_design_function(Vg_ll_RMS=Vg_ll_RMS,
                                                                           S_rated=S_rated,
                                                                           I_rated =I_rated,
                                                                           fsw=fsw,
                                                                           omega_sw = omega_sw,
                                                                           fo=f,
                                                                           Udc_rated=Vdc_rated,
                                                                           M_rated=M_rated,
                                                                           inverter_phases=inverter_phases,
                                                                           modulation_scheme=modulation_scheme,
                                                                           print_values = False,
                                                                           current_ripple_limit=0.30,
                                                                           delta=0.19,
                                                                           num_C_values=10)

L2 = L2_optimum

# ----------------------------------------#
# Time profile
# ----------------------------------------#

t = np.arange(0, Profile_size, dt)                                                                                      # Create the simulation time vector from 0 to Profile_size seconds using the fixed time step dt
samples_per_second = resolution_per_cycle * f                                                                           # Number of simulation samples in one second, used to expand 1-second mission-profile values into time-domain waveforms

########################################################################################################################
# Electrical model
########################################################################################################################

Vg = np.sqrt(2) * np.repeat(Vg_RMS, samples_per_second) * np.sin(omega * t)                                             # Generate the instantaneous grid voltage waveform from the RMS grid-voltage profile
Ig_RMS = S_RMS / (3 * Vg_RMS)                                                                                           # Compute the RMS inverter output current required to deliver the specified apparent power to the grid

pf_inst = np.repeat(pf, samples_per_second)                                                                             # Expand the 1-second power-factor profile so it has the same length as the time vector
phi = np.arccos(np.abs(pf_inst))                                                                                        # Compute the power-factor phase angle from the magnitude of the power factor
phase_shift = np.sign(pf_inst) * phi                                                                                    # Apply the sign of the power factor to determine whether the current leads or lags the voltage
Ig_ref = np.sqrt(2) * np.repeat(Ig_RMS, samples_per_second) * np.sin(omega * t + phase_shift)                           # Generate the instantaneous current reference that the inverter should inject into the grid


#Vs_ref = Calculation_functions.Inverse_LCL_Filter_Grid_Connected_for_Vs(t=t, V_g=Vg, I_L2=Ig_ref, L1=L1, L2=L2, C=C, R1=R1, R2=R2,R3=R3)  # Calculate the inverter-side voltage reference required before the LCL filter so that I_L2 follows Ig_ref
Vs_ref = Calculation_functions.compute_Vs_ref_phasor(t=t, f=f, Ig_RMS=Ig_RMS, Vg_RMS=Vg_RMS, phase_shift=phase_shift,
                                                     L1=L1, L2=L2, C=C, R1=R1, R2=R2, R3=R3, Profile_size=Profile_size,
                                                     samples_per_second=samples_per_second)


# update validate_required_inverter_voltage
Calculation_functions.validate_required_inverter_voltage(Vs_ref=Vs_ref, Vo_available=Vo_rated)                          # Check whether the required inverter voltage is within the available PWM voltage capability


# Here I put Vs == Vs_ref. Vs_ref is the required output from the inverter and Vs is the switching output of the inverter.
# Usually there is a close loop controller which I will not built.
# With Vs the THD is too high but when Vs == Vs_ref the THD becomes zero which is unrealistic but its not my job to do that.
# My job is to find the voltage and current at each of the components of LCL but fix the control issues.
# Still try to fix and built a better PMW but for the time being just put Vs == Vs_ref  and move on

Vs = Calculation_functions.Sinusoidal_Pulse_Width_Modulation_One_Phase_updated(P_RMS, t, Vo, Vs_ref, Tsw, f)
#Vs = Vs_ref

V_L1, I_L1, V_C, I_C, V_L2, I_L2 = Calculation_functions.LCL_Filter_Grid_Connected(t=t, Vs=Vs, Vg=Vg, L1=L1, L2=L2, C=C, R1=R1, R2=R2, R3=R3) # Simulate the LCL filter response and obtain voltages and currents of L1, C, and L2

_, THD_percent, _, _, _ = Calculation_functions.compute_THD(t=t, signal=I_C, f=f, Location=r"Figures/I_L2_Harmonics.png", IL2_THD_plotting=True, max_plot_harmonic=20) # Compute the Total Harmonic Distortion of the grid-side current I_L2

#Calculation_functions.compare_I_ref_and_I_L2(t=t, I_L2=I_L2, Ig_ref=Ig_ref, f=f, dt=dt, resolution_per_cycle=resolution_per_cycle, save_path=f"Figures/Current_comparing_{pf[-1]}.png")
#Calculation_functions.plot_LCL_signals( t=t, V_L1=V_L1, I_L1=I_L1, V_C=V_C, I_C=I_C, V_L2=V_L2, I_L2=I_L2, resolution_per_cycle=resolution_per_cycle,save_path=f"Figures/{pf[-1]}.png")

########################################################################################################################
# Thermal model
########################################################################################################################

# ----------------------------------------#
# LCL filter middle branch [C]
# ----------------------------------------#

I_C_RMS_harmonics = Calculation_functions.compute_I_C_RMS_per_harmonic_for_capacitor(I_C=I_C, f=f, fsw=fsw, resolution_per_cycle=resolution_per_cycle, Profile_size=Profile_size)
P_total_C = Calculation_functions.Capacitor_total_power_losses(fsw=fsw,f=f, I_C_RMS_harmonics=I_C_RMS_harmonics,tan_delta_0=tan_delta_0,C=C,Rs=Rs)
T_C = Calculation_functions.Capacitor_hotspot_temperature(T_amb=T_amb, P_total_C=P_total_C, Thermal_resistance_C=Thermal_resistance_C)
V_C_RMS = Calculation_functions.Capacitor_voltage_RMS(V_C=V_C, resolution_per_cycle=resolution_per_cycle, f=f)
Calculation_functions.validate_capacitor_operating_limits(T_C=T_C, V_C_RMS=V_C_RMS, V_C=V_C, T_C_Rated=T_C_Rated, V_C_RMS_Rated=V_C_RMS_Rated, V_C_Peak_Rated=V_C_Peak_Rated)
L_capacitor = Calculation_functions.Capacitor_lifetime(T_C= T_C, V_C_RMS=V_C_RMS, V_C_RMS_Rated=V_C_RMS_Rated, lifetime_curves=lifetime_curves_capacitor)

# ----------------------------------------#
# LCL filter inverter side [L1]
# ----------------------------------------#

#df = pd.DataFrame({'t': t, 'V_L1': V_L1, 'I_L1': I_L1})
#df.to_parquet('Figures/L1_signals.parquet', index=False)
