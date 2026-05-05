import numpy as np
import matplotlib.pyplot as plt

from All_the_functions import All_the_functions_class

Functions = All_the_functions_class()

# -------------------------
# System parameters
# -------------------------

Vdc_rated = 800                                  # [V] Rated DC bus voltage, defines maximum available inverter voltage level
Vo_rated = 400                                   # [V] Rated PWM pulse amplitude, instantaneous switched level (±Vo), must be ≤ allowed by topology
inverter_phases = 1                              # [-] ["1" or "3"] Number of phases: 1 = single-phase inverter, 3 = three-phase inverter
M_rated = 1                                      # [-] Modulation index, controls PWM pulse widths and sets fundamental output voltage magnitude (0 ≤ M ≤ 1 in linear region)
single_phase_inverter_topology = "full"          # ["full" or "half"] Single-phase topology: "half" = ±Vdc/2 output, "full" = ±Vdc output (H-bridge)
waveform_voltage_definition = "switched_output"  # ["switched_output" or "pole_voltage"] Voltage meaning: "switched_output" = load voltage, "pole_voltage" = single leg voltage (±Vdc/2)
modulation_scheme = "spwm"                       # ["spwm" or "svm"] # PWM strategy used to generate switching signals; "spwm" = Sinusoidal PWM , "svm" = Space Vector Modulation; NOTE: current system supports only "spwm" and does NOT support "svm"
f = 50                                           # [Hz] Fundamental frequency, desired AC output frequency of the inverter (e.g., grid frequency 50 Hz)
fsw = 10000                                      # [Hz] Switching frequency, frequency at which PWM switches turn ON/OFF (carrier frequency)
T = 1 / f                                        # [s] Fundamental period, time for one full AC cycle (e.g., 20 ms for 50 Hz)
Tsw = 1 / fsw                                    # [s] Switching period, time for one PWM switching cycle (e.g., 100 µs for 10 kHz)

Vo_rated, Vo_theoretical_max = Functions.validate_or_set_pulse_amplitude(Vdc_rated=Vdc_rated,inverter_phases=inverter_phases,
                                                       single_phase_inverter_topology=single_phase_inverter_topology,
                                                       waveform_voltage_definition=waveform_voltage_definition,Vo=Vo_rated)

Vs_RMS_max_theoretical = Functions.compute_theoretical_fundamental_rms_limit(Vdc_rated=Vdc_rated,M=M_rated,inverter_phases=inverter_phases,
                                                    modulation_scheme=modulation_scheme,single_phase_inverter_topology=single_phase_inverter_topology)

# -------------------------
# Mission profiles
# -------------------------

Profile_size = 1
Vdc = np.full(Profile_size, 800)  # [V] Mission profile of DC bus voltage at a resolution of 1 sec
M = np.full(Profile_size, 1)      # [-] Mission profile of modulation index at a resolution of 1 sec
Vo = np.full(Profile_size, 400)   # [V] PWM pulse amplitude
Pref_profile = np.array([1000.0])   # [W] active power reference, 1-second resolution
Qref_profile = np.array([0.0])      # [var] reactive power reference, 1-second resolution

# -------------------------
# Time vector for simulation
# -------------------------

mission_profile_size = len(Vdc)
points_per_cycle = 3000           # resolution per cycle  # This looks optimum for the time being change that later on with some more iteration

t = np.linspace(0, mission_profile_size, mission_profile_size * f * points_per_cycle )




# -------------------------
# PWM output voltage waveform
# -------------------------

V_s = Functions.Sinusoidal_Pulse_Width_Modulation_One_Phase(t=t ,M=M ,f=f ,Tsw=Tsw ,Vo=Vo ,T=T ,Vdc=Vdc )

#Functions.Plotting_PWM_Output_Voltage(t=t, V_s=V_s)

########################################################################################################################
# LCL filter simulation
########################################################################################################################

V_g = 230 * np.sqrt(2) * np.sin(2 * np.pi * f * t)      # Instantaneous Voltage of the grid
Vg_rms = np.sqrt(np.mean(V_g**2))
Ig_rms = Pref_profile/Vg_rms
Ig_rms_t = np.repeat(Ig_rms, points_per_cycle*f)
I_g = Ig_rms_t * np.sqrt(2) * np.sin(2 * np.pi * f * t)      # Instantaneous Voltage of the grid



t   = t
V_s = V_s
L1  = 100e-6
L2  = 100e-6
C   = 50e-6
R1 = 0.05
R2 = 0.05
I_L2_known = I_g

V_L1, I_L1, V_C, I_C, V_L2, I_L2 = Functions.Solving_LCL_Filter_Grid_Connected_Known_IL2(t=t, V_s=V_s, V_g=V_g, I_L2_known=I_L2_known, L1=L1, C=C, R1=R1, R2=R2, L2=L2)

Functions.Plotting_Grid_Connected_LCL_filter(t=t ,V_L1=V_L1 ,I_L1=I_L1 ,V_C=V_C ,I_C=I_C ,V_L2=V_L2 ,I_L2=I_L2 , f=f)

Functions.THD_and_harmonics(signal=I_L2,t_ss=t)