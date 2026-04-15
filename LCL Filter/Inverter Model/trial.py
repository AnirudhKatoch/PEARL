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
M = np.full(Profile_size, 0.8)    # [-] Mission profile of modulation index at a resolution of 1 sec
Vo = np.full(Profile_size, 400)

# -------------------------
# Time vector for simulation
# -------------------------

mission_profile_size = len(Vdc)
points_per_cycle = 1750  # resolution per cycle
t = np.linspace(0, mission_profile_size, mission_profile_size * f * points_per_cycle ,endpoint=False)

# -------------------------
# PWM output voltage waveform
# -------------------------


V_s = Functions.Sinusoidal_Pulse_Width_Modulation_One_Phase(t=t ,M=M ,f=f ,Tsw=Tsw ,Vo=Vo ,T=T ,Vdc=Vdc )


