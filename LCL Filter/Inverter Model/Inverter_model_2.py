import numpy as np


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
M = np.full(Profile_size, 0.5)    # [-] Mission profile of modulation index at a resolution of 1 sec
Vo = np.full(Profile_size, 400)

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

V_g = 230 * np.sqrt(2) * np.sin(2 * np.pi * f * t)           # Instantaneous Voltage of the grid



t   = t
V_s = V_s
L1  = 100e-6
L2  = 100e-6
C   = 50e-6
R1 = 0.05
R2 = 0.05

V_L1, I_L1, V_C, I_C, V_L2, I_L2 = Functions.Solving_LCL_Filter_Grid_Connected(t=t ,V_s=V_s ,V_g=V_g ,L1=L1 ,L2=L2 ,C=C,R1=R1 ,R2=R2)

Functions.Plotting_Grid_Connected_LCL_filter(t=t ,V_L1=V_L1 ,I_L1=I_L1 ,V_C=V_C ,I_C=I_C ,V_L2=V_L2 ,I_L2=I_L2 , f=f)








'''
# Last two cycles mask
mask = t >= (t[-1] - (2 / f))




# Means (last 2 cycles)
print("mean(V_s) last 2 cycles  =", np.mean(V_s[mask]))
print("mean(V_g) last 2 cycles  =", np.mean(V_g[mask]))
print("mean(I_L1) last 2 cycles =", np.mean(I_L1[mask]))
print("mean(I_L2) last 2 cycles =", np.mean(I_L2[mask]))
print("mean(I_C) last 2 cycles  =", np.mean(I_C[mask]))

# KCL / KVL errors (last 2 cycles)
print("KCL error max (last 2 cycles) =",
      np.max(np.abs(I_L1[mask] - I_C[mask] - I_L2[mask])))

print("KVL left max (last 2 cycles) =",
      np.max(np.abs(V_s[mask] - R1 * I_L1[mask] - V_L1[mask] - V_C[mask])))

print("KVL right max (last 2 cycles) =",
      np.max(np.abs(V_C[mask] - R2 * I_L2[mask] - V_L2[mask] - V_g[mask])))

# Peak currents (last 2 cycles)
print("I_L1 peak (last 2 cycles) =", np.max(np.abs(I_L1[mask])))
print("I_L2 peak (last 2 cycles) =", np.max(np.abs(I_L2[mask])))
'''

'''
mask = t >= t[-1] - 2*(1/f)

print("mean(V_s) last 2 cycles  =", np.mean(V_s[mask]))
print("mean(I_L1) last 2 cycles =", np.mean(I_L1[mask]))
print("mean(I_L2) last 2 cycles =", np.mean(I_L2[mask]))
print("mean(I_C) last 2 cycles  =", np.mean(I_C[mask]))
'''

# Improve the formation of V_s
# I think when you are flipping the V_s from positive cycle to negative there are some un symmetries
# First fix that then we can look at optimum LCL and what to do with damping.
# Once this is done we will look for controlling the AC power of the whole inverter
# Then we will change it to Three phase
# Then we will look into phase changing