import numpy as np
from matplotlib import pyplot as plt
from Input_parameters import Input_parameters_class
from Calculation_functions import Calculation_functions_class

params = Input_parameters_class()
Calculation_functions = Calculation_functions_class()

Vdc_rated = params.Vdc_rated; Vo_rated = params.Vo_rated; inverter_phases = params.inverter_phases; M_rated = params.M_rated; single_phase_inverter_topology = params.single_phase_inverter_topology; waveform_voltage_definition = params.waveform_voltage_definition; modulation_scheme = params.modulation_scheme; f = params.f; fsw = params.fsw; T = params.T; Tsw = params.Tsw; omega = params.omega
Profile_size = params.Profile_size; Vdc_RMS = params.Vdc_RMS; M = params.M; Vo = params.Vo; Vg_RMS = params.Vg_RMS; S_RMS = params.S_RMS; pf = params.pf; P_RMS = params.P_RMS; Q_RMS = params.Q_RMS
resolution_per_cycle = params.resolution_per_cycle; dt = params.dt; samples_per_switching_period = params.samples_per_switching_period; Minimum_required_samples_per_switching_period = params.Minimum_required_samples_per_switching_period
L1 = params.L1; R1 = params.R1; C = params.C; L2 = params.L2; R2 = params.R2

Calculation_functions.validate_pwm_pulse_amplitude(Vdc_rated=Vdc_rated, inverter_phases=inverter_phases, single_phase_inverter_topology=single_phase_inverter_topology,waveform_voltage_definition=waveform_voltage_definition, Vo=Vo)
Calculation_functions.validate_ac_rms_voltage_limit(Vdc_rated=Vdc_rated, M=M, inverter_phases=inverter_phases, modulation_scheme=modulation_scheme, single_phase_inverter_topology=single_phase_inverter_topology, Vg_RMS=Vg_RMS)
Calculation_functions.validate_simulation_resolution(samples_per_switching_period=samples_per_switching_period, Minimum_required_samples_per_switching_period=Minimum_required_samples_per_switching_period)


#########################################################################################################################

# ----------------------------------------#
# Optimum LCL Filter
# ----------------------------------------#

#Make a fucntion to find what is optimum values of L1,C,L2,R1 and R2
#and then check the input L1,C,L2,R1 and R2 is as per the required values, just make a warning like
#hey yours values are not optimum, just a warning not a value error


Vg_ll_RMS = 690                            # [V] RMS of fundamental line-to-line grid voltage
S_rated = 1e6                              # [VA] Rated apparent Inverter  power
fsw = fsw                                  # [Hz] Inverter Switching Frequency
fo = f                                     # [Hz] Grid frequency or fundamental frequency
I_rated = S_rated/(np.sqrt(3) * Vg_ll_RMS) # [Hz] Rated Inverter  current
Udc_rated = Vdc_rated                      # [V] Max DC side voltage
M_rated = M_rated                          # [-] Rated modulation index
inverter_phases = inverter_phases          # [-] Number of inverter phases
modulation_scheme = modulation_scheme      # [-] Modulation scheme
current_ripple_limit = 0.30                # [-] Current ripple is usually limited to 20%–30% of rated current.Here 30% is used.
delta = 0.20                               # [-] 20% initial harmonic attenuation ratio recommended for LCL filter design
omega_sw = 2 * np.pi * fsw                 # [rad/s] Switching angular frequency


# Choosing Capactitor
C_max = 0.05 * S_rated / (2 * np.pi * fo * (Vg_ll_RMS ** 2))    # [F] Capacitor's max capacitance value

# Choosing total inductance which is L1 + L2
L_T_max = 0.10 * (Vg_ll_RMS ** 2) / (2 * np.pi * fo * S_rated)  # [H] Total inductance value of the two capacitors


# Now for choosing L1
if inverter_phases == 1:
    if modulation_scheme == "spwm":
        r = 2 # Single-phase bipolar SPWM,  Use r = 2 for bipolar SPWM
        L1_min = Udc_rated / (current_ripple_limit * I_rated * r * fsw)
    else:
        raise ValueError("For single-phase inverter, only SPWM is currently supported.")

elif inverter_phases == 3:
    if modulation_scheme == "spwm" or modulation_scheme == "svm":
        L1_min = ((np.sqrt(3) / 12) * (Udc_rated / (current_ripple_limit * I_rated * fsw)) * M_rated)
    else:
        raise ValueError("modulation_scheme must be 'spwm' or 'svm'.")
else:
    raise ValueError("inverter_phases must be 1 or 3.")


# ----------------------------------------#
# Now choosing L2
# Only satisfy total inductance constraint
# Without for/while loop
# ----------------------------------------#

# Capacitor candidates from C_max/2 downward with 10% steps

# Number of capacitor divisions
num_C_values = 100
C_candidates = C_max * (np.arange(1, num_C_values + 1) / num_C_values)
aL_candidates = np.abs(((1 / delta) - 1)/(1 - L1_min * C_candidates * omega_sw**2))
L2_candidates = aL_candidates * L1_min
L_total_candidates = L1_min + L2_candidates

# Find capacitor values that satisfy:
# L_total <= L_T_max

valid_indices = np.where(L_total_candidates <= L_T_max)[0]
C_target = C_max / 2 # Desired capacitor target: # closest to C_max/2

if len(valid_indices) > 0:
    valid_C_values = C_candidates[valid_indices] # Extract only valid capacitor candidates
    filtered_indices = np.where((valid_C_values >= C_target) & (valid_C_values <= C_max))[0] # Keep only values between: C_max/2 and C_max

    if len(filtered_indices) > 0:
        filtered_C_values = valid_C_values[filtered_indices] # Capacitor values within desired range
        best_local_index = np.argmin(np.abs(filtered_C_values - C_target)) # Find value closest to C_max/2
        C = filtered_C_values[best_local_index] # Actual capacitor value selected
        global_index = valid_indices[filtered_indices[best_local_index]] # Recover original global index
        L2_calculated = L2_candidates[global_index] # Corresponding L2 and total inductance
        L_total = L_total_candidates[global_index]

        #print("Valid design found.\n")

        #print(f"C = {C}")
        #print(f"L2 = {L2_calculated}")
        #print(f"L_total = {L_total}")

    else:
        print("No valid capacitor value exists between C_max/2 and C_max.")
else:
    print("No capacitor value satisfies the total inductance constraint.")

C = C                                                            # Final Chosen capacitor value
L1 = L1_min                                                      # Inverter side inductor
L2 = L1 * (np.abs(((1 / delta) - 1)/(1 - L1 * C * omega_sw**2))) # Grid side inductor
if (L1 + L2) > L_T_max:
    raise ValueError("L1 + L2 exceeds L_T_max.")

#Now put the fr constraint within the loop

fr = np.sqrt((L1+L2)/(L1*L2*C))






'''


valid_indices = np.where(L_total_candidates <= L_T_max)[0]

if len(valid_indices) > 0:

    idx = valid_indices[0]

    C = C_candidates[idx]
    aL = aL_candidates[idx]
    L2_calculated = L2_candidates[idx]
    L_total = L_total_candidates[idx]

    print("Valid total inductance found.")
    print("C =", C)
    print("aL =", aL)
    print("L1 =", L1_min)
    print("L2 =", L2_calculated)
    print("L_total =", L_total)
    print("L_T_max =", L_T_max)

else:
    print("Could not satisfy total inductance constraint.")

'''









#########################################################################################################################

'''


# ----------------------------------------#
# Time profile
# ----------------------------------------#

t = np.arange(0, Profile_size, dt)                                                                                      # Create the simulation time vector from 0 to Profile_size seconds using the fixed time step dt
samples_per_second = resolution_per_cycle * f                                                                           # Number of simulation samples in one second, used to expand 1-second mission-profile values into time-domain waveforms

# ----------------------------------------#
# Electrical model
# ----------------------------------------#

Vg = np.sqrt(2) * np.repeat(Vg_RMS, samples_per_second) * np.sin(omega * t)                                             # Generate the instantaneous grid voltage waveform from the RMS grid-voltage profile
Ig_RMS = S_RMS / (3 * Vg_RMS)                                                                                                 # Compute the RMS inverter output current required to deliver the specified apparent power to the grid



pf_inst = np.repeat(pf, samples_per_second)                                                                             # Expand the 1-second power-factor profile so it has the same length as the time vector
phi = np.arccos(np.abs(pf_inst))                                                                                        # Compute the power-factor phase angle from the magnitude of the power factor
phase_shift = np.sign(pf_inst) * phi                                                                                    # Apply the sign of the power factor to determine whether the current leads or lags the voltage
Ig_ref = np.sqrt(2) * np.repeat(Ig_RMS, samples_per_second) * np.sin(omega * t + phase_shift)                           # Generate the instantaneous current reference that the inverter should inject into the grid

Vs_ref = Calculation_functions.Inverse_LCL_Filter_Grid_Connected_for_Vs(t=t, V_g=Vg, I_L2=Ig_ref, L1=L1, L2=L2, C=C, R1=R1, R2=R2)  # Calculate the inverter-side voltage reference required before the LCL filter so that I_L2 follows Ig_ref
Calculation_functions.validate_required_inverter_voltage(Vs_ref=Vs_ref, Vo_available=Vo_rated)                          # Check whether the required inverter voltage is within the available PWM voltage capability

Vs = Calculation_functions.Sinusoidal_Pulse_Width_Modulation_One_Phase(P_RMS=P_RMS, t=t, Vo=Vo, Vs_ref=Vs_ref, Tsw=Tsw) # Generate the actual switched PWM inverter voltage waveform from the continuous voltage reference Vs_ref

V_L1, I_L1, V_C, I_C, V_L2, I_L2 = Calculation_functions.LCL_Filter_Grid_Connected(t=t, Vs=Vs, Vg=Vg, L1=L1, L2=L2, C=C, R1=R1, R2=R2) # Simulate the LCL filter response and obtain voltages and currents of L1, C, and L2

_, THD_percent, _, _, _ = Calculation_functions.compute_THD(t=t, signal=I_L2, f=f, Location=r"Figures/I_L2_Harmonics.png", IL2_THD_plotting=False, max_plot_harmonic=20) # Compute the Total Harmonic Distortion of the grid-side current I_L2

'''