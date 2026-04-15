from Calculation_functions import Calculation_functions_class
from Input_parameters import Input_parameters_class
import numpy as np
from matplotlib import pyplot as plt

Calculation_functions = Calculation_functions_class()

S_in = 1e6
pf_in = 1

P_in = abs(S_in*pf_in)
Q_in = np.sqrt(S_in**2 - P_in**2)
T_env_in = 273.15 + 25
Profile_size = 2

params = Input_parameters_class(P_in=np.full(Profile_size,P_in) ,Q_in=np.full(Profile_size,Q_in) ,T_env_in=np.full(Profile_size,T_env_in))

P = params.P; Q = params.Q; Vs = params.Vs; V_dc = params.V_dc; M = params.M; Is = params.Is; phi = params.phi; pf = params.pf; S = params.S; inverter_phases = params.inverter_phases
T_env = params.T_env
f_sw = params.f_sw; f_gf = params.f_gf; omega = params.omega
THD_input_I = params.THD_input_I; THD_input_V = params.THD_input_V; dt = params.dt

I_instantaneous = Calculation_functions.synthetic_profile(THD_input=THD_input_I, dt=dt, rms_values=Is, phi=phi, omega=omega,
                          harmonic_orders = [5, 7, 11, 13],
                          harmonic_weights = [0.5, 0.3, 0.15, 0.05],
                          harmonic_phases = [0, 0, 0, 0])

V_instantaneous = Calculation_functions.synthetic_profile(THD_input=THD_input_V, dt=dt, rms_values=Vs, phi=phi, omega=omega,
                          harmonic_orders = [5, 7],
                          harmonic_weights = [0.7, 0.3],
                          harmonic_phases = [0, 0])























'''
#################################################
# LCL design trial
#################################################

P_rated = 1e6     # [W] rated active power
V_ph_rated = 230  # [V] Rated phase RMS grid voltage
V_dc_rated = 600  # [V] Rated Inverter DC side voltage

Capacitor_design = "delta"  # 3-phase capacitor connection: "wye" = phase-to-neutral, "delta" = line-to-line

# RMS voltage across each capacitor
if Capacitor_design.lower() == "wye":
    U_cap_rms = V_ph_rated
elif Capacitor_design.lower() == "delta":
    U_cap_rms = np.sqrt(3) * V_ph_rated
else:
    raise ValueError("Capacitor_design must be 'wye' or 'delta'")



# RATED CURRENT (per-phase RMS)
I_rated = P_rated / (3 * V_ph_rated)

############################################
# Constraints for choosing LCL-filter parameters
############################################

# Filter capacitor constraint
# The designed capacitor value C should satisfy: C <= C_max
# This ensures that the reactive power absorbed by the capacitor remains below 5% of the rated active power (design guideline)
C_max = 0.05 * (P_rated / (2 * np.pi * f_gf * U_cap_rms**2)) # Farads (F)

# Total inductance constraint
# The total inductance L_T = L1 + L2 should satisfy: L_T <= L_T_max
# This ensures that the voltage drop across the filter inductors remains below 10% of the grid voltage (design guideline)
L_T_max = 0.10 * (U_cap_rms**2 / (2 * np.pi * f_gf * P_rated)) # Henry (H)


if inverter_phases == 1:
    # --- Single-phase inverter (unipolar / bipolar SPWM) ---
    # L1 ≥ U_dc / ((20% ~ 30%) * I_rated * f_sw)
    # The computed value is the MINIMUM required inductance.
    # The chosen L_inverter_side in design should be >= this value to ensure current ripple stays within the desired limit.
    k = 0.3  # choose between 0.2 and 0.3 depending on design (20%–30%)
    L_inverter_side = V_dc_rated / (k * I_rated * f_sw)

elif inverter_phases==3:
    # --- Three-phase two-level inverter (SPWM / SVM) ---
    # The computed value is the MINIMUM required inductance.
    # The chosen L_inverter_side in design should be >= this value to ensure current ripple stays within the desired limit.
    L_inverter_side = (np.sqrt(3) / (12 * 0.3)) * (V_dc_rated / (I_rated * f_sw)) * M.max()

'''

















