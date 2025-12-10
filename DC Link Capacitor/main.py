# This tool estimates capacitor lifetime from a mission profile using multiple
# industry models. Electrolytic capacitors are supported through four empirical
# lifetime equations (Nichicon, Rubycon, Panasonic, and a vendor-specific model).
# Film capacitors (metallized polypropylene) use the Cornell Dubilier and
# Faratronic lifetime formulations. In addition, two generic methods are
# available: (1) a universal Arrhenius voltage–temperature model for cases where
# activation energy and voltage exponent are known, and (2) a graph-based
# interpolation or machine-learning model that reconstructs lifetime directly
# from manufacturer derating curves. Any model can be selected by the user, and
# lifetime over the mission profile is computed using Miner’s rule by evaluating
# the chosen model across all operating conditions.



from Calculation_functions import Calculation_functions_class
from Input_parameters import Input_parameters_class


Idcl = (np.sqrt(2) * Is ) * np.sqrt(M * ((np.sqrt(3)/4*np.pi) + (np.sqrt(3)/np.pi - 9/16*M) * np.cos(phi)**2))    # [A rms]

# Voltage per capacitor
V_per_cap = V_dc/N_series
# Current per capacitor
I_per_cap = Idcl/N_parallel

#Capacitor chosen : B32778G8107


Calculation_functions_class.check_max_capacitor_current_limit(Max_voltage_datasheet_cap=Max_voltage_datasheet_cap,
                                                              Max_current_datasheet_cap=Max_current_datasheet_cap,
                                                              V_per_cap=V_per_cap, I_per_cap=I_per_cap)




P_ripple = I_per_cap**2 * ESR_eff
P_leak = I_leak * V_per_cap

T_c = T_amb + 1.0 * Thermal_resistance * (P_ripple + P_leak)

#DeltaT = 20.0 * (I_per_cap / Max_current_datasheet_cap)**2
#T_case = T_amb + DeltaT


# Running the formula

V0 = 800 # 70 degree celsius
n = 9
L0 = 100000
V = V_per_cap
E_a = 0.8 # eV
k_B = 8.617e-5 # eV/K
T = T_c
T0 = 343.13

# Film Capacitor
L = L0 * (V / V0)**(-n) * np.exp(E_a / k_B * (1/T - 1/T0))
L = L /(365*24)

print(L)

Lb = 100000
Va = V_per_cap
Vr = 800
Tm = 105+273.15
Tc = T_c


# Aluminum Electrolytic Capacitor
Lc = Lb * (4.3 - 3.3 * (Va / Vr)) * (2 ** ((Tm - Tc) / 10))
Lc = Lc /(365*24)
print(Lc)