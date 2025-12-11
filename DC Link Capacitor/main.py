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
from numpy.f2py.crackfortran import param_eval

from Calculation_functions import Calculation_functions_class
from Input_parameters import Input_parameters_class
import numpy as np

Calculation_functions = Calculation_functions_class()

params = Input_parameters_class()

P = params.P; Q = params.Q; Vs = params.Vs; V_dc = params.V_dc; M = params.M; Is = params.Is; phi = params.phi; pf = params.pf; S = params.S
T_amb = params.T_amb
N_parallel = params.N_parallel; N_series = params.N_series
capacitor_type = params.capacitor_type
Max_voltage_datasheet_cap = params.Max_voltage_datasheet_cap; Max_current_datasheet_cap = params.Max_current_datasheet_cap; Max_temperature_cap_dict = params.Max_temperature_cap_dict
Rated_voltage_datasheet_cap = params.Rated_voltage_datasheet_cap; Rated_current_datasheet_cap = params.Rated_current_datasheet_cap; Rated_temperature_cap = params.Rated_temperature_cap; Rated_lifetime = params.Rated_lifetime
Thermal_resistance = params.Thermal_resistance
ESR_eff = params.ESR_eff
minimum_insulation_resistance = params.minimum_insulation_resistance
lifetime_graph_dictionary = params.lifetime_graph_dictionary

# ----------------------------------------#
# DC-link capacitor RMS ripple current from 3-phase inverter (Kolar model)
# ----------------------------------------#

Idcl =  Calculation_functions.capacitor_RMS_ripple_current(Is,M,phi)


# ----------------------------------------#
# Voltage and Current per capacitor
# ----------------------------------------#

V_per_cap = V_dc/N_series
I_per_cap = Idcl/N_parallel

# ----------------------------------------#
# Check each capacitor  Voltage and Current limits
# ----------------------------------------#

Calculation_functions_class.check_max_capacitor_voltage_and_current_limit(Max_voltage_datasheet_cap=Max_voltage_datasheet_cap,
                                                                          Max_current_datasheet_cap=Max_current_datasheet_cap,
                                                                          V_per_cap=V_per_cap, I_per_cap=I_per_cap)
# ----------------------------------------#
# Capacitor core temperature
# ----------------------------------------#

calibration_factor_core_temp = Calculation_functions.core_temperature_calibration_factor(I_per_cap=Rated_current_datasheet_cap,
                                                                   ESR_eff=ESR_eff,
                                                                   V_per_cap=Rated_voltage_datasheet_cap,
                                                                   minimum_insulation_resistance=minimum_insulation_resistance,
                                                                   T_core=Rated_temperature_cap,T_amb=298.15,Thermal_resistance=Thermal_resistance)

T_core = Calculation_functions.core_temperature_calculationsI_cap(I_per_cap=I_per_cap, ESR_eff=ESR_eff, V_per_cap=V_per_cap,
                                                minimum_insulation_resistance=minimum_insulation_resistance, T_amb= T_amb,
                                               Thermal_resistance=Thermal_resistance, calibration_factor_core_temp=calibration_factor_core_temp)


Calculation_functions_class.check_max_capacitor_temperature_limit(Max_temperature_cap_dict=Max_temperature_cap_dict,
                                                                  T_core=T_core,
                                                                  Rated_voltage_datasheet_cap=Rated_voltage_datasheet_cap,
                                                                  V_per_cap=V_per_cap )

# ----------------------------------------#
# Lifetime calculations
# ----------------------------------------#

# ----------------------------------------#
# Electrolytic Capacitor Lifetime Models
# ----------------------------------------#


Lifetime_model = "Graph_Based_lifetime"

model_dispatch = {"Nichion_lifetime": lambda L_r, T_r, T, Delta_t_r, I_r, K : Calculation_functions.Nichion_lifetime_model(L_r, T_r, T, Delta_t_r, I_r, K),
                  "Rubycon_lifetime": lambda L_r, T_r, T, Delta_t_r, Delta_t, V_r, V : Calculation_functions.Rubycon_lifetime_model(L_r, T_r, T, Delta_t_r, Delta_t, V_r, V),
                  "Panasonic_lifetime": lambda L_r,T_r,T_a : Calculation_functions.Panasonic_lifetime_model(L_r,T_r,T_a),
                  "Cornell_Dubilier_lifetime": lambda L_r, T_r, T, V_r, F, V : Calculation_functions.Cornell_Dubilier_lifetime_model(L_r, T_r, T, V_r, F, V),
                  "Faratronic_lifetime" : lambda L_r, T_r, V_r, V : Calculation_functions.Faratronic_lifetime_model(L_r, T_r, V_r, V),
                  "Generic_Arrhenius_lifetime" : lambda L_0,V,V_0,n,Ea,kB,T,T_0 : Calculation_functions.Generic_Arrhenius_lifetime_model(L_0,V,V_0,n,Ea,kB,T,T_0),
                  "Graph_Based_lifetime" : lambda T_core, V_ratio, lifetime_graph_dictionary : Calculation_functions.get_lifetime_from_graph(T_core, V_ratio, lifetime_graph_dictionary)}

# ----------------------------------------#
# Total Lifetime
# ----------------------------------------#

Life_cap = model_dispatch["Graph_Based_lifetime"]( T_core=T_core, V_ratio=(V_per_cap / Rated_voltage_datasheet_cap), lifetime_graph_dictionary=lifetime_graph_dictionary)

_, L_tot = Calculation_functions.miners_rule_lifetime(L_hours= Life_cap, Simulation_durations=len(V_per_cap))

print("Lifetime [years]:", L_tot)
