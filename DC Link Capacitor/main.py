from Calculation_functions import Calculation_functions_class
from Input_parameters import Input_parameters_class
import numpy as np
import pandas as pd
from Plotting_function import Plotting

Calculation_functions = Calculation_functions_class()

params = Input_parameters_class()

P = params.P; Q = params.Q; Vs = params.Vs; V_dc = params.V_dc; M = params.M; Is = params.Is; phi = params.phi; pf = params.pf; S = params.S
T_env = params.T_env
N_parallel = params.N_parallel; N_series = params.N_series
capacitor_type = params.capacitor_type
Max_voltage_datasheet_cap = params.Max_voltage_datasheet_cap; Max_current_datasheet_cap = params.Max_current_datasheet_cap; Max_temperature_cap_dict = params.Max_temperature_cap_dict
Rated_voltage_datasheet_cap = params.Rated_voltage_datasheet_cap; Rated_current_datasheet_cap = params.Rated_current_datasheet_cap; Rated_temperature_cap = params.Rated_temperature_cap; Rated_lifetime = params.Rated_lifetime
Thermal_resistance = params.Thermal_resistance
ESR_eff = params.ESR_eff
minimum_insulation_resistance = params.minimum_insulation_resistance
lifetime_graph_dictionary = params.lifetime_graph_dictionary

sim_dir, dataframes_dir, Figures_dir = Calculation_functions.create_simulation_folders()

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
                                                                   T_core=Rated_temperature_cap,T_env=298.15,Thermal_resistance=Thermal_resistance)

T_core = Calculation_functions.core_temperature_calculationsI_cap(I_per_cap=I_per_cap, ESR_eff=ESR_eff, V_per_cap=V_per_cap,
                                                minimum_insulation_resistance=minimum_insulation_resistance, T_env= T_env,
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

df = pd.DataFrame({ "S":S, "P":P, "Q":Q, "V_dc":V_dc, "Vs":Vs, "Is": Is, "pf":pf, "phi":phi,
                    "T_env":T_env,
                    "Idcl":Idcl,
                    "V_per_cap":V_per_cap, "I_per_cap":I_per_cap,
                    "T_core":T_core})

df.loc[df.index[0], ["N_series","N_parallel","L_tot"]] = [N_series,N_parallel,L_tot]

df.to_parquet(dataframes_dir / f"df.parquet", index=False,engine="pyarrow")


Plotting(df,Figures_dir)











