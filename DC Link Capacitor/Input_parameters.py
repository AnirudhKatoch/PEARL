import numpy as np
from Calculation_functions import Calculation_functions_class

Calculation_functions = Calculation_functions_class()

class Input_parameters_class:

    def __init__(self, P_in, Q_in , T_env_in):

        '''

        The following electrical inputs are required run the capacitor lifetime simulation:

        - V_per_cap  : Capacitor Voltage
        - I_per_cap  : Capacitor Current

        These values may be provided directly by the user, or they can be computed from a full inverter setup using mission
        profile data. Using the inverter setup is often preferred, since these values can be extracted can be extracted directly
        from realistic operating conditions (Mission profiles of Active and reactive power of the inverter).

        '''

        # ----------------------------------------#
        # Inverter setup
        # ----------------------------------------#

        self.P = P_in
        self.Q = Q_in

        Profile_size = len(P_in)

        self.Vs = np.full(Profile_size, 230)   # [V] Inverter phase RMS AC side voltage
        self.V_dc = np.full(Profile_size, 800*0.7043) # [V] Inverter DC side voltage  #  800*0.9 for Case Study 1 , 800*0.7043 Case Study 2

        self.M = np.full(Profile_size, 1) # [-] Inverter modulation index # Modulation cannot be above 1 as model does not take into account. +

        modulation_scheme = "svm"  # options: "spwm" or "svm" , the type of modulation once can choose for three phase inverters."svm" is  Space Vector PWM (or Third-Harmonic Injection) and "spwm" is Sinusoidal PWM (reference = pure sine).
        if modulation_scheme not in ("spwm", "svm"):  # when inverter_phases == 1 this variable is invalid.
            raise ValueError("modulation_scheme must be 'spwm' or 'svm'")

        _, self.Is, self.phi, _, self.pf, self.M, self.S = Calculation_functions.compute_power_flow(P=self.P, Q=self.Q,
                                                                                                    V_dc=self.V_dc,
                                                                                                    Vs=self.Vs, M=self.M,
                                                                                                    modulation_scheme=modulation_scheme)

        # ----------------------------------------#
        # Ambient temperatures
        # ----------------------------------------#

        self.Plotting_flag = False  # True for plotting and False for not plotting

        # ----------------------------------------#
        # Ambient temperatures
        # ----------------------------------------#

        self.T_env = np.full(Profile_size, T_env_in) # use this for case study 1

        #self.T_env = np.full(Profile_size, 298.15 + 13.661)  # use this for case study 1

        # ----------------------------------------#
        # Number of capacitors in parallel and series
        # ----------------------------------------#

        self.N_series = 1  # Number of Capacitors in series
        self.N_parallel = 11  # Number of Capacitors in parallel

        # Capacitor chosen : B32778J8277K000

        # ----------------------------------------#
        # Capacitor Type
        # ----------------------------------------#

        self.capacitor_type = "film" # "electrolytic" or "film"

        # ----------------------------------------#
        # Capacitor rated Voltage,current and current limits
        # ----------------------------------------#

        self.Max_voltage_datasheet_cap = 800   # [V]
        self.Max_current_datasheet_cap  = 70.5 # [A]
        self.Max_temperature_cap_dict = { 1.0:343.15,
                                          0.89:358.15,
                                          0.83:363.15,
                                          0.775:368.15,
                                          0.725:373.15,
                                          0.65:378.15} # V/V_r [V]: Max temperature[K]

        # ----------------------------------------#
        # Rated values of capacitor
        # ----------------------------------------#

        self.Rated_voltage_datasheet_cap = 800   # [V]
        self.Rated_current_datasheet_cap  = 70.5 # [A]
        self.Rated_temperature_cap = 343.15      # [K]
        self.Rated_lifetime = 1e5                # [hours]

        # ----------------------------------------#
        # Capacitor dimensions
        # ----------------------------------------#

        Width = 130e-3    # [m]
        Height = 58e-3   # [m]
        Length = 57.5e-3 # [m]

        # ----------------------------------------#
        # Thermal resistance of capacitor
        # ----------------------------------------#

        Heat_coefficient = 300e-3  # [W/°C]
        self.Thermal_resistance = 1/Heat_coefficient # [°C/W] or [K/W]

        # ----------------------------------------#
        # Effective capacitor ESR at inverter switching frequency
        # ----------------------------------------#

        f_sw = 10 * 1000  # [Hz] Inverter switching frequency
        self.ESR_eff = 1.2e-3 # [Ohm]

        # ----------------------------------------#
        # Minimum insulation resistance of capacitor (from RC time constant)
        # ----------------------------------------#

        #Insulation resistance via a time constant
        Time_constant = 10000
        Capacitance_cap = 270e-6 # F
        #We know that Time_constant = Capacitance_cap * m   inimum_insulation_resistance
        self.minimum_insulation_resistance = Time_constant/Capacitance_cap # Value needs to calculate leakage current,
        # If available leakage current can be given directly as an input at different voltage levels

        # ----------------------------------------#
        # Lifetime Model Parameters
        # ----------------------------------------#

        # Graph-Based  Lifetime model

        self.lifetime_graph_dictionary = {
            70 + 273.15: {"V_ratio": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
                          "L_hours": [1e5, 1.75e5, 2.3e5, 3.3e5, 4.8e5, 6.9e5]},

            85 + 273.15: {"V_ratio": [0.89, 0.8, 0.7, 0.6, 0.5],
                          "L_hours": [7e4, 1e5, 1.5e5, 2e5, 2.9e5]},

            90 + 273.15: {"V_ratio": [0.83, 0.8, 0.7, 0.6, 0.5],
                          "L_hours": [6.5e4, 7.1e4, 1e5, 1.5e5, 2.1e5]},

            95 + 273.15: {"V_ratio": [0.775, 0.7, 0.6, 0.5],
                          "L_hours": [6e4, 8e4, 1.25e5, 1.65e5]},

            100 + 273.15: {"V_ratio": [0.72, 0.7, 0.6, 0.5],
                           "L_hours": [5.6e4, 6e4, 8.5e4, 1.25e5]},

            105 + 273.15: {"V_ratio": [0.66, 0.6, 0.5],
                           "L_hours": [5.25e4, 6.4e4, 9e4]}}





