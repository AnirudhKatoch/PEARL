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
        self.V_dc = np.full(Profile_size, 800) # [V] Inverter DC side voltage

        self.M = np.full(Profile_size, 1) # [-] Inverter modulation index # Modulation cannot be above 1 as model does not take into account. +

        self.modulation_scheme = "svm"  # options: "spwm" or "svm" , the type of modulation once can choose for three phase inverters."svm" is  Space Vector PWM (or Third-Harmonic Injection) and "spwm" is Sinusoidal PWM (reference = pure sine).
        if self.modulation_scheme not in ("spwm", "svm"):  # when inverter_phases == 1 this variable is invalid.
            raise ValueError("modulation_scheme must be 'spwm' or 'svm'")

        self.inverter_phases = 3   # 1 or 3 (single-phase or three-phase)
        if self.inverter_phases not in (1, 3):
            raise ValueError("phases must be 1 or 3")

        self.single_phase_inverter_topology = "full"  # options: "half" or "full"  # One can choose is the single phase inverter half bridge or full bridge
        if self.single_phase_inverter_topology not in ("half", "full"):  # when inverter_phases == 3 this variable is invalid.
            raise ValueError("single_phase_inverter_topology must be 'half' or 'full'")

        _, self.Is, self.phi, _, self.pf, self.M, self.S = Calculation_functions.compute_power_flow(P=self.P, Q=self.Q,
                                                                                                    V_dc=self.V_dc,
                                                                                                    Vs=self.Vs, M=self.M,
                                                                                                    modulation_scheme=self.modulation_scheme,
                                                                                                    inverter_phases=self.inverter_phases,
                                                                                                    single_phase_inverter_topology=self.single_phase_inverter_topology)

        # ----------------------------------------#
        # Ambient temperatures
        # ----------------------------------------#

        self.T_env = T_env_in

        # ----------------------------------------#
        # Inverter switching frequency
        # ----------------------------------------#

        self.f_sw = 10 * 1000  # [Hz]

        # ----------------------------------------#
        # Grid frequency
        # ----------------------------------------#

        self.f_gf = 50  # [Hz]
        self.omega = 2 * np.pi * self.f_gf

        # =========================================================
        # Synthetic profile build
        # =========================================================

        self.THD_input_I = 0.2  # [-] THD for current waveform
        self.THD_input_V = 0.8  # [-] THD for voltage waveform
        self.dt = 0.02 / 100    # [s] Time step





