import numpy as np
from Calculation_functions import Calculation_functions_class

Calculation_functions = Calculation_functions_class()

class Input_parameters_class:

    def __init__(self):

        '''

        The following electrical inputs are required run the LCL filter lifetime simulation:

        Naming convention
        -----------------
        I_L1 : current through left inductor
        V_L1 : voltage across left inductor
        I_L2 : current through right inductor
        V_L2 : voltage across right inductor
        V_C  : capacitor voltage
        I_C  : capacitor current

        These values may be provided directly by the user, or they can be computed from a full inverter setup using mission
        profile data so simply using system level characteristics to calculate component characteristics.
         Using the inverter setup is often preferred, since these values can be extracted can be extracted directly
        from realistic operating conditions (Mission profiles of Active and reactive power of the inverter). Obviously one has to make some assumptions

        '''

        # ----------------------------------------#
        # System Parameters
        # ----------------------------------------#

        self.Vdc_rated = 2000                             # [V] Rated DC bus voltage, defines maximum available inverter voltage level
        self.Vo_rated = 1000                              # [V] Rated PWM pulse amplitude, instantaneous switched level (±Vo), must be ≤ allowed by topology
        self.inverter_phases = 3                          # [-] ["1" or "3"] Number of phases: 1 = single-phase inverter, 3 = three-phase inverter
        self.M_rated = 1                                  # [-] Modulation index, controls PWM pulse widths and sets fundamental output voltage magnitude (0 ≤ M ≤ 1 in linear region)
        self.single_phase_inverter_topology = "full"      # ["full" or "half"] Single-phase topology: "half" = ±Vdc/2 output, "full" = ±Vdc output (H-bridge)
        self.waveform_voltage_definition = "pole_voltage" # ["switched_output" or "pole_voltage"] Voltage meaning: "switched_output" = load voltage, "pole_voltage" = single leg voltage (±Vdc/2) # One phase inverter can have "switched_output" or "pole_voltage", Three phae inverter only has pole_voltage
        self.modulation_scheme = "spwm"                   # ["spwm" or "svm"] # PWM strategy used to generate switching signals; "spwm" = Sinusoidal PWM , "svm" = Space Vector Modulation; NOTE: current system supports only "spwm" and does NOT support "svm"
        self.f = 50                                       # [Hz] Fundamental frequency, desired AC output frequency of the inverter (e.g., grid frequency 50 Hz)
        self.fsw = 10000                                  # [Hz] Switching frequency, frequency at which PWM switches turn ON/OFF (carrier frequency)
        self.T = 1 / self.f                               # [s] Fundamental period, time for one full AC cycle (e.g., 20 ms for 50 Hz)
        self.Tsw = 1 / self.fsw                           # [s] Switching period, time for one PWM switching cycle (e.g., 100 µs for 10 kHz)
        self.omega = 2 * np.pi * self.f                   # [Hz] Angular frequency

        # -------------------------#
        # Mission profiles
        # -------------------------#

        self.Profile_size = 2                                                  # [s] Total duration of the mission profile; each array entry represents one operating point with 1-second resolution
        self.Vdc_RMS = np.full(self.Profile_size,self.Vdc_rated)                  # [V] Mission profile of DC bus voltage sampled at 1-second intervals
        self.M = np.full(self.Profile_size, 1)                         # [-] Mission profile of modulation index sampled at 1-second intervals
        self.Vo = np.full(self.Profile_size, self.Vo_rated)                    # [V] Mission profile of PWM pulse amplitude (instantaneous switched voltage level)
        self.Vg_RMS = np.full(self.Profile_size, 400)                  # [V] Mission profile of inverter/grid-side RMS AC voltage sampled at 1-second intervals
        self.S_RMS = np.full(self.Profile_size,1e6)                    # [VA] Mission profile of apparent power sampled at 1-second intervals
        self.pf = np.full(self.Profile_size,1)                         # [-] Mission profile of power factor; negative = inductive operation, positive = capacitive operation
        self.P_RMS = self.S_RMS * self.pf                                      # [W] Mission profile of active power computed from apparent power and power factor
        self.Q_RMS = self.S_RMS * np.sqrt(1 - self.pf ** 2) * np.sign(self.pf) # [Var] Mission profile of reactive power computed from apparent power and power factor sign

        # ----------------------------------------#
        # Ambient Temperature
        # ----------------------------------------#

        self.T_amb = np.full(self.Profile_size, 273+25)

        # ----------------------------------------#
        # Time Discretization and Simulation Resolution
        # ----------------------------------------#

        self.resolution_per_cycle = 3000                        # [-] Number of discrete simulation samples used to represent one fundamental AC cycle; higher values improve waveform fidelity but increase computational cost, This value is also very important for calculation actual value of Vs_ref
        self.dt = self.T / self.resolution_per_cycle            # [s] Simulation time-step size derived from resolution_per_cycle
        self.samples_per_switching_period = self.Tsw / self.dt  # [-] Number of simulation samples contained within one PWM switching period; determines PWM waveform resolution accuracy
        self.Minimum_required_samples_per_switching_period = 15 # [-] Minimum acceptable PWM numerical resolution required to accurately capture switching events and carrier intersections

        # ----------------------------------------#
        # LCL filter design papameters
        # ----------------------------------------#

        self.Vg_ll_RMS = 690                                        # [V] RMS of fundamental line-to-line grid voltage
        self.S_rated = 1e6                                          # [VA] Rated apparent Inverter  power
        self.I_rated = self.S_rated / (np.sqrt(3) * self.Vg_ll_RMS) # [Hz] Rated Inverter  current
        self.current_ripple_limit = 0.30                            # [-] Current ripple is usually limited to 20%–30% of rated current.Here 30% is used.
        self.delta = 0.19                                           # [-] 20% initial harmonic attenuation ratio recommended for LCL filter design
        self.omega_sw = 2 * np.pi * self.fsw                        # [rad/s] Switching angular frequency

        # ----------------------------------------#
        # LCL filter inverter side [Inductor properties]
        # ----------------------------------------#

        self.L1 = 115e-6

        # ----------------------------------------#
        # LCL filter inverter side [Resistor properties]
        # ----------------------------------------#

        self.R1 = 0.05



        # ----------------------------------------#
        # LCL filter grid side [Inductor properties]
        # ----------------------------------------#

        self.L2 = 100e-6

        # ----------------------------------------#
        # LCL filter grid side [Resistor properties]
        # ----------------------------------------#

        self.R2 = 0.05

        # ----------------------------------------#
        # LCL filter middle [Capacitor properties]
        # ----------------------------------------#

        # Product code - B32362A4157J080 # FilterCap MKD AC – Single phase # TDK Electronics

        self.C = 150e-6                     # [F] This is the capacitor's capacitance
        self.I_C_RMS_rated = 50             # [A] This is the capacitor rated current value
        self.Thermal_resistance_C = 3.2     # [K/W] This is the capacitor's thermal resistance [Not given hence it is assumed based on capacitor from the same company of the same dimension(The product from the value is copied is B32373F5127J030)]
        self.Rs = 1.9e-3                    # [Ohm] series resistance of the capacitor itself
        self.tan_delta_0 = 2e-4             # [-] Material property of polypropylene dielectric -NOT the tan δ from the product table # Source: IEC 60068, polypropylene material data
        self.T_C_Rated = 273 + 85           # [K] Capacitor max Hotspot temperature
        self.V_C_RMS_Rated = 480            # [V] Capacitor max RMS voltage
        self.V_C_Peak_Rated = 680           # [V] Capacitor max peak voltage

        self.lifetime_curves_capacitor = {1.3: {"T": np.array([273+85,   273+75,    273+70,  273+65,  273+60, 273+55,  273+50]),
                                                "L": np.array([0.15*1e5, 0.31*1e5,  0.5*1e5, 0.8*1e5, 1e5,    1.6*1e5, 2.25*1e5])},
                                          1.2: {"T": np.array([273+85,   273+75,  273+70,   273+65,  273+60, 273+55]),
                                                "L": np.array([0.26*1e5, 0.6*1e5, 0.85*1e5, 1.4*1e5, 1.9*1e5, 2.5*1e5 ])},
                                          1.1: {"T": np.array([273+85,  273+75,  273+70,   273+65]),
                                                "L": np.array([0.5*1e5, 1.1*1e5, 1.75*1e5, 2.4*1e5 ])},
                                          1.0: {"T": np.array([273+85, 273+75,  273+70]),
                                                "L": np.array([1e5,    2.1*1e5, 3*1e5])},
                                          0.9: {"T": np.array([273+85, 273+80]),
                                                "L": np.array([2*1e5, 2.8*1e5])},
                                          0.8: {"T": np.array([273+85,  273+80]),
                                                "L": np.array([4.5*1e5, 6*1e5])},}

        # ----------------------------------------#
        # LCL filter middle [Resistor properties]
        # ----------------------------------------#

        self.R3 = 0.010579                  # [Ohn] This is the series resistance in parallel to capacitor










