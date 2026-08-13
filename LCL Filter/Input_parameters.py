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

        self.Profile_size = 365                                               # [s] Total duration of the mission profile; each array entry represents one operating point with 1-second resolution
        self.Vdc_RMS = np.full(self.Profile_size,self.Vdc_rated)               # [V] Mission profile of DC bus voltage sampled at 1-second intervals
        self.M = np.full(self.Profile_size, 1)                         # [-] Mission profile of modulation index sampled at 1-second intervals
        self.Vo = np.full(self.Profile_size, self.Vo_rated)                    # [V] Mission profile of PWM pulse amplitude (instantaneous switched voltage level)
        self.Vg_RMS = np.full(self.Profile_size, 400)                  # [V] Mission profile of inverter/grid-side RMS AC voltage sampled at 1-second intervals
        self.S_RMS = np.full(self.Profile_size,1e6)                       # [VA] Mission profile of apparent power sampled at 1-second intervals
        self.pf = np.full(self.Profile_size,1)                         # [-] Mission profile of power factor; negative = inductive operation, positive = capacitive operation
        self.P_RMS = self.S_RMS * self.pf                                      # [W] Mission profile of active power computed from apparent power and power factor
        self.Q_RMS = self.S_RMS * np.sqrt(1 - self.pf ** 2) * np.sign(self.pf) # [Var] Mission profile of reactive power computed from apparent power and power factor sign
        self.Ig_RMS = self.S_RMS / (self.inverter_phases * self.Vg_RMS)        # Compute the RMS inverter output current required to deliver the specified apparent power to the grid

        # ----------------------------------------#
        # Thermal related parameters
        # ----------------------------------------#

        self.T_amb = np.full(self.Profile_size, 273+25) # [K]
        self.heat_transfer_coefficient = 10             # [W/(m²·K)]
        # Typical values:
        # 10  W/(m²·K) — natural convection, still air; 50  W/(m²·K) — moderate forced air cooling
        # 250 W/(m²·K) — high-velocity forced air; 500 W/(m²·K) — Liquid Cooling Source: Incropera et al., Table 1.1

        # ----------------------------------------#
        # Time Discretization and simulation Resolution
        # ----------------------------------------#

        self.resolution_per_cycle = 2000                       # [-] Number of discrete simulation samples used to represent one fundamental AC cycle; higher values improve waveform fidelity but increase computational cost, This value is also very important for calculation actual value of Vs_ref
        self.dt = self.T / self.resolution_per_cycle            # [s] Simulation time-step size derived from resolution_per_cycle
        self.samples_per_switching_period = self.Tsw / self.dt  # [-] Number of simulation samples contained within one PWM switching period; determines PWM waveform resolution accuracy
        self.Minimum_required_samples_per_switching_period = 5  # [-] Minimum acceptable PWM numerical resolution required to accurately capture switching events and carrier intersections
        self.seconds_per_sample = 24 * 3600                     # [s] Wall-clock duration each mission-profile sample represents(e.g. 1 = per-second profile, 86400 = one sample per day)
        self.h_max = 500                                        # [-] The harmonic order until which one wants the simulation to be done
        # ----------------------------------------#
        # LCL filter design parameters
        # ----------------------------------------#

        self.Vg_ll_RMS = 690                                            # [V] RMS of fundamental line-to-line grid voltage
        self.S_rated = 1e6                                              # [VA] Rated apparent Inverter  power
        self.I_rated_RMS = self.S_rated / (np.sqrt(3) * self.Vg_ll_RMS) # [A] Rated Inverter  current
        self.I_rated_peak = np.sqrt(2) * self.I_rated_RMS               # [A] Peak current
        self.current_ripple_limit = 0.30                               # [-] Current ripple is usually limited to 20%–30% of rated current.Here 30% is used.
        self.delta = 0.19                                               # [-] 20% initial harmonic attenuation ratio recommended for LCL filter design
        self.omega_sw = 2 * np.pi * self.fsw                            # [rad/s] Switching angular frequency

        # ----------------------------------------#
        # LCL filter inverter side [Inductor properties]
        # ----------------------------------------#

        self.L1_specs = {

            # References
            # [A] Core datasheet    → part number 4216L1R-B
            # [B] Material datasheet → Metglas Inc. 2605SA1

            # Inductance
            'L1': 115e-6,                   # [H] Inductance

            # ── Core material (Metglas 2605SA1) ──────────────────────────────────
            'k': 0.00336922369454695*7180,  # [W/m³]  from [A] Table 5 sine row # We might need to use Sawtooth/Trapezoidal 50% duty, if there is too much error
            'a': 1.30103359460677,          # [-]     Steinmetz α from [A] Table 5 sine row # We might need to use Sawtooth/Trapezoidal 50% duty, if there is too much error
            'b': 2.13595976775746,          # [-]     Steinmetz β from [A] Table 5 sine row # We might need to use Sawtooth/Trapezoidal 50% duty, if there is too much error
            'rho_mass': 7180,               # [kg/m³] density. From [B] page 1 physical table
            'Bsat': 1.56,                   # [T]     saturation. From [B] page 1 electromagnetic table
            'B_max': 0.7 * 1.56,            # [T]     70% operating limit
            'mu_r': 3000,                   # [-]     relative permeability at 10 kHz. From [A] Fig 12a at 10kHz # Assumed constant for simplicity

            # ── Core geometry (scaled 4216L1R-B × 1.55) ──────────────────────────
            'A_core': 180e-3 * 1.90222,    # [m]  Overall width;         outer horizontal dimension;                    from [A] Table 1
            'B_core': 240e-3 * 1.90222,    # [m]  Overall height;        outer vertical dimension;                      from [A] Table 1
            'D_core': 30e-3 * 1.90222,     # [m]  Depth (cast width);    dimension going into the page;                 from [A] Table 1
            'E_core': 50e-3 * 1.90222,     # [m]  Thickness (build);                                                    from [A] Table 1
            'F_core': 80e-3 * 1.90222,     # [m]  Window width;          inner horizontal opening for winding;          from [A] Table 1
            'G_core': 140e-3 * 1.90222,    # [m]  Window height;         inner vertical opening for winding;            from [A] Table 1
            'kf': 0.82,                    # [-]  Stacking factor;                                                      from [A] Table 2

            # ── Winding (Elektrisola Amidester 200 A200 — IEC 60317-8 / NEMA MW 74) ──────
            # Source: Elektrisola Enamelled Copper Wire — Manufacturing Programme and Technical Data, Page 3 (product table) and Page 4 (dimensional table) https://www.mintex.si/wp-content/uploads/2018/06/CuL-englisch.pdf
            # Note: Rac ≈ Rdc assumed — conductor type not specified; skin and proximity effects neglected

            'J_max': 4e6,               # [A/m²]  Maximum current density = 4 A/mm²; standard design practice for naturally cooled inductors
            'd_strand': 0.500e-3,       # [m]     Strand diameter — largest available; Source: Elektrisola datasheet page 4, nominal diameter column
            'A_strand': 0.196350e-6,    # [m²]    Bare copper area of one strand;     Source: Elektrisola datasheet page 4, section column
            'rho': 1.709e-8,            # [ohm·m]   Copper resistivity at 20°C;         standard material constant

            # ── Lifetime (Elektrisola Amidester 200 A200) ──────────────────────
            # Source: Elektrisola Enamelled Copper Wire — Manufacturing Programme and Technical Data, Page 3, product table, thermal values row https://www.mintex.si/wp-content/uploads/2018/06/CuL-englisch.pdf
            # Model:  Arrhenius thermal aging — IEC 60216 Montanari, G.C., IEEE Electrical Insulation Magazine, Vol. 9, No. 5, 1993.

            'T_insulation_rated': 273 + 210,   # [K]   Temperature index of Amidester 200 — continuous operating temperature at which insulation reaches reference lifetime of 20,000 h
            'L_insulation_rated': 20000,       # [h]   Reference lifetime at T_ins_rated. Source: Elektrisola datasheet page 3 + IEC 60172 reference point
            'Ea_insulation': 1.1 * 1.602e-19,  # [J]   Activation energy = 1.1 eV converted to Joules Source: Emery, F.T., "Arrhenius model for insulation aging", IEEE Electrical Insulation Magazine — standard value for Class 200 polyesterimide; NOT from Elektrisola datasheet
            'kb': 1.381e-23,                   # [J/K] Boltzmann constant — physical constant (SI units, matches Ea in Joules)
            'V_bd': 2400,                      # [V]   Minimum breakdown voltage — Grade 1, cylinder test, 0.500mm wire. Source: Elektrisola datasheet page 4, minimum breakdown voltage column

            'mu_0' : 4 * np.pi * 1e-7,  # [H/m]  Permeability of free space (physical constant)

            # Inductor series resistance
            'R1' : 0.05,                # [ohm]

            'L_max_years' : 30          # [Years] Maximum lifetime possible
        }

        # ----------------------------------------#
        # LCL filter grid side [Inductor properties]
        # ----------------------------------------#

        self.L2_specs = {

            # References
            # [A] Core datasheet    → part number 4216L1R-B
            # [B] Material datasheet → Metglas Inc. 2605SA1

            # Inductance
            'L2': 6.54e-6,  # [H] Inductance

            # ── Core material (Metglas 2605SA1) ──────────────────────────────────
            'k': 0.00336922369454695 * 7180, # [W/m³]  from [A] Table 5 sine row # We might need to use Sawtooth/Trapezoidal 50% duty, if there is too much error
            'a': 1.30103359460677,           # [-]     Steinmetz α from [A] Table 5 sine row # We might need to use Sawtooth/Trapezoidal 50% duty, if there is too much error
            'b': 2.13595976775746,           # [-]     Steinmetz β from [A] Table 5 sine row # We might need to use Sawtooth/Trapezoidal 50% duty, if there is too much error
            'rho_mass': 7180,                # [kg/m³] density. From [B] page 1 physical table
            'Bsat': 1.56,                    # [T]     saturation. From [B] page 1 electromagnetic table
            'B_max': 0.7 * 1.56,             # [T]     70% operating limit
            'mu_r': 3000,                    # [-]     relative permeability at 10 kHz. From [A] Fig 12a at 10kHz # Assumed constant for simplicity

            # ── Core geometry (scaled 4216L1R-B × 1.55) ──────────────────────────
            'A_core': 180e-3  *  0.705,  # [m]  Overall width;         outer horizontal dimension;                    from [A] Table 1
            'B_core': 240e-3 *  0.705,   # [m]  Overall height;        outer vertical dimension;                      from [A] Table 1
            'D_core': 30e-3  *  0.705,   # [m]  Depth (cast width);    dimension going into the page;                 from [A] Table 1
            'E_core': 50e-3  *  0.705,   # [m]  Thickness (build);                                                    from [A] Table 1
            'F_core': 80e-3  *  0.705,   # [m]  Window width;          inner horizontal opening for winding;          from [A] Table 1
            'G_core': 140e-3  *  0.705,  # [m]  Window height;         inner vertical opening for winding;            from [A] Table 1
            'kf'    : 0.82,             # [-]  Stacking factor;                                                      from [A] Table 2

            # ── Winding (Elektrisola Amidester 200 A200 — IEC 60317-8 / NEMA MW 74) ──────
            # Source: Elektrisola Enamelled Copper Wire — Manufacturing Programme and Technical Data, Page 3 (product table) and Page 4 (dimensional table) https://www.mintex.si/wp-content/uploads/2018/06/CuL-englisch.pdf
            # Note: Rac ≈ Rdc assumed — conductor type not specified; skin and proximity effects neglected

            'J_max': 4e6,            # [A/m²]  Maximum current density = 4 A/mm²; standard design practice for naturally cooled inductors
            'd_strand': 0.500e-3,    # [m]     Strand diameter — largest available; Source: Elektrisola datasheet page 4, nominal diameter column
            'A_strand': 0.196350e-6, # [m²]    Bare copper area of one strand;     Source: Elektrisola datasheet page 4, section column
            'rho': 1.709e-8,         # [ohm·m]   Copper resistivity at 20°C;         standard material constant

            # ── Lifetime (Elektrisola Amidester 200 A200) ──────────────────────
            # Source: Elektrisola Enamelled Copper Wire — Manufacturing Programme and Technical Data, Page 3, product table, thermal values row https://www.mintex.si/wp-content/uploads/2018/06/CuL-englisch.pdf
            # Model:  Arrhenius thermal aging — IEC 60216 Montanari, G.C., IEEE Electrical Insulation Magazine, Vol. 9, No. 5, 1993.

            'T_insulation_rated': 273 + 210,  # [K]   Temperature index of Amidester 200 — continuous operating temperature at which insulation reaches reference lifetime of 20,000 h
            'L_insulation_rated': 20000,      # [h]   Reference lifetime at T_ins_rated. Source: Elektrisola datasheet page 3 + IEC 60172 reference point
            'Ea_insulation': 1.1 * 1.602e-19, # [J]   Activation energy = 1.1 eV converted to Joules Source: Emery, F.T., "Arrhenius model for insulation aging", IEEE Electrical Insulation Magazine — standard value for Class 200 polyesterimide; NOT from Elektrisola datasheet
            'kb': 1.381e-23,                  # [J/K] Boltzmann constant — physical constant (SI units, matches Ea in Joules)
            'V_bd': 2400,                     # [V]   Minimum breakdown voltage — Grade 1, cylinder test, 0.500mm wire. Source: Elektrisola datasheet page 4, minimum breakdown voltage column

            'mu_0': 4 * np.pi * 1e-7,  # [H/m]  Permeability of free space (physical constant)

            # Inductor series resistance
            'R2': 0.05,  # [ohm]

            'L_max_years' : 30 # [Years]  Maximum lifetime possible
        }

        # ----------------------------------------#
        # LCL filter middle [Capacitor properties] and series resistance
        # ----------------------------------------#

        self.C_specs = {

            # Product code - B32362A3157J030 # FilterCap MKD AC – Single phase # TDK Electronics

            # Capacitance
            'C': 167e-6,  # [F]    Capacitance# Keep C what it is just retune the L1 and L2

            #'C': 167e-6,  # [F]    Capacitance# Keep C what it is just retune the L1 and L2

            # Dimensions
            'D_case': 75e-3,
            'H_case': 152e-3,
            'L_case': None,
            'W_case': None,

            # Ratings
            'I_C_RMS_rated': 30,              # [A]     Rated RMS current # 330
            'V_C_RMS_Rated': 330,             # [V]     Rated RMS voltage
            'V_C_Peak_Rated': 460,            # [V]     Rated peak voltage
            'Temperature_Rated': 273.15 + 70, # [K]     Rated temperature
            'Lifetime_Rated': 1e5,            # [hours] Rated lifetime

            # Constants
            'A': 8.5,    # 8.5
            'n': 9.4,    # 9.4

            # Thermal
            'T_C_Rated': 273 + 85,  # [K]    Maximum hotspot temperature
            'Thermal_resistance_C': None,
            # [K/W]  Thermal resistance — not given in datasheet; assumed from same company, same dimension (reference product: B32373F5127J030)

            # Loss model
            'Rs': 3.7e-3,  # [Ohm]  ESR — series resistance of capacitor itself
            'tan_delta_measured' : 1e-3, # [-]   Total dissipation factor of the capacitor measured at 100 Hz. page 3
            'f_measured_for_tan_delta':100,  # [Hz] Frequency at which tan_delta_measured was specified page 3
            'tan_delta_0': None,
            # [-] Dielectric loss tangent of polypropylene. Derived from datasheet page 3 and page 16:
            # tan_delta(f) = tan_delta_0 + Rs * omega * C [page 16], tan_delta <= 1.0e-3 at 100 Hz [page 3]
            # tan_delta_0 = 1e-3 - Rs * 2*pi*100 * C = 1e-3 - 1.9e-3 * 628.3 * 150e-6 = 1e-3 - 1.79e-4 = 8.21e-4
            # Source: TDK B3236X datasheet page 3 + page 16

            # Lifetime curves — L [hours] vs T [K] at each voltage stress ratio V/V_rated
            # Source: TDK Electronics datasheet for B32362A4157J080
            'lifetime_curves': {1.3: {"T": np.array([273 + 85, 273 + 75, 273 + 70, 273 + 65, 273 + 60, 273 + 55, 273 + 50]),
                                      "L": np.array([0.15 * 1e5, 0.31 * 1e5, 0.5 * 1e5, 0.8 * 1e5, 1e5, 1.6 * 1e5, 2.25 * 1e5])},
                                1.2: {"T": np.array([273 + 85, 273 + 75, 273 + 70, 273 + 65, 273 + 60, 273 + 55]),
                                      "L": np.array([0.26 * 1e5, 0.6 * 1e5, 0.85 * 1e5, 1.4 * 1e5, 1.9 * 1e5, 2.5 * 1e5])},
                                1.1: {"T": np.array([273 + 85, 273 + 75, 273 + 70, 273 + 65]),
                                      "L": np.array([0.5 * 1e5, 1.1 * 1e5, 1.75 * 1e5, 2.4 * 1e5])},
                                1.0: {"T": np.array([273 + 85, 273 + 75, 273 + 70]),
                                      "L": np.array([1e5, 2.1 * 1e5, 3 * 1e5])},
                                0.9: {"T": np.array([273 + 85, 273 + 80]),
                                      "L": np.array([2 * 1e5, 2.8 * 1e5])},
                                0.8: {"T": np.array([273 + 85, 273 + 80]),
                                      "L": np.array([4.5 * 1e5, 6 * 1e5])}, },

            'Lifetime_calculations':'Graphical', # [-] 'Graphical' or 'Analytical' # 'Graphical' option finds the lifetime graphically and 'Analytical' finds the lifetime of capacitor analytically

            # Damping resistor
            'R3': 0.0011754 # [Ohm]  Series resistance placed in parallel with capacitor for passive damping of LCL resonance (PD-3 method)
        }





