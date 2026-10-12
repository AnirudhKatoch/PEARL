"""
Input_parameters.py  (PEARL v3)
===============================

All inputs in one place. Any attribute can be overridden without editing this file:

    params = Input_parameters_class(P, Q, T_env, N_parallel=2, r_sink=0.72, Np=80)

Overrides are applied BEFORE the derived quantities (currents, angles, modulation) are computed.
"""

import warnings
import numpy as np
from Calculation_functions import Calculation_functions_class


class Input_parameters_class:

    def __init__(self, P, Q, T_env, **overrides):

        # ============================================================================== #
        # 1. Method settings (what PEARL v2 does and how finely)
        # ============================================================================== #

        self.profile_dt = 1.0            # [s] resolution of P, Q, T_env (mission profile)
        self.loss_model = "physical"     # "physical" (v2) | "legacy" (v1 formulas, for comparison)
        self.junction_model = "foster"   # "foster" (datasheet Zth_jc as defined) | "ladder" (v1)
        self.ton_rule = "semikron"       # "semikron": t_on = max(t_on, 40 ms) (AN 21-001) | "raw"
        self.ton_min = 0.04              # [s] the 40 ms floor of AN 21-001
        self.Np = 40                     # samples per fundamental period (40 -> 0.5 ms at 50 Hz)
        self.op_resolution = (0.02, 1e-3)  # bin size (dIs [A], dphi [rad]) for operating points; None = exact
        self.init = "steady"             # "steady" | "ambient" (cold start) | "periodic" (profile repeats)
        self.slow_gate = 0.05            # [K] ignore mission-profile reversals smaller than this
        self.quasi_static_tau = 0.0      # [s] thermal modes faster than this use Rth only (0 = exact)
        self.temperature_limit_action = "warn"   # "warn" | "raise"
        self.matrix_bins = (0.5, 2.0)    # rainflow-matrix bin size (dTj [K], Tjm [degC]) for plots
        self.save_per_second = False     # write a compact per-second parquet (float32) as well
        self.results_dir = "Results"

        # Electro-thermal coupling: losses grow with temperature, P(Tj) = P_ref*(1 + kappa*(Tj - T_ref))
        # kappa = 0 switches it off. Typical silicon IGBT/diode: 0.002 ... 0.004 1/K (fit it from the
        # datasheet VCEsat / VF / Eon / Eoff / Erec values at 25 degC and 150 degC).
        self.kappa_IGBT = 0.0            # [1/K]
        self.kappa_Diode = 0.0           # [1/K]
        self.kappa_T_ref = 298.15        # [K] temperature at which the loss parameters are given
        self.coupling_iterations = 4

        # Plotting
        self.Plotting_flag = True
        self.plot_day = None             # index of the day to zoom into (None = busiest day)

        # ============================================================================== #
        # 2. Power-cycle lifetime model (Semikron Danfoss AN 21-001, baseplate modules)
        # ============================================================================== #

        self.A0 = 2.9e9
        self.A1 = 60
        self.T0_K = 40
        self.lambda_K = 17
        self.alpha = -4.3
        self.Ea_J = 4.50e-20
        self.kB_J_per_K = 1.38e-23
        self.C = 1
        self.gamma = -0.75
        self.k_thickness = {"IGBT": 1, "Diode": 0.65}
        #  AN 21-001: 1.00 IGBT <= 1200 V | 0.65 1700 V IGBT and CAL / freewheeling diodes |
        #             0.50 rectifier (mains) diodes and thyristors | 0.33 SiC <= 1200 V
        #  PEARL v1/v2 used 0.5 for the anti-parallel diode; a freewheeling diode belongs to the 0.65 class.

        # Which lifetime model(s) to use (see Lifetime_models.py):
        #   "semikron2024" (default, uses the parameters above) | "lesit" | "bayerer2008" | "cma" | "curves"
        #   a dict, e.g. {"model": "bayerer2008", "I_per_wire_A": 8, "D_wire_um": 400, "V_blocking_V": 600}
        #   your digitised curves: copy Lifetime_models.USER_CURVES_EXAMPLE and fill in your numbers
        #   a list = several models at once; the first one is the main result, all are reported
        self.lifetime_model = "semikron2024"

        # Monte Carlo (uncertainty of stress and model). 1-sigma values; +-5 % at 3 sigma by default.
        self.MC_samples = 5000
        self.MC_samples_other_models = 2000
        self.MC_sigma = {"dT": 0.05 / 3, "Tjm_K": 2.0, "A": 0.05 / 3, "alpha": 0.05 / 3}
        self.MC_seed = 0

        # ============================================================================== #
        # 3. Package limits
        # ============================================================================== #

        self.max_V_CE = 600
        self.max_IGBT_current = 50
        self.max_IGBT_temperature = 448.15
        self.max_Diode_current = 30
        self.max_Diode_temperature = 448.15
        self.IGBT_max_life = 15          # [years] cap used only for the reported "capped" value
        self.Diode_max_life = 15

        # ============================================================================== #
        # 4. Grid
        # ============================================================================== #

        self.f = 50

        # ============================================================================== #
        # 5. Thermal network (Infineon IKW50N60H3, Thermal Grizzly Kryonaut, CI 3-121211U)
        # ============================================================================== #

        # Junction -> case, datasheet Foster coefficients (Fig. 21 / Fig. 22)
        self.r_I = np.array([7.0e-3, 3.736e-2, 9.205e-2, 1.2996e-1, 1.8355e-1])     # [K/W]
        self.tau_I = np.array([4.4e-5, 1.0e-4, 7.2e-4, 8.3e-3, 7.425e-2])           # [s]
        self.r_D = np.array([4.915956e-2, 2.254532e-1, 3.125229e-1, 2.677344e-1, 1.951733e-1])
        self.tau_D = np.array([7.5e-6, 2.2e-4, 2.3e-3, 1.546046e-2, 1.078904e-1])

        # Case -> sink (paste) and sink -> ambient (heat sink)
        self.r_paste = np.array([0.0032])        # [K/W]
        self.tau_paste = np.array([1e-3])        # [s]
        self.r_sink = np.array([2.0])            # [K/W] 3-121211U at 1 m/s (0.72 = 10U at 4 m/s)
        self.sink_mass = 19e-3                   # [kg]
        self.sink_specific_heat = 900.0          # [J/(kg K)] aluminium

        # ============================================================================== #
        # 6. Device loss parameters (IKW50N60H3 datasheet, 25 degC)
        # ============================================================================== #

        self.f_sw = 10e3              # [Hz]
        self.t_on = 60e-9             # [s] td(on) + tr
        self.t_off = 259e-9           # [s] td(off) + tf
        self.I_ref = 30.0             # [A] diode reverse-recovery test current
        self.V_ref = 400.0            # [V] diode reverse-recovery test voltage
        self.Q_rr = 0.88e-6           # [C]
        self.Err_D = None             # [J] if None: Q_rr * V_ref (upper bound; use datasheet E_rec if given)
        self.I_I_points = np.array([25, 50, 100]); self.V_I_points = np.array([1.425, 1.800, 2.500])
        self.I_D_points = np.array([15, 30, 60]);  self.V_D_points = np.array([1.35, 1.65, 2.1])

        # ============================================================================== #
        # 7. Inverter
        # ============================================================================== #

        self.Vs_rms = 230.0           # [V] phase RMS voltage (scalar or array)
        self.V_dc_value = 600.0       # [V] DC-link voltage (scalar or array)
        self.M = 1                    # fraction of the maximum linear voltage (used for the Vs check)
        self.inverter_phases = 3
        self.modulation_scheme = "svm"            # "svm" | "spwm"
        self.single_phase_inverter_topology = "full"
        self.N_parallel = 1           # devices in parallel per switch position

        # ---------------------------------------------------------------- overrides ------
        unknown = [k for k in overrides if not hasattr(self, k)]
        if unknown:
            raise AttributeError(f"Unknown parameter(s): {unknown}")
        for k, v in overrides.items():
            setattr(self, k, v)

        # ============================================================================== #
        # Derived quantities
        # ============================================================================== #

        self.version = "3.0"
        self.omega = 2 * np.pi * self.f
        self.T_env = np.asarray(T_env, dtype=np.float64)
        if self.T_env.size and np.nanmax(self.T_env) < 150.0:
            raise ValueError("T_env must be in kelvin (all values are below 150). Did you pass degC? Add 273.15.")
        self.P = np.asarray(P, dtype=np.float64)
        self.Q = np.asarray(Q, dtype=np.float64)
        n = self.P.size
        if self.Q.size != n or self.T_env.size != n:
            raise ValueError("P, Q and T_env must have the same length")

        self.cap_I = self.tau_I / self.r_I            # Foster capacitances (ladder option only)
        self.cap_D = self.tau_D / self.r_D
        self.cap_paste = self.tau_paste / self.r_paste
        self.r_sink = np.atleast_1d(np.asarray(self.r_sink, dtype=np.float64))
        self.cap_sink = np.atleast_1d(np.full(self.r_sink.size, self.sink_mass * self.sink_specific_heat / self.r_sink.size))
        self.tau_sink = self.r_sink * self.cap_sink

        if self.Err_D is None:
            self.Err_D = self.Q_rr * self.V_ref
        self.R_IGBT, self.V_0_IGBT = np.polyfit(self.I_I_points, self.V_I_points, 1)
        self.R_D, self.V_0_D = np.polyfit(self.I_D_points, self.V_D_points, 1)

        if self.inverter_phases not in (1, 3):
            raise ValueError("inverter_phases must be 1 or 3")
        if self.modulation_scheme not in ("spwm", "svm"):
            raise ValueError("modulation_scheme must be 'spwm' or 'svm'")
        if self.single_phase_inverter_topology not in ("half", "full"):
            raise ValueError("single_phase_inverter_topology must be 'half' or 'full'")
        self.zero_sequence = (self.inverter_phases == 3 and self.modulation_scheme == "svm")

        self.Vs = np.broadcast_to(np.asarray(self.Vs_rms, dtype=np.float64), (n,)).copy()
        self.V_dc = np.broadcast_to(np.asarray(self.V_dc_value, dtype=np.float64), (n,)).copy()

        self.Vs, self.Is, self.phi, self.V_dc, self.pf, self.M, self.S = Calculation_functions_class.compute_power_flow(
            P=self.P, Q=self.Q, V_dc=self.V_dc, Vs=self.Vs, M=self.M,
            single_phase_inverter_topology=self.single_phase_inverter_topology,
            inverter_phases=self.inverter_phases, modulation_scheme=self.modulation_scheme,
            N_parallel=self.N_parallel)

        # Four-quadrant angle (v3): phi = angle by which the phase voltage leads the phase current,
        # generator convention: P = Vs*Is*cos(phi), Q = Vs*Is*sin(phi). Q > 0 = reactive power delivered.
        # For P >= 0 this equals the v1 angle; for P < 0 (battery charging, rectifier operation) v1 lost the
        # sign of P, so the IGBT carried the current that the diode really carries.
        self.phi = np.arctan2(self.Q / self.N_parallel, self.P / self.N_parallel)
        self.Is = np.abs(self.Is)          # v1 returned a negative RMS current for P < 0
        self.phi[(self.P == 0) & (self.Q == 0)] = 0.0

        # modulation amplitude actually needed for Vs at V_dc (drives the duty cycle)
        self.ma = Calculation_functions_class.modulation_amplitude(self.Vs, self.V_dc, self.inverter_phases,
                                                                   self.single_phase_inverter_topology)
        Calculation_functions_class.check_modulation(self.ma, self.inverter_phases, self.modulation_scheme)

        Calculation_functions_class.check_max_package_current_limit(Is=self.Is, M=self.M,
                                                                    max_IGBT_current=self.max_IGBT_current,
                                                                    max_diode_current=self.max_Diode_current)
        Calculation_functions_class.check_vce(self.V_dc, self.max_V_CE)

        peak = np.sqrt(2) * np.max(self.Is) if n else 0.0
        if peak > 2 * self.max_IGBT_current:
            warnings.warn(f"Peak device current {peak:.0f} A is above twice the nominal current "
                          f"({self.max_IGBT_current} A). Consider N_parallel > 1.")

        self.n_switches = 6 if self.inverter_phases == 3 else (4 if self.single_phase_inverter_topology == "full" else 2)
        self.n_devices = self.n_switches * self.N_parallel     # IGBT+diode pairs in the inverter

        self.lifetime_parameters = dict(A0=self.A0, A1=self.A1, T0_K=self.T0_K, lambda_K=self.lambda_K,
                                        alpha=self.alpha, Ea_J=self.Ea_J, kB_J_per_K=self.kB_J_per_K,
                                        C=self.C, gamma=self.gamma)
