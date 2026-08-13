import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import cont2discrete
from scipy.interpolate import interp1d
from pathlib import Path



class Calculation_functions_class:

    @staticmethod
    def validate_pwm_pulse_amplitude(Vdc_rated, inverter_phases, single_phase_inverter_topology,waveform_voltage_definition, Vo):

        """
        Validate the instantaneous PWM pulse amplitude Vo from Vdc_rated depending on the inverter configuration.

        Parameters
        ----------
        Vdc_rated : float
            Rated DC bus voltage [V]
        inverter_phases : {1, 3}
            Number of inverter phases.
        single_phase_inverter_topology : {"half", "full"}, optional
            Required when inverter_phases == 1.
        waveform_voltage_definition : {"switched_output", "pole_voltage"}
            Meaning of the waveform being generated.

            For single-phase:
            - "switched_output":
                half-bridge  -> max |Vo| = Vdc_rated/2
                full-bridge  -> max |Vo| = Vdc_rated
            - "pole_voltage":
                max |Vo| = Vdc_rated/2

            For three-phase:
            - "pole_voltage":
                max |Vo| = Vdc_rated/2
            - "switched_output":
                "Not Supported".

        Vo : Array
            Requested pulse amplitude [V].

        """

        if Vdc_rated <= 0:
            raise ValueError("Vdc_rated must be positive.")

        if inverter_phases not in (1, 3):
            raise ValueError("inverter_phases must be 1 or 3.")

        if waveform_voltage_definition not in ("switched_output", "pole_voltage"):
            raise ValueError("waveform_voltage_definition must be 'switched_output' or 'pole_voltage'.")

        if inverter_phases == 1:
            if single_phase_inverter_topology not in ("half", "full"):
                raise ValueError("For single-phase inverter, single_phase_inverter_topology must be 'half' or 'full'.")

            if waveform_voltage_definition == "pole_voltage":
                Vo_theoretical_max = Vdc_rated / 2.0
            else:  # switched_output
                if single_phase_inverter_topology == "half":
                    Vo_theoretical_max = Vdc_rated / 2.0
                else:  # full bridge
                    Vo_theoretical_max = Vdc_rated

        else:  # inverter_phases == 3
            if waveform_voltage_definition == "pole_voltage":
                # For standard 2-level three-phase leg voltages
                Vo_theoretical_max = Vdc_rated / 2.0
            else:
                raise ValueError(f"waveform_voltage_definition='switched_output' is not supported "
                    f"for inverter_phases={inverter_phases}.Three-phase inverters currently support only 'pole_voltage' definition.")

        if np.any(Vo < 0):
            raise ValueError(f"Invalid Vo detected. All values in Vo must be non-negative. "
                             f"Minimum detected value is {np.min(Vo):.3f} V.")

        if np.any(Vo > Vo_theoretical_max):
            raise ValueError(f"Invalid Vo detected. All values in Vo must be less than or equal to "
                             f"Vo_theoretical_max={Vo_theoretical_max:.3f} V. "
                             f"Maximum detected value is {np.max(Vo):.3f} V.")

    @staticmethod
    def validate_ac_rms_voltage_limit(Vdc_RMS, M, inverter_phases, modulation_scheme, single_phase_inverter_topology, Vg_RMS):
        """
        Validate that the requested AC RMS grid-voltage profile does not exceed
        the theoretical maximum AC RMS voltage that the inverter can generate at
        each mission-profile time step.

        This function uses the DC-link voltage profile Vdc_RMS rather than a fixed
        rated DC-link voltage. Therefore, for every profile index i, it checks:

            Vg_RMS[i] <= Vg_RMS_max_theoretical[i]

        where Vg_RMS_max_theoretical[i] is calculated from Vdc_RMS[i], M[i],
        inverter topology, and modulation scheme.

        Parameters
        ----------
        Vdc_RMS : array_like
            DC-link voltage profile [V]. Each value represents the available
            DC-link voltage at one mission-profile time step.

        M : array_like
            Modulation-index profile [-]. Each value should normally be between
            0 and 1 for linear modulation.

        inverter_phases : {1, 3}
            Number of inverter phases.

        modulation_scheme : {"spwm", "svm"}
            PWM modulation scheme. Current framework supports only "spwm".

        single_phase_inverter_topology : {"half", "full"}
            Single-phase inverter topology. Required only when inverter_phases == 1.

        Vg_RMS : array_like
            Requested AC RMS voltage profile [V].

            In the current three-phase time-domain model, this should be the
            phase RMS voltage because the grid waveform is generated as:

                Vg = sqrt(2) * Vg_RMS * sin(omega*t)

        Raises
        ------
        ValueError
            Raised if any requested Vg_RMS value is negative, if the DC-link
            voltage or modulation index is invalid, or if the requested AC RMS
            voltage exceeds the theoretical inverter voltage capability.
        """

        Vdc_RMS = np.asarray(Vdc_RMS)
        M = np.asarray(M)
        Vg_RMS = np.asarray(Vg_RMS)

        if Vdc_RMS.shape != Vg_RMS.shape:
            raise ValueError(
                "Vdc_RMS and Vg_RMS must have the same shape.\n"
                f"Vdc_RMS shape = {Vdc_RMS.shape}\n"
                f"Vg_RMS shape  = {Vg_RMS.shape}"
            )

        if M.shape != Vg_RMS.shape:
            raise ValueError(
                "M and Vg_RMS must have the same shape.\n"
                f"M shape      = {M.shape}\n"
                f"Vg_RMS shape = {Vg_RMS.shape}"
            )

        if np.any(Vdc_RMS <= 0):
            violation_idx = np.where(Vdc_RMS <= 0)[0]
            raise ValueError(
                "Invalid Vdc_RMS detected. All DC-link voltage values must be positive.\n"
                f"First violation at index {violation_idx[0]}: "
                f"Vdc_RMS={Vdc_RMS[violation_idx[0]]:.3f} V"
            )

        if np.any(M < 0):
            violation_idx = np.where(M < 0)[0]
            raise ValueError(
                "Invalid modulation index detected. All M values must be non-negative.\n"
                f"First violation at index {violation_idx[0]}: "
                f"M={M[violation_idx[0]]:.3f}"
            )

        if np.any(M > 1):
            violation_idx = np.where(M > 1)[0]
            raise ValueError(
                "Invalid modulation index detected. Current framework assumes linear modulation, so M must be <= 1.\n"
                f"First violation at index {violation_idx[0]}: "
                f"M={M[violation_idx[0]]:.3f}"
            )

        if np.any(Vg_RMS < 0):
            violation_idx = np.where(Vg_RMS < 0)[0]
            raise ValueError(
                "Invalid Vg_RMS detected. All AC RMS voltage values must be non-negative.\n"
                f"First violation at index {violation_idx[0]}: "
                f"Vg_RMS={Vg_RMS[violation_idx[0]]:.3f} V"
            )

        if inverter_phases not in (1, 3):
            raise ValueError("inverter_phases must be 1 or 3.")

        if modulation_scheme not in ("spwm", "svm"):
            raise ValueError("modulation_scheme must be 'spwm' or 'svm'.")

        if modulation_scheme == "svm":
            raise ValueError("Current framework only supports 'spwm' modulation.")

        if inverter_phases == 1:

            if single_phase_inverter_topology not in ("half", "full"):
                raise ValueError(
                    "For single-phase inverter, single_phase_inverter_topology must be 'half' or 'full'."
                )

            if single_phase_inverter_topology == "full":
                Vg_RMS_max_theoretical = (M * Vdc_RMS) / np.sqrt(2.0)

            else:  # half bridge
                Vg_RMS_max_theoretical = (M * Vdc_RMS) / (2.0 * np.sqrt(2.0))

        else:  # three-phase SPWM, phase RMS limit

            Vg_RMS_max_theoretical = (M * Vdc_RMS) / (2.0 * np.sqrt(2.0))

        if np.any(Vg_RMS > Vg_RMS_max_theoretical):
            violation_idx = np.where(Vg_RMS > Vg_RMS_max_theoretical)[0]
            first_idx = violation_idx[0]

            raise ValueError(
                "Invalid Vg_RMS detected. Requested AC RMS voltage exceeds inverter capability.\n"
                "Each Vg_RMS value must be less than or equal to the theoretical maximum "
                "allowed by the corresponding Vdc_RMS and M values.\n\n"
                f"First violation at index {first_idx}:\n"
                f"Vdc_RMS[{first_idx}] = {Vdc_RMS[first_idx]:.3f} V\n"
                f"M[{first_idx}] = {M[first_idx]:.3f}\n"
                f"Vg_RMS[{first_idx}] = {Vg_RMS[first_idx]:.3f} V\n"
                f"Vg_RMS_max_theoretical[{first_idx}] = {Vg_RMS_max_theoretical[first_idx]:.3f} V"
            )

    @staticmethod
    def validate_simulation_resolution(samples_per_switching_period, Minimum_required_samples_per_switching_period):

        """
        Validate that the simulation time resolution is sufficient to accurately
        resolve PWM switching events.

        Parameters
        ----------
        samples_per_switching_period : float
            Actual number of simulation samples within one PWM switching period.

        minimum_required_samples_per_switching_period : int or float
            Minimum required number of samples per switching period.

        Raises
        ------
        ValueError
            Raised when the simulation resolution is too low for accurate PWM
            waveform representation.
        """

        if samples_per_switching_period < Minimum_required_samples_per_switching_period:
            raise ValueError(f"Insufficient PWM simulation resolution.\n"
                             f"Current samples per switching period = {samples_per_switching_period:.2f}\n"
                             f"At least {Minimum_required_samples_per_switching_period} samples per switching"
                             f" period are required to accurately resolve PWM switching events and also perfectly calculate Vs_ref value as less of error as possible."
                             f" Increase the value of resolution_per_cycle.")

    @staticmethod
    def validate_required_inverter_voltage(Vs_ref, Vo_available):

        """
        Validate that the required inverter reference voltage does not exceed
        the available inverter switching voltage capability.

        Parameters
        ----------
        Vs_ref : array_like
            Required inverter reference voltage waveform [V]

        Vo_available : float or array_like
            Available inverter pulse amplitude capability [V]

        Raises
        ------
        ValueError
            Raised when the required inverter voltage exceeds the available
            inverter capability.
        """

        Vs_peak = np.max(np.abs(Vs_ref))

        if np.any(Vs_peak > Vo_available):

            if np.isscalar(Vo_available):
                available = Vo_available
            else:
                available = np.min(Vo_available)

            raise ValueError("Not feasible: required inverter voltage exceeds capability.\n"
                             f"Required peak = {Vs_peak:.2f} V, "
                             f"Available = {available:.2f} V")

    @staticmethod
    def LCL_Filter_Grid_Connected(t, Vs, Vg, L1, L2, C, R1, R2, R3):

        """
        Solve the time-domain response of an LCL filter connected between an inverter and the grid.

        The function computes the inductor currents, capacitor voltage, and related voltages
        based on the differential equations of the LCL filter.

        Differential equations
        ----------------------
        dI_L1_dt = (Vs - (R1 * I_L1) - V_C - R3 * (I_L1 - I_L2)) / L1
        dI_L2_dt = (V_C + R3 * (I_L1 - I_L2) - V_g - R2 * I_L2) / L2
        dV_C_dt = (I_L1 - I_L2) / C

        Naming convention
        -----------------
        Vs   : Inverter voltage input
        Vg   : Known grid voltage
        I_L1 : Current through left inductor
        V_L1 : Voltage across left inductor
        I_L2 : Current through right inductor
        V_L2 : Voltage across right inductor
        V_C  : Capacitor voltage
        I_C  : Capacitor current
        R1   : Series resistance of inverter-side inductor
        R2   : Series resistance of grid-side inductor
        R3   : Series resistance of capacitor
        L1   : Inverter side inductor
        L2   : Grid side inductor
        C    : Capacitance

        Parameters
        ----------
        t : array
        Time vector [s]
        Vs : array
        Inverter output voltage [V]
        Vg : array
        Grid voltage [V]
        L1 : float
        Inverter-side inductance [H]
        L2 : float
        Grid-side inductance [H]
        C : float
        Filter capacitance [F]
        R1 : float
        Series resistance of inverter-side inductor [Ohm]
        R2 : float
        Series resistance of grid-side inductor [Ohm]
        R3 : float
        Series resistance of capacitor [Ohm]

        Returns
        -------
        V_L1 : array
        Voltage across inverter-side inductor [V]
        I_L1 : array
        Current through inverter-side inductor [A]
        V_C : array
        Capacitor voltage [V]
        I_C : array
        Capacitor current [A]
        V_L2 : array
        Voltage across grid-side inductor [V]
        I_L2 : array
        Current through grid-side inductor [A]
        """

        t = np.asarray(t)
        Vs = np.asarray(Vs)
        Vg = np.asarray(Vg)

        if L1 <= 0 or L2 <= 0 or C <= 0:
            raise ValueError("L1, L2, and C must be positive.")

        if R1 < 0 or R2 < 0 or R3 < 0:
            raise ValueError("R1, R2, and R3 must be non-negative.")

        if len(t) != len(Vs) or len(t) != len(Vg):
            raise ValueError("t, Vs, and Vg must have the same length.")

        dt_array = np.diff(t)
        if not np.allclose(dt_array, dt_array[0]):
            raise ValueError("This discrete state-space method requires a fixed time step.")

        dt = dt_array[0]
        n = len(t)

        # State vector:
        # x = [I_L1, I_L2, V_C]
        A = np.array([[-(R1 + R3) / L1, R3 / L1, -1.0 / L1],
                      [R3 / L2, -(R2 + R3) / L2, 1.0 / L2],
                      [1.0 / C, -1.0 / C, 0.0]])

        # Input vector:
        # u = [Vs, Vg]
        B = np.array([[1.0 / L1, 0.0],
                      [0.0, -1.0 / L2],
                      [0.0, 0.0]])

        # Discretize continuous-time system
        Ad, Bd, _, _, _ = cont2discrete((A, B, np.eye(3), np.zeros((3, 2))), dt, method="zoh")

        # Allocate state array
        x = np.zeros((3, n))

        # Time-domain simulation
        for k in range(n - 1):
            u_k = np.array([Vs[k], Vg[k]])
            x[:, k + 1] = Ad @ x[:, k] + Bd @ u_k

        # State variables
        I_L1 = x[0, :]
        I_L2 = x[1, :]
        V_C = x[2, :]

        # Derived quantities
        I_C = I_L1 - I_L2
        V_L1 = Vs - V_C - R3 * I_C - R1 * I_L1
        V_L2 = V_C + R3 * I_C - Vg - R2 * I_L2


        # Optional consistency checks
        kcl_ok = np.allclose(I_L1, I_C + I_L2)
        kvl_left_ok = np.allclose(Vs, R1 * I_L1 + V_L1 + V_C + R3 * I_C)
        kvl_right_ok = np.allclose(V_C + R3 * I_C, R2 * I_L2 + V_L2 + Vg)
        branch_ok = np.allclose(V_C + R3 * I_C, V_C + R3 * I_C)

        if not (kcl_ok and kvl_left_ok and kvl_right_ok and branch_ok):
            print("Warning: one or more KCL/KVL checks are not within tolerance.")
            
        return V_L1, I_L1, V_C, I_C, V_L2, I_L2

    @staticmethod
    def validate_grid_phase_voltage_matches_line_to_line_voltage(Vg_RMS, Vg_ll_RMS, tolerance_percent=1.0):
        """
        Validate that the phase RMS grid-voltage profile is consistent with the
        specified line-to-line RMS grid voltage used for LCL filter design.

        In a balanced three-phase system:

            V_phase_RMS = V_line_line_RMS / sqrt(3)

        The LCL filter design uses Vg_ll_RMS as the line-to-line RMS grid voltage.
        The time-domain simulation uses Vg_RMS as the phase RMS voltage because
        the grid voltage waveform is generated as:

            Vg = sqrt(2) * Vg_RMS * sin(omega*t)

        Therefore, this function checks that every value in Vg_RMS is close to:

            Vg_ll_RMS / sqrt(3)

        Parameters
        ----------
        Vg_RMS : array_like
            Phase RMS grid-voltage profile [V]. Each value should represent the
            phase-to-neutral RMS voltage used to generate the time-domain grid
            voltage waveform.

        Vg_ll_RMS : float
            Line-to-line RMS grid voltage [V]. This is the voltage used for LCL
            filter design.

        tolerance_percent : float, optional
            Allowed percentage mismatch between Vg_RMS and Vg_ll_RMS/sqrt(3).
            Default is 1.0 percent.

        Raises
        ------
        ValueError
            Raised when Vg_RMS is not consistent with the line-to-line voltage
            used for LCL filter design.
        """

        Vg_RMS = np.asarray(Vg_RMS)

        if Vg_ll_RMS <= 0:
            raise ValueError("Vg_ll_RMS must be positive.")

        if np.any(Vg_RMS <= 0):
            raise ValueError("Invalid Vg_RMS detected. All phase RMS grid-voltage values must be positive.")

        if tolerance_percent < 0:
            raise ValueError("tolerance_percent must be non-negative.")

        expected_Vg_phase_RMS = Vg_ll_RMS / np.sqrt(3)

        tolerance_absolute = expected_Vg_phase_RMS * (tolerance_percent / 100)

        difference = np.abs(Vg_RMS - expected_Vg_phase_RMS)

        if np.any(difference > tolerance_absolute):
            violation_idx = np.where(difference > tolerance_absolute)[0]

            first_idx = violation_idx[0]

            raise ValueError(
                "Grid voltage mismatch detected.\n"
                "The LCL filter was designed for a line-to-line RMS grid voltage, "
                "but the phase RMS voltage profile used in the time-domain simulation "
                "is not consistent with it.\n\n"
                f"Vg_ll_RMS = {Vg_ll_RMS:.3f} V\n"
                f"Expected phase RMS voltage = Vg_ll_RMS / sqrt(3) = {expected_Vg_phase_RMS:.3f} V\n"
                f"Allowed tolerance = ±{tolerance_absolute:.3f} V ({tolerance_percent:.2f}%)\n\n"
                f"First violation at index {first_idx}:\n"
                f"Vg_RMS[{first_idx}] = {Vg_RMS[first_idx]:.3f} V\n"
                f"Difference = {difference[first_idx]:.3f} V\n\n"
                "Fix:\n"
                "Set Vg_RMS as the phase RMS value corresponding to the line-to-line grid voltage:\n"
                "    Vg_RMS = np.full(Profile_size, Vg_ll_RMS / np.sqrt(3))")

    @staticmethod
    def validate_mission_profile_lengths(Profile_size, Vdc_RMS, M, Vo, Vg_RMS, S_RMS, pf):
        """
        Validate that all mission-profile arrays have the same length as Profile_size.

        Parameters
        ----------
        Profile_size : int
            Number of mission-profile operating points.

        Vdc_RMS : array_like
            DC-link voltage profile [V].

        M : array_like
            Modulation-index profile [-].

        Vo : array_like
            PWM pulse-amplitude profile [V].

        Vg_RMS : array_like
            Phase RMS grid-voltage profile [V].

        S_RMS : array_like
            Apparent-power profile [VA].

        pf : array_like
            Power-factor profile [-].

        Raises
        ------
        ValueError
            Raised if any profile does not have length Profile_size.
        """

        profiles = {
            "Vdc_RMS": Vdc_RMS,
            "M": M,
            "Vo": Vo,
            "Vg_RMS": Vg_RMS,
            "S_RMS": S_RMS,
            "pf": pf
        }

        for name, values in profiles.items():
            values = np.asarray(values)

            if len(values) != Profile_size:
                raise ValueError(
                    f"Invalid mission-profile length for {name}.\n"
                    f"Expected length = {Profile_size}\n"
                    f"Actual length   = {len(values)}"
                )

    @staticmethod
    def validate_power_factor_profile(pf):
        """
        Validate that the power-factor profile is within the physical range [-1, 1].
        """

        pf = np.asarray(pf)

        if np.any(pf < -1) or np.any(pf > 1):
            violation_idx = np.where((pf < -1) | (pf > 1))[0]
            first_idx = violation_idx[0]

            raise ValueError(
                "Invalid power factor detected. Power factor must be between -1 and 1.\n"
                f"First violation at index {first_idx}: pf = {pf[first_idx]:.3f}")

    @staticmethod
    def validate_apparent_power_limit(S_RMS, S_rated):
        """
        Validate that the apparent-power mission profile does not exceed the rated
        inverter apparent power used for LCL filter design.
        """

        S_RMS = np.asarray(S_RMS)

        if S_rated <= 0:
            raise ValueError("S_rated must be positive.")

        if np.any(S_RMS < 0):
            violation_idx = np.where(S_RMS < 0)[0]
            first_idx = violation_idx[0]

            raise ValueError(
                "Invalid S_RMS detected. Apparent power must be non-negative.\n"
                f"First violation at index {first_idx}: S_RMS = {S_RMS[first_idx]:.3f} VA"
            )

        if np.any(S_RMS > S_rated):
            violation_idx = np.where(S_RMS > S_rated)[0]
            first_idx = violation_idx[0]

            raise ValueError(
                "Apparent-power profile exceeds rated inverter power.\n"
                f"S_rated = {S_rated:.3f} VA\n"
                f"First violation at index {first_idx}: S_RMS = {S_RMS[first_idx]:.3f} VA"
            )

    @staticmethod
    def validate_pwm_pulse_amplitude_profile(Vdc_RMS, inverter_phases, single_phase_inverter_topology, waveform_voltage_definition, Vo):
        """
        Validate the PWM pulse amplitude profile against the available DC-link
        voltage profile at every mission-profile time step.
        """

        Vdc_RMS = np.asarray(Vdc_RMS)
        Vo = np.asarray(Vo)

        if Vdc_RMS.shape != Vo.shape:
            raise ValueError(
                "Vdc_RMS and Vo must have the same shape.\n"
                f"Vdc_RMS shape = {Vdc_RMS.shape}\n"
                f"Vo shape      = {Vo.shape}"
            )

        if np.any(Vdc_RMS <= 0):
            violation_idx = np.where(Vdc_RMS <= 0)[0]
            first_idx = violation_idx[0]

            raise ValueError(
                "Invalid Vdc_RMS detected. All DC-link voltage values must be positive.\n"
                f"First violation at index {first_idx}: Vdc_RMS = {Vdc_RMS[first_idx]:.3f} V"
            )

        if np.any(Vo < 0):
            violation_idx = np.where(Vo < 0)[0]
            first_idx = violation_idx[0]

            raise ValueError(
                "Invalid Vo detected. PWM pulse amplitude must be non-negative.\n"
                f"First violation at index {first_idx}: Vo = {Vo[first_idx]:.3f} V"
            )

        if inverter_phases == 3:
            if waveform_voltage_definition != "pole_voltage":
                raise ValueError(
                    "For three-phase inverters, only 'pole_voltage' is currently supported."
                )

            Vo_max = Vdc_RMS / 2.0

        elif inverter_phases == 1:
            if single_phase_inverter_topology not in ("half", "full"):
                raise ValueError(
                    "For single-phase inverter, single_phase_inverter_topology must be 'half' or 'full'."
                )

            if waveform_voltage_definition == "pole_voltage":
                Vo_max = Vdc_RMS / 2.0

            elif waveform_voltage_definition == "switched_output":
                if single_phase_inverter_topology == "half":
                    Vo_max = Vdc_RMS / 2.0
                else:
                    Vo_max = Vdc_RMS

            else:
                raise ValueError(
                    "waveform_voltage_definition must be 'switched_output' or 'pole_voltage'."
                )

        else:
            raise ValueError("inverter_phases must be 1 or 3.")

        if np.any(Vo > Vo_max):
            violation_idx = np.where(Vo > Vo_max)[0]
            first_idx = violation_idx[0]

            raise ValueError(
                "Invalid Vo detected. PWM pulse amplitude exceeds available DC-link capability.\n"
                f"First violation at index {first_idx}:\n"
                f"Vdc_RMS[{first_idx}] = {Vdc_RMS[first_idx]:.3f} V\n"
                f"Vo[{first_idx}] = {Vo[first_idx]:.3f} V\n"
                f"Vo_max[{first_idx}] = {Vo_max[first_idx]:.3f} V")

    @staticmethod
    def compute_Vs_ref_phasor(t, f, Ig_RMS, Vg_RMS, phase_shift, L1, L2, C, R1, R2, R3, Profile_size, samples_per_second):

        """
        Compute the required inverter voltage reference Vs_ref analytically
        using phasor (frequency-domain) analysis of the LCL filter.

        Since Ig_ref and Vg are pure sinusoids and the LCL filter is linear,
        the phasor solution is exact with zero numerical error.

        Parameters
        ----------
        t : array
            Time vector [s]
        f : float
            Fundamental frequency [Hz]
        Ig_RMS : array
            RMS grid current profile, one value per second [A]
        Vg_RMS : array
            RMS grid voltage profile, one value per second [V]
        phase_shift : array
            Instantaneous phase shift waveform, same length as t [rad]
        L1 : float
            Inverter-side inductance [H]
        L2 : float
            Grid-side inductance [H]
        C : float
            Filter capacitance [F]
        R1 : float
            Series resistance of inverter-side inductor [Ohm]
        R2 : float
            Series resistance of grid-side inductor [Ohm]
        R3 : float
            Series resistance of capacitor [Ohm]
        Profile_size : int
            Number of seconds in the mission profile
        samples_per_second : int
            Number of simulation samples per second

        Returns
        -------
        Vs_ref : array
            Required inverter phase voltage waveform [V]
        """

        j = 1j
        omega = 2 * np.pi * f

        # Pre-compute impedances (constant for all profile segments)
        Z_L1 = R1 + j * omega * L1
        Z_L2 = R2 + j * omega * L2
        Z_cap = R3 + 1.0 / (j * omega * C)

        Vs_ref = np.zeros(len(t))

        for k in range(Profile_size):
            idx = slice(k * samples_per_second, (k + 1) * samples_per_second)
            t_k = t[idx]

            # Current phasor: Ig_ref = A*sin(ωt + φ) → phasor = A*e^{jφ}
            A_k = np.sqrt(2) * Ig_RMS[k]
            phi_k = phase_shift[k * samples_per_second]
            I_L2_ph = A_k * np.exp(1j * phi_k)

            # Grid voltage phasor: Vg = Vg_peak*sin(ωt) → reference angle = 0
            Vg_ph = np.sqrt(2) * Vg_RMS[k] * np.exp(1j * 0)

            # Step 1: V_node from right side (KVL)
            V_node_ph = Vg_ph + Z_L2 * I_L2_ph

            # Step 2: capacitor branch
            I_C_ph = V_node_ph / Z_cap

            # Step 3: inverter side (KCL + KVL)
            I_L1_ph = I_L2_ph + I_C_ph
            Vs_ph = V_node_ph + Z_L1 * I_L1_ph

            # Reconstruct time-domain waveform for this segment
            Vs_ref[idx] = np.abs(Vs_ph) * np.sin(omega * t_k + np.angle(Vs_ph))

        return Vs_ref

    @staticmethod
    def compute_THD(t, Signal, Signal_ref, f, dt, resolution_per_cycle, save_path, printing, n_cycles=1,max_harmonic=None,plot=True, ):
        """
        Compute the THD of the grid-side current Signal over the last n_cycles
        fundamental periods, and report tracking metrics vs the reference Signal_ref.

        The analysis window is selected by integer sample count (not by a
        float-time mask), so every harmonic lands exactly on an FFT bin and
        single-bin extraction is leakage-free.

        Parameters
        ----------
        t : array                  Time vector [s]
        Signal : array               Simulated grid-side current [A]
        Signal_ref : array             Reference grid current [A]
        f : float                  Fundamental frequency [Hz]
        dt : float                 Simulation time step [s]
        resolution_per_cycle : int Samples per fundamental cycle [-]
        save_path : str            Path for the saved comparison figure
        n_cycles : int, optional   Number of trailing cycles to analyse. Default 1.
        max_harmonic : int, optional
            Highest harmonic order included in THD. Default None = up to Nyquist.
        plot : bool, optional      Save the waveform/error plot. Default True.

        Returns
        -------
        THD_percent : float        THD of Signal [%]
        """

        Signal = np.asarray(Signal)
        Signal_ref = np.asarray(Signal_ref)
        t = np.asarray(t)

        # --- exact-integer-cycle window (last n_cycles periods) ---
        spc = int(round(resolution_per_cycle))  # samples per cycle
        win = n_cycles * spc
        if win > len(Signal):
            raise ValueError(
                f"Window of {win} samples exceeds signal length {len(Signal)}.")

        last = slice(-win, None)
        Signal_w = Signal[last]
        Signal_ref_w = Signal_ref[last]
        t_w = t[last]

        # ── 1. RMS ────────────────────────────────────────────────
        Signal_RMS = np.sqrt(np.mean(Signal_w ** 2))
        Signal_ref_RMS = np.sqrt(np.mean(Signal_ref_w ** 2))

        # ── 2. Tracking error ─────────────────────────────────────
        error = Signal_w - Signal_ref_w
        error_RMS = np.sqrt(np.mean(error ** 2))
        error_peak = np.max(np.abs(error))
        NRMSE = error_RMS / Signal_ref_RMS * 100

        # ── 3. FFT (DC removed) ───────────────────────────────────
        N = len(Signal_w)
        freqs = np.fft.rfftfreq(N, d=dt)

        fft_L2 = np.fft.rfft(Signal_w - np.mean(Signal_w))
        fft_ref = np.fft.rfft(Signal_ref_w - np.mean(Signal_ref_w))

        # fundamental sits at bin = n_cycles (n_cycles periods in the window)
        idx_f = n_cycles

        # RMS amplitude of fundamental: (2|X|/N)/sqrt(2) = sqrt(2)|X|/N
        amp_L2 = np.sqrt(2) * np.abs(fft_L2[idx_f]) / N
        amp_ref = np.sqrt(2) * np.abs(fft_ref[idx_f]) / N

        phase_L2 = np.angle(fft_L2[idx_f], deg=True)
        phase_ref = np.angle(fft_ref[idx_f], deg=True)
        phase_err = (phase_L2 - phase_ref + 180) % 360 - 180  # wrap to ±180

        # ── 4. THD of Signal (harmonics 2..max up to Nyquist) ───────
        # harmonic h sits exactly on bin h*n_cycles
        nyq_order = (len(fft_L2) - 1) // n_cycles  # highest resolvable order
        top = nyq_order if max_harmonic is None else min(max_harmonic, nyq_order)
        h_orders = np.arange(2, top + 1)
        h_bins = h_orders * n_cycles

        P_harmonics = np.sum(np.abs(fft_L2[h_bins]) ** 2)
        P_fundamental = np.abs(fft_L2[idx_f]) ** 2
        THD = np.sqrt(P_harmonics / P_fundamental)
        THD_percent = THD * 100

        # ── Print summary (unchanged format) ──────────────────────
        if printing == True:
            print("=" * 46)
            print(f"  Signal_ref RMS          : {Signal_ref_RMS:>10.4f}  A")
            print(f"  Signal   RMS          : {Signal_RMS:>10.4f}  A")
            print("-" * 46)
            print(f"  Tracking error RMS  : {error_RMS:>10.4f}  A")
            print(f"  Tracking error peak : {error_peak:>10.4f}  A")
            print(f"  NRMSE               : {NRMSE:>10.4f}  %")
            print("-" * 46)
            print(f"  Fundamental amp ref : {amp_ref:>10.4f}  A (RMS)")
            print(f"  Fundamental amp L2  : {amp_L2:>10.4f}  A (RMS)")
            print(f"  Phase ref           : {phase_ref:>10.4f}  deg")
            print(f"  Phase Signal          : {phase_L2:>10.4f}  deg")
            print(f"  Phase error         : {phase_err:>10.4f}  deg")
            print("-" * 46)
            print(f"  THD of Signal         : {THD_percent:>10.4f}  %")
            print("=" * 46)

        # ── Plot ──────────────────────────────────────────────────
        if plot:
            fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            axes[0].plot(t_w, Signal_ref_w, label='Signal_ref', linewidth=1.5)
            axes[0].plot(t_w, Signal_w, label='Signal', linewidth=1.0, linestyle='--')
            axes[0].set_ylabel("Current [A]")
            axes[0].legend()
            axes[0].set_title("Waveform comparison")

            axes[1].plot(t_w, error, color='red', linewidth=1.0,
                         label='Error (Signal − Signal_ref)')
            axes[1].axhline(0, color='k', linewidth=0.5)
            axes[1].set_ylabel("Error [A]")
            axes[1].set_xlabel("Time [s]")
            axes[1].legend()
            axes[1].set_title(
                f"Tracking error  |  NRMSE = {NRMSE:.3f}%  |  THD = {THD_percent:.3f}%")

            plt.xlim(t_w[0], t_w[-1])
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

        return THD_percent

    @staticmethod
    def compute_I_C_RMS_per_harmonic_for_capacitor(I_C, f, fsw, resolution_per_cycle, Profile_size):

        """
        Decompose the capacitor current I_C into RMS values per harmonic order
        for each second of the mission profile.

        Harmonic orders are fixed to the physically significant harmonics of a
        three-phase grid-connected inverter:
        - Low-order characteristic harmonics: 1, 5, 7, 11, 13, 17, 19
        - Switching frequency harmonic:       200 (= 10 kHz at 50 Hz fundamental)
        - Second switching harmonic:          400 (= 20 kHz at 50 Hz fundamental)

        Parameters
        ----------
        I_C : np.ndarray
            Time-domain capacitor current signal, length = Profile_size * resolution_per_cycle * f
        f : float
            Fundamental frequency [Hz]
        fsw : float
            Inverter switching frequency [Hz]
        resolution_per_cycle : int
            Number of samples per fundamental cycle
        Profile_size : int
            Number of seconds in the mission profile

        Returns
        -------
        I_C_RMS_harmonics : np.ndarray
            Shape: (Profile_size, 9)
            Columns correspond to harmonic orders [1, 5, 7, 11, 13, 17, 19, 200, 400]
            I_C_RMS_harmonics[i, n] = RMS current at harmonic order n during second i
        """
        # ----------------------------------------#
        # Compute harmonic orders dynamically
        # ----------------------------------------#
        mf = int(round(fsw / f))  # Frequency modulation ratio e.g. 200 for 50Hz, 167 for 60Hz

        low_order_harmonics = [1, 5, 7, 11, 13, 17, 19]
        switching_band_1 = list(range(mf - 2, mf + 3))  # e.g. [198,199,200,201,202] for 50Hz
        switching_band_2 = list(range(2 * mf - 2, 2 * mf + 3))  # e.g. [398,399,400,401,402] for 50Hz

        harmonic_orders = np.array(low_order_harmonics + switching_band_1 + switching_band_2)


        samples_per_second = int(resolution_per_cycle * f)  # Number of samples in 1 second
        N = samples_per_second

        # Reshape I_C from (Profile_size * N,) → (Profile_size, N)
        # Each row is one second of data — removes outer sec loop
        I_matrix = I_C.reshape(Profile_size, N)

        # FFT for all seconds simultaneously along time axis
        # Shape: (Profile_size, N//2 + 1)
        fft_vals = np.fft.rfft(I_matrix, axis=1)

        # Frequency axis — computed once for all seconds
        fft_freq = np.fft.rfftfreq(N, d=1.0 / samples_per_second)  # Shape: (N//2 + 1,)

        # Target frequencies for all harmonic orders simultaneously
        target_freqs = harmonic_orders * f  # Shape: (9,)

        # Find closest FFT bin for each harmonic order — removes inner harmonic loop
        # Broadcasting: (N//2+1, 1) vs (1, 9) → argmin along frequency axis → (9,)
        bin_indices = np.argmin(np.abs(fft_freq[:, np.newaxis] - target_freqs[np.newaxis, :]), axis=0)

        # Extract complex FFT values at harmonic bins for all seconds at once
        # fft_vals[:, bin_indices] → Shape: (Profile_size, 9)
        amplitudes_peak = (2 * np.abs(fft_vals[:, bin_indices])) / N  # Peak amplitude

        # Convert peak amplitude to RMS
        I_C_RMS_harmonics = amplitudes_peak / np.sqrt(2)  # Shape: (Profile_size, 9)

        return I_C_RMS_harmonics

    @staticmethod
    def Capacitor_total_power_losses(f, I_C_RMS_harmonics, harmonic_orders, tan_delta_0, C, Rs):

        '''
        Compute the total power dissipation in the LCL filter capacitor,
        summed over every harmonic present in the supplied spectrum.

        Parameters
        ----------
        f : float
            Fundamental grid frequency [Hz].
        I_C_RMS_harmonics : ndarray, shape (Profile_size, num_harmonics)
            RMS current through the capacitor at each harmonic order, for each
            second of the mission profile. Rows = mission-profile seconds,
            columns = harmonic orders.
        harmonic_orders : ndarray, shape (num_harmonics,)
            Harmonic order corresponding to each column of I_C_RMS_harmonics.
        tan_delta_0 : float
            Dielectric dissipation factor of the polypropylene film [-].
        C : float
            Capacitance of the filter capacitor [F].
        Rs : float
            Series resistance of the capacitor [Ω].

        Returns
        -------
        P_total_C : ndarray, shape (Profile_size,)
            Total power dissipation in the capacitor [W] for each second of
            the mission profile.
        '''

        harmonic_orders = np.asarray(harmonic_orders, dtype=float)
        f_i = harmonic_orders * f  # absolute frequency of each harmonic [Hz]

        # Dielectric losses at each harmonic: P_D(f_i) = I(f_i)^2 * tan_delta_0 / (2*pi*f_i*C)
        P_D = I_C_RMS_harmonics ** 2 * tan_delta_0 / (2 * np.pi * f_i * C)

        # Resistive losses at each harmonic: P_R(f_i) = I(f_i)^2 * Rs
        P_R = I_C_RMS_harmonics ** 2 * Rs

        P_total_per_harmonic = P_D + P_R
        P_total_C = np.sum(P_total_per_harmonic, axis=1)

        return P_total_C

    @staticmethod
    def Capacitor_hotspot_temperature(T_amb, P_total_C, Thermal_resistance_C):
        """
        Compute the capacitor hotspot temperature from ambient temperature and power dissipation.

        Parameters
        ----------
        T_amb : ndarray, shape (Profile_size,)
            Ambient temperature surrounding the capacitor for each mission-profile second [K].
        P_total_C : ndarray, shape (Profile_size,)
            Total power dissipation in the capacitor for each mission-profile second [W].
        Thermal_resistance_C : float
            Thermal resistance from ambient to hotspot of the capacitor [K/W].

        Returns
        -------
        T_C : ndarray, shape (Profile_size,)
            Capacitor hotspot temperature for each mission-profile second [K].
        """

        T_C = T_amb + P_total_C * Thermal_resistance_C  # Hotspot temperature: T_HS = T_amb + P * Rth [°C]

        return T_C

    @staticmethod
    def Singal_RMS(Signal, resolution_per_cycle, f):
        """
        Compute the RMS voltage across the LCL filter capacitor for each second of the mission profile.

        Parameters
        ----------
        Signal : ndarray, shape (Profile_size * samples_per_second,)
            Instantaneous signal waveform over the full mission profile.
        resolution_per_cycle : int
            Number of simulation samples per fundamental AC cycle [-].
        f : float
            Fundamental grid frequency [Hz].

        Returns
        -------
        Signal_RMS : ndarray, shape (Profile_size,)
            RMS voltage across the capacitor for each mission-profile second [V].

        """

        samples_per_second = resolution_per_cycle * f  # Number of simulation samples in one second
        samples_per_second = int(samples_per_second)

        Profile_size = len(Signal) // samples_per_second  # Recover the number of mission-profile seconds

        Signal_reshaped = Signal.reshape(Profile_size,
                                         samples_per_second)  # Reshape into (Profile_size, samples_per_second) so each row = one second of data

        Signal_RMS = np.sqrt(
            np.mean(Signal_reshaped ** 2, axis=1))  # Compute RMS over each row (each mission-profile second)

        return Signal_RMS

    @staticmethod
    def validate_capacitor_operating_limits(T_C, V_C_RMS, V_C, T_C_Rated, V_C_RMS_Rated, V_C_Peak_Rated, V_RMS_overvoltage_factor=1.0, V_peak_overvoltage_factor=1.0):
        """
        Validate that the LCL filter capacitor operates within its rated limits at all times.

        Temperature is a hard limit — it cannot be extended under any circumstances.
        Voltage limits can be extended using overvoltage factors, which multiply the
        rated voltage to allow operation above the nominal rating when justified
        (e.g. short-duration transients, accepted derating trade-off).

        Physical basis for voltage overvoltage factors
        -----------------------------------------------
        TDK B3236X datasheet page 14 defines:
            - Max recurrent peak voltage û: permissible for max 1% of the period
            - Non-recurrent surge voltage Vs: allowed for limited occurrences
        This means the capacitor CAN tolerate voltages above V_RMS_Rated and
        V_Peak_Rated under controlled conditions. The overvoltage factors encode
        this allowance explicitly so it is visible and auditable in the simulation.

        Parameters
        ----------
        T_C : ndarray, shape (Profile_size,)
            Capacitor hotspot temperature for each mission-profile second [K].
        V_C_RMS : ndarray, shape (Profile_size,)
            RMS voltage across the capacitor for each mission-profile second [V].
        V_C : ndarray, shape (Profile_size * samples_per_second,)
            Instantaneous capacitor voltage over the full mission profile [V].
        T_C_Rated : float
            Maximum allowable capacitor hotspot temperature [K].
            Hard limit — cannot be extended by any factor.
            Source: TDK B3236X datasheet page 3, T_hs = 85°C.
        V_C_RMS_Rated : float
            Nominal rated RMS voltage of the capacitor [V].
            Source: TDK B3236X datasheet page 10.
        V_C_Peak_Rated : float
            Nominal rated peak voltage of the capacitor [V].
            Source: TDK B3236X datasheet page 10.
        V_RMS_overvoltage_factor : float, optional
            Multiplicative factor applied to V_C_RMS_Rated to define the
            actual RMS voltage limit used in the check [−].
            Default = 1.0 (no extension beyond rated value).
            Example: 1.1 means 10% overvoltage above rated RMS is accepted.
            Effective limit = V_C_RMS_Rated * V_RMS_overvoltage_factor.
        V_peak_overvoltage_factor : float, optional
            Multiplicative factor applied to V_C_Peak_Rated to define the
            actual peak voltage limit used in the check [−].
            Default = 1.0 (no extension beyond rated value).
            Example: 1.1 means 10% overvoltage above rated peak is accepted.
            Effective limit = V_C_Peak_Rated * V_peak_overvoltage_factor.

        Returns
        -------
        None
            Returns nothing if all conditions are satisfied.

        Raises
        ------
        ValueError
            If any operating limit is exceeded, with a detailed message identifying
            the violated condition, the effective limit used, the worst-case value,
            and the mission-profile second or sample index where the violation occurs.
        """

        # ----------------------------------------#
        # Compute effective voltage limits
        # ----------------------------------------#

        V_C_RMS_limit = V_C_RMS_Rated * V_RMS_overvoltage_factor  # [V] effective RMS limit
        V_C_Peak_limit = V_C_Peak_Rated * V_peak_overvoltage_factor  # [V] effective peak limit

        errors = []

        # ----------------------------------------#
        # Check 1: Hotspot temperature limit
        # HARD LIMIT — no factor applied
        # ----------------------------------------#

        if np.any(T_C > T_C_Rated):
            violated_seconds = np.where(T_C > T_C_Rated)[0] + 1
            worst_index = np.argmax(T_C)
            worst_value = T_C[worst_index]
            errors.append(
                f"\nCondition 1 FAILED: Capacitor hotspot temperature exceeds rated limit.\n"
                f"  Hard limit (cannot be extended) : {T_C_Rated} K ({T_C_Rated - 273:.0f} °C)\n"
                f"  Worst value                     : {worst_value:.2f} K ({worst_value - 273:.2f} °C) "
                f"at mission-profile second {worst_index + 1}\n"
                f"  Exceeded at seconds             : {violated_seconds.tolist()}\n")

        # ----------------------------------------#
        # Check 2: RMS voltage limit
        # SOFT LIMIT — extended by V_RMS_overvoltage_factor
        # ----------------------------------------#

        if np.any(V_C_RMS > V_C_RMS_limit):
            violated_seconds = np.where(V_C_RMS > V_C_RMS_limit)[0] + 1
            worst_index = np.argmax(V_C_RMS)
            worst_value = V_C_RMS[worst_index]
            errors.append(
                f"\nCondition 2 FAILED: Capacitor RMS voltage exceeds effective limit.\n"
                f"  Rated limit                     : {V_C_RMS_Rated} V\n"
                f"  Overvoltage factor applied      : {V_RMS_overvoltage_factor}\n"
                f"  Effective limit                 : {V_C_RMS_limit:.2f} V\n"
                f"  Worst value                     : {worst_value:.2f} V "
                f"at mission-profile second {worst_index + 1}\n"
                f"  Exceeded at seconds             : {violated_seconds.tolist()}\n")

        # ----------------------------------------#
        # Check 3: Instantaneous peak voltage limit
        # SOFT LIMIT — extended by V_peak_overvoltage_factor
        # ----------------------------------------#

        if np.any(np.abs(V_C) > V_C_Peak_limit):
            violated_samples = np.where(np.abs(V_C) > V_C_Peak_limit)[0]
            worst_index = np.argmax(np.abs(V_C))
            worst_value = V_C[worst_index]
            errors.append(
                f"\nCondition 3 FAILED: Capacitor instantaneous voltage exceeds effective peak limit.\n"
                f"  Rated limit                     : ±{V_C_Peak_Rated} V\n"
                f"  Overvoltage factor applied      : {V_peak_overvoltage_factor}\n"
                f"  Effective limit                 : ±{V_C_Peak_limit:.2f} V\n"
                f"  Worst value                     : {worst_value:.2f} V at sample index {worst_index}\n"
                f"  Number of violated samples      : {len(violated_samples)}\n")

        # ----------------------------------------#
        # Raise all errors together if any occurred
        # ----------------------------------------#

        if errors:
            raise ValueError(
                "\n" + "=" * 60 +
                "\nCAPACITOR OPERATING LIMIT VIOLATION DETECTED" +
                "\n" + "=" * 60 +
                "".join(errors))

    @staticmethod
    def Capacitor_lifetime_graphical(T_C, V_C_RMS, V_C_RMS_Rated, lifetime_curves):
        """
        Parameters
        ----------
        T_C : ndarray, shape (Profile_size,)
            Capacitor hotspot temperature for each mission-profile second [K].
        V_C_RMS : ndarray, shape (Profile_size,)
            RMS voltage across the capacitor for each mission-profile second [V].
        V_C_RMS_Rated : float
            Rated RMS voltage of the capacitor [V].
        lifetime_curves : dict
            Dictionary of digitized lifetime curves read from the manufacturer graph. Each key is a voltage ratio
            (V_C_RMS / V_C_RMS_Rated) [-]. Each value is a sub-dictionary with:
                "T" : ndarray
                    Temperature breakpoints for this curve [K]. Can be in any order.
                "L" : ndarray
                    Lifetime values [hours] at each temperature breakpoint.
        Returns
        -------
        L : ndarray, shape (Profile_size,)
            Expected capacitor lifetime [years] at each mission-profile operating point.
        """

        V_ratio = V_C_RMS / V_C_RMS_Rated  # Voltage stress ratio for each mission-profile second [-]
        V_ratio_keys = np.array(
            sorted(lifetime_curves.keys()))  # Sorted voltage ratio values available from the digitized curves

        # ----------------------------------------#
        # Step 1: For each voltage curve, interpolate
        # lifetime in temperature at every mission-
        # profile second
        # ----------------------------------------#

        log_L_per_curve = np.zeros((len(T_C), len(V_ratio_keys)))

        for j, vr in enumerate(V_ratio_keys):
            curve = lifetime_curves[vr]
            T_curve_K = curve["T"]
            L_curve = curve["L"]

            sort_idx = np.argsort(
                T_curve_K)  # Sort indices so temperature is strictly increasing, regardless of user input order
            T_curve_K = T_curve_K[sort_idx]  # Reorder T to ascending
            L_curve = L_curve[sort_idx]  # Reorder L to match sorted T

            log_L_curve = np.log(L_curve)  # Work in log space to respect the logarithmic y-axis of the lifetime graph

            interp_func = interp1d(T_curve_K, log_L_curve,
                                   kind='linear',
                                   bounds_error=False,
                                   fill_value='extrapolate')  # Extrapolate linearly in log space for temperatures outside the digitized range

            log_L_per_curve[:, j] = interp_func(
                T_C)  # Interpolated log-lifetime at each mission-profile second for this voltage curve

        # ----------------------------------------#
        # Step 2: For each mission-profile second,
        # interpolate across voltage ratio curves
        # ----------------------------------------#

        L = np.zeros(len(T_C))

        for i in range(len(T_C)):
            interp_v = interp1d(V_ratio_keys, log_L_per_curve[i, :],
                                kind='linear',
                                bounds_error=False,
                                fill_value='extrapolate')  # Extrapolate linearly in log space for voltage ratios outside the digitized range

            L[i] = np.exp(interp_v(V_ratio[i]))  # Interpolate in voltage ratio and convert back from log space to hours

        L = L / (365 * 24)  # Convert from hours to years

        return L

    @staticmethod
    def plot_LCL_signals(t, V_L1, I_L1, V_C, I_C, V_L2, I_L2, resolution_per_cycle, save_path):
        """
        Plot all LCL filter voltages and currents for the last fundamental cycle.

        Parameters
        ----------
        t : array
            Time vector [s]
        V_L1 : array
            Voltage across inverter-side inductor [V]
        I_L1 : array
            Current through inverter-side inductor [A]
        V_C : array
            Capacitor voltage [V]
        I_C : array
            Capacitor current [A]
        V_L2 : array
            Voltage across grid-side inductor [V]
        I_L2 : array
            Current through grid-side inductor [A]
        resolution_per_cycle : int
            Number of samples per fundamental cycle
        save_path : str, optional
            File path for the saved figure
        """

        last = slice(-resolution_per_cycle, None)
        t_w = t[last]

        signals = [
            ("V_L1", "Voltage [V]", V_L1[last], "tab:blue"),
            ("I_L1", "Current [A]", I_L1[last], "tab:orange"),
            ("V_C", "Voltage [V]", V_C[last], "tab:green"),
            ("I_C", "Current [A]", I_C[last], "tab:red"),
            ("V_L2", "Voltage [V]", V_L2[last], "tab:purple"),
            ("I_L2", "Current [A]", I_L2[last], "tab:brown"),
        ]

        fig, axes = plt.subplots(6, 1, figsize=(10, 14), sharex=True)

        for ax, (name, ylabel, data, color) in zip(axes, signals):
            rms = np.sqrt(np.mean(data ** 2))
            peak = np.max(np.abs(data))
            ax.plot(t_w, data, color=color, linewidth=1.2)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{name}   |   RMS = {rms:.3f}   |   Peak = {peak:.3f}")
            ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time [s]")
        plt.xlim(t_w[0], t_w[-1])
        plt.suptitle("LCL Filter Signals — Last Cycle", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    @staticmethod
    def calculate_core_surface_area(A_core, B_core, D_core, F_core, G_core):
        """
        Calculate the total exposed surface area of a rectangular tape-wound core.

        The core geometry is a rectangular frame (picture-frame cross-section)
        with depth D. Three surface groups contribute:

            Outer perimeter surfaces : 2(A + B) × D
            Inner window surfaces    : 2(F + G) × D
            Two end faces            : 2(A×B - F×G)

        Parameters
        ----------
        A_core : float  Overall width          [m]  outer horizontal dimension
        B_core : float  Overall height         [m]  outer vertical dimension
        D_core : float  Depth (cast width)     [m]  dimension into the page
        F_core : float  Window width           [m]  inner horizontal opening
        G_core : float  Window height          [m]  inner vertical opening

        Returns
        -------
        A_surface : float  Total exposed surface area [m²]
        """

        outer_perimeter = 2 * (A_core + B_core) * D_core  # [m²] four outer side faces
        inner_perimeter = 2 * (F_core + G_core) * D_core  # [m²] four inner window faces
        end_faces = 2 * (A_core * B_core - F_core * G_core)  # [m²] front and back faces

        A_surface = outer_perimeter + inner_perimeter + end_faces

        return A_surface

    @staticmethod
    def calculate_Ae(method, Ae_user=None, kf=None, D_core=None, E_core=None):

        """
        Calculate or supply the effective cross-sectional area of the inductor core.

        Two methods are supported:

        Method 1 — "user":
            User provides Ae directly from the datasheet Table 2.
            Use this when the manufacturer has already accounted for the stacking
            factor and published the effective area directly.

        Method 2 — "geometry":
            Ae is computed from core geometry and stacking factor:
                Ae = kf * D * E
            where D is the core depth, E is the build (thickness), and kf is the
            stacking factor accounting for gaps between lamination layers.
            Use this when scaling an existing core or estimating a custom core size.
            Reference: Kazimierczuk, M.K., "High-Frequency Magnetic Components",
                       2nd Ed., Wiley-IEEE Press, 2014, Chapter 1.

        Parameters
        ----------
        method : str
            Calculation method. One of: "user", "geometry".

        Ae_user : float, optional
            User-supplied effective cross-sectional area [m²].
            Required when method = "user".

        kf : float, optional
            Core stacking factor [-]; ratio of magnetic material to total cross-section.
            Accounts for gaps between lamination layers.
            For Metglas 2605SA1: kf = 0.82 from [A] Table 2.
            Required when method = "geometry".

        D_core : float, optional
            Depth of the core (cast width) [m]; one side of the cross-section.
            From datasheet Table 1.
            Required when method = "geometry".

        E_core : float, optional
            Build (thickness) of the core [m]; other side of the cross-section.
            From datasheet Table 1.
            Required when method = "geometry".

        Returns
        -------
        Ae : float
            Effective cross-sectional area of the core [m²].
        """

        if method == "user":
            if Ae_user is None:
                raise ValueError("method='user' requires Ae_user to be provided.")
            Ae = Ae_user

        elif method == "geometry":
            if kf is None or D_core is None or E_core is None:
                raise ValueError("method='geometry' requires kf, D_core, and E_core.")
            Ae = kf * D_core * E_core  # [m²]  Ae = kf × D × E

        else:
            raise ValueError(f"Unknown method '{method}'. Choose from: 'user', 'geometry'.")

        return Ae

    @staticmethod
    def calculate_le(method, le_user=None, A_core=None, B_core=None, F_core=None, G_core=None):
        """
        Calculate or supply the effective magnetic path length of the inductor core.

        Two methods are supported:

        Method 1 — "user":
            User provides le directly from the datasheet Table 2.
            Use this when the manufacturer has published the effective magnetic
            path length directly. Preferred when available — more accurate than
            the geometry estimate because it accounts for corner rounding.

        Method 2 — "geometry":
            le is estimated from the core outer and window dimensions as the
            perimeter of the centreline rectangle through the core material:

                leg_w = (A - F) / 2          horizontal leg half-width
                leg_h = (B - G) / 2          vertical leg half-height
                le    = (A + F) + (B + G)    centreline perimeter

            Physical origin: flux travels through the midpoint of each leg,
            not along the outer or inner edge. The centreline lies halfway
            between the outer dimension and the window dimension on each side.
            This formula gives a close approximation; the small residual error
            (~10%) relative to the datasheet value comes from corner rounding
            of the wound core, which shortens the actual path slightly.

            Reference: Kazimierczuk, M.K., "High-Frequency Magnetic Components",
                       2nd Ed., Wiley-IEEE Press, 2014, Chapter 1.

        Parameters
        ----------
        method : str
            Calculation method. One of: "user", "geometry".

        le_user : float, optional
            User-supplied effective magnetic path length [m].
            Required when method = "user".

        A_core : float, optional
            Overall width of the core [m]; outer horizontal dimension.
            From datasheet Table 1. Required when method = "geometry".

        B_core : float, optional
            Overall height of the core [m]; outer vertical dimension.
            From datasheet Table 1. Required when method = "geometry".

        F_core : float, optional
            Width of the core window [m]; inner horizontal opening.
            From datasheet Table 1. Required when method = "geometry".

        G_core : float, optional
            Height of the core window [m]; inner vertical opening.
            From datasheet Table 1. Required when method = "geometry".

        Returns
        -------
        le : float
            Effective magnetic path length [m].
        """

        if method == "user":
            if le_user is None:
                raise ValueError("method='user' requires le_user to be provided.")
            le = le_user

        elif method == "geometry":
            if any(v is None for v in [A_core, B_core, F_core, G_core]):
                raise ValueError("method='geometry' requires A_core, B_core, F_core, and G_core.")

            leg_w = (A_core - F_core) / 2  # [m]  horizontal leg half-width
            leg_h = (B_core - G_core) / 2  # [m]  vertical leg half-height
            le = 2 * (F_core + leg_w) + 2 * (G_core + leg_h)  # [m]  centreline perimeter
            # Simplified: le = (A_core + F_core) + (B_core + G_core)

        else:
            raise ValueError(f"Unknown method '{method}'. Choose from: 'user', 'geometry'.")

        return le

    @staticmethod
    def calculate_core_volume(Ae, le):
        """
        Calculate the effective core volume.

        Ve = Ae × le

        Parameters
        ----------
        Ae : float  Effective cross-sectional area [m²]
        le : float  Effective magnetic path length [m]

        Returns
        -------
        Ve : float  Effective core volume [m³]
        """

        Ve = Ae * le

        return Ve

    @staticmethod
    def calculate_turns(L, I_peak, B_max, Ae):
        """
        Calculate the minimum number of turns required for an inductor.

        Physical origin
        ---------------
        Start from the definition of inductance:
            L = N * Phi / I = N * B * Ae / I

        Rearranging for B:
            B = L * I / (N * Ae)

        At peak current I = I_peak, we want B to stay below B_max:
            B_peak = L * I_peak / (N * Ae) <= B_max

        Solving for the minimum N:
            N >= L * I_peak / (B_max * Ae)

        We round UP to the nearest integer because:
            - N must be a whole number of turns
            - Rounding down would give B_peak > B_max (saturation risk)
            - Rounding up keeps B_peak safely below B_max

        Reference
        ---------
        Kazimierczuk, M.K., "High-Frequency Magnetic Components",
        2nd Ed., Wiley-IEEE Press, 2014, Chapter 2.

        Parameters
        ----------
        L      : float  target inductance            [H]
        I_peak : float  peak current                 [A]
        B_max  : float  maximum allowed flux density [T]
        Ae     : float  effective cross section area [m²]

        Returns
        -------
        N : int
            Minimum number of turns required.
            Unit: [-]
        """
        N = int(np.ceil((L * I_peak) / (B_max * Ae)))
        return N

    @staticmethod
    def calculate_air_gap(mu_0, N, Ae, L, le, mu_r):
        """
        Calculate the required air gap length for a gapped inductor core.

        Physical origin
        ---------------
        From Ampere's law around the magnetic circuit:
            N * I = H_core * le  +  H_gap * lg
        Where:
            H_core = B / (mu_0 * mu_r)
            H_gap  = B / mu_0
        Using B = L * I / (N * Ae) and solving for lg:
            lg = (mu_0 * N² * Ae / L) - (le / mu_r)

        Reference
        ---------
        Kazimierczuk, M.K., "High-Frequency Magnetic Components",
        2nd Ed., Wiley-IEEE Press, 2014, Chapter 2.

        Parameters
        ----------
        mu_0 : float  permeability of free space  [H/m]
        N    : int    number of turns             [-]
        Ae   : float  effective cross section     [m²]
        L    : float  target inductance           [H]
        le   : float  magnetic path length        [m]
        mu_r : float  relative permeability       [-]

        Returns
        -------
        lg : float
            Required air gap length.
            Unit: [m]
        """
        lg = (mu_0 * N ** 2 * Ae / L) - (le / mu_r)
        return lg

    @staticmethod
    def calculate_B_peak(mu_0, N, I_peak, lg, le, mu_r):
        """
        Calculate the peak flux density in the inductor core.

        Physical origin
        ---------------
        From Ampere's law:

            N * I = B * le / (mu_0 * mu_r)  +  B * lg / mu_0
            N * I = B * (le/mu_r + lg) / mu_0

        Solving for B at I = I_peak:

            B_peak = mu_0 * N * I_peak / (lg + le/mu_r)

        Reference
        ---------
        Kazimierczuk, M.K., "High-Frequency Magnetic Components",
        2nd Ed., Wiley-IEEE Press, 2014, Chapter 2.

        Parameters
        ----------
        mu_0   : float  permeability of free space  [H/m]
        N      : int    number of turns             [-]
        I_peak : float  peak current                [A]
        lg     : float  air gap length              [m]
        le     : float  magnetic path length        [m]
        mu_r   : float  relative permeability       [-]

        Returns
        -------
        B_peak : float
            Peak flux density in the core.
            Unit: [T]
        """
        B_peak = (mu_0 * N * I_peak) / (lg + le / mu_r)
        return B_peak

    @staticmethod
    def safety_checks(B_peak, B_max, Bsat, lg, le):
        # ── Check 1: B_peak must be below B_max ──────────────────────────────────────
        if B_peak >= B_max:
            raise ValueError(
                f"\nSAFETY CHECK FAILED: Flux density exceeds maximum operating limit.\n"
                f"B_peak = {B_peak:.4f} T\n"
                f"B_max  = {B_max:.4f} T\n"
                f"\nRecommendation:\n"
                f"  Increase N by 1 and recalculate lg, or\n"
                f"  Increase Ae to reduce required N, or\n"
                f"  Reduce I_peak by using more parallel inductor units.")

        # ── Check 2: B_peak must be below Bsat ───────────────────────────────────────
        if B_peak >= Bsat:
            raise ValueError(
                f"\nSAFETY CHECK FAILED: Flux density exceeds saturation limit.\n"
                f"B_peak = {B_peak:.4f} T\n"
                f"Bsat   = {Bsat:.4f} T\n"
                f"\nRecommendation:\n"
                f"  Core will saturate and inductance will collapse.\n"
                f"  Increase N or increase Ae immediately.")

        # ── Check 3: lg must be positive ─────────────────────────────────────────────
        if lg <= 0:
            raise ValueError(
                f"\nSAFETY CHECK FAILED: Air gap is zero or negative.\n"
                f"lg = {lg * 1000:.2f} mm\n"
                f"\nRecommendation:\n"
                f"  Core is too large for the required inductance.\n"
                f"  Reduce Ae or reduce N.")

        # ── Check 4: lg must be less than le ─────────────────────────────────────────
        if lg >= le:
            raise ValueError(
                f"\nSAFETY CHECK FAILED: Air gap is larger than magnetic path length.\n"
                f"lg = {lg * 1000:.2f} mm\n"
                f"le = {le * 1000:.2f} mm\n"
                f"\nRecommendation:\n"
                f"  This core is far too small for this current level.\n"
                f"  Increase Ae significantly, or\n"
                f"  Use multiple cores in parallel, or\n"
                f"  Use a custom larger core.")

        # ── Check 5: lg/le ratio warning ─────────────────────────────────────────────
        lg_le_ratio = lg / le
        if lg_le_ratio > 0.10:
            raise ValueError(
                f"\nWARNING CHECK 5: Air gap ratio lg/le = {lg_le_ratio * 100:.1f}%\n"
                f"  Recommended maximum is 10%.\n"
                f"  Large air gap causes fringing flux which increases losses.\n"
                f"  Recommendation: Increase Ae to reduce required air gap.")

    @staticmethod
    def calculate_minimum_required_wire_area(I_RMS_rated, J_max):
        """
        Calculate the minimum copper cross-section required to carry the rated
        current within the maximum allowed current density.

        A_wire = I_RMS / J_max
        d_wire = sqrt(4 × A_wire / π)

        Parameters
        ----------
        I_RMS_rated : float  Rated RMS current          [A]
        J_max       : float  Maximum current density    [A/m²]

        Returns
        -------
        A_wire_minimum : float  Minimum copper cross-section  [m²]
        d_wire_minimum : float  Equivalent minimum wire diameter [m]
        """

        A_wire_minimum = I_RMS_rated / J_max
        d_wire_minimum = np.sqrt((4 * A_wire_minimum) / np.pi)

        return A_wire_minimum, d_wire_minimum

    @staticmethod
    def calculate_parallel_strands(A_wire_minimum, A_strand):
        """
        Calculate the number of parallel strands required to achieve the
        minimum copper cross-section from individual strand area.

        N_parallel = ceil(A_wire_minimum / A_strand)

        Ceiling is used because rounding down would give insufficient
        copper area, potentially exceeding the current density limit.

        Parameters
        ----------
        A_wire_minimum : float  Minimum required copper cross-section  [m²]
        A_strand       : float  Bare copper area of one strand         [m²]

        Returns
        -------
        N_parallel : int  Number of parallel strands required  [-]
        """

        N_parallel = int(np.ceil(A_wire_minimum / A_strand))

        return N_parallel

    @staticmethod
    def calculate_actual_wire_area(N_parallel, A_strand):
        """
        Calculate the actual total copper cross-section after rounding up
        the number of parallel strands.

        A_wire_actual = N_parallel × A_strand

        Parameters
        ----------
        N_parallel : int    Number of parallel strands  [-]
        A_strand   : float  Bare copper area per strand [m²]

        Returns
        -------
        A_wire_actual : float  Actual total copper cross-section [m²]
        """

        A_wire_actual = N_parallel * A_strand

        return A_wire_actual

    @staticmethod
    def check_window_fill(N_turns, N_parallel, A_wire_bare, F_core, G_core, kf_window_max):
        """
        Check whether the winding physically fits inside the core window
        using the standard window utilization factor ku defined in Kazimierczuk.

        Definition (Kazimierczuk, Chapter 10):
            ku = (N_turns * N_parallel * A_bare) / A_window

        The limits below use bare copper area — insulation, air gaps, and
        imperfect packing are already absorbed into the empirical limit values:
            ku ≤ 0.3  — hand-wound toroid, random lay
            ku ≤ 0.4  — machine-wound, random lay
            ku ≤ 0.6  — organised / orthocyclic winding

        Reference: Kazimierczuk, M.K., "High-Frequency Magnetic Components", 2nd Ed., Wiley-IEEE Press, 2014, Chapter 10.

        Parameters
        ----------
        N_turns      : int    number of winding turns              [-]
        N_parallel   : int    number of parallel strands per turn  [-]
        A_wire_bare  : float  bare copper area of one strand       [m²]
                              Use A_strand_wire_L1 (not outer area)
        F_core       : float  core window width  (inner)           [m]
        G_core       : float  core window height (inner)           [m]
        kf_window_max: float  maximum allowed ku                   [-]

        Returns
        -------
        ku : float  actual window utilization factor [-]
        """

        A_window = F_core * G_core  # [m²] window area
        A_copper_total = N_turns * N_parallel * A_wire_bare  # [m²] total bare copper area

        ku = A_copper_total / A_window  # [-]  utilization factor

        if ku > kf_window_max:
            raise ValueError(
                f"\nSAFETY CHECK FAILED: Winding does not fit inside core window.\n"
                f"\n  Core window area        : {A_window * 1e6:.1f} mm²"
                f"  (F={F_core * 1e3:.1f} mm × G={G_core * 1e3:.1f} mm)\n"
                f"  Total bare copper area  : {A_copper_total * 1e6:.1f} mm²"
                f"  ({N_turns} turns × {N_parallel} strands"
                f" × {A_wire_bare * 1e6:.4f} mm² per strand)\n"
                f"  Actual ku               : {ku:.3f}\n"
                f"  Maximum allowed ku      : {kf_window_max:.3f}\n"
                f"\nRecommendations:\n"
                f"  (1) Increase core window: scale F_core and G_core up, or\n"
                f"  (2) Reduce N_turns: increase Ae to allow fewer turns, or\n"
                f"  (3) Switch to copper foil or busbar winding\n"
                f"      (standard practice at MW current levels), or\n"
                f"  (4) Use organised winding and set kf_window_max=0.6.")

    @staticmethod
    def calculate_l_turn(D_core, E_core):
        """
        Estimate mean length of one turn for a rectangular toroidal core.

        The copper winding wraps around one leg of the core.
        The mean turn path goes around the perimeter of the core cross-section,
        which is D (depth) × E (build/thickness).

        Physical origin:
            The winding sits at the midpoint of the core leg cross-section.
            Mean turn = perimeter of the D × E rectangle.

        Parameters
        ----------
        D_core : float  core depth (cast width)     [m]   from datasheet Table 1
        E_core : float  core build (thickness)       [m]   from datasheet Table 1

        Returns
        -------
        l_turn : float  mean length of one turn      [m]
        """
        l_turn = 2 * (D_core + E_core)  # [m]  perimeter of D × E rectangle
        return l_turn

    @staticmethod
    def calculate_Rdc(rho, N, l_turn, A_wire):

        """
        Calculate DC winding resistance.
        Assumed  no Skin or Proximity Effect

        Reference
        ---------
        Kazimierczuk, M.K., "High-Frequency Magnetic Components",
        2nd Ed., Wiley-IEEE Press, 2014, Chapter 3, Eq. (3.1).

        Parameters
        ----------
        rho : float  copper resistivity  [ohm·m]
        N          : int    number of turns     [-]
        l_turn     : float  mean turn length    [m]
        A_wire     : float  wire cross section  [m²]

        Returns
        -------
        Rdc : float  DC winding resistance  [ohm]
        """

        Rdc = (rho * N * l_turn) / A_wire
        return Rdc

    @staticmethod
    def compute_I_L_peak_per_harmonic_for_inductor(I_L, f, resolution_per_cycle, Profile_size):
        """
        Decompose the inductor current I_L into peak amplitudes at each harmonic of the fundamental frequency, for each
         second of the mission profile.

        Unlike the capacitor function which targets specific harmonics, this function includes ALL harmonics from order 1
        up to the Nyquist limit. This is required for inductor core loss calculation via Steinmetz's equation (Eq. 8),
        where the f^alpha term amplifies contributions from higher harmonics, making cherry-picking insufficient. The paper
        (Martin-Arroyo et al., ICREPQ 2022) showed that using only 511 harmonics gives 73.6% error, while 5000 harmonics
        gives 0.6% error.

        Note: Returns PEAK amplitudes (not RMS) because the Steinmetz equation requires peak flux density:
         B_j = (mu_0 * N * I_j_peak) / (lg + le/mu_r).

        FFT bin spacing
        ---------------
        The signal is 1 second long with N = resolution_per_cycle * f samples.
        The FFT frequency resolution is therefore:
            df = sampling_rate / N = N / N = 1 Hz per bin
        So FFT bin k corresponds to exactly k Hz.
        Harmonic order j of the fundamental (f Hz) sits at frequency j*f Hz, which is FFT bin j*f. Only these bins carry
        real signal energy. All other bins between harmonics contain only numerical noise and must be discarded — including
        them would corrupt the Steinmetz sum.

        Parameters
        ----------
        I_L : np.ndarray
            Time-domain inductor current signal [A].
            Length = Profile_size * resolution_per_cycle * f
        f : float
            Fundamental frequency [Hz]
        resolution_per_cycle : int
            Number of discrete simulation samples per fundamental cycle. Controls the Nyquist limit: higher values resolve more harmonics.
        Profile_size : int
            Number of seconds in the mission profile.

        Returns
        -------
        I_L1_peak_harmonics : np.ndarray, shape (Profile_size, max_harmonic_order)
            Peak current amplitude [A] at each harmonic of f, for each mission-profile second.
            I_L1_peak_harmonics[i, j] = peak amplitude at harmonic order (j+1) during second i.
        harmonic_orders : np.ndarray, shape (max_harmonic_order,)
            Harmonic order indices [1, 2, 3, ..., max_harmonic_order]. Multiply by f to get frequency in Hz.
        harmonic_freqs : np.ndarray, shape (max_harmonic_order,)
            Physical frequency [Hz] of each harmonic order.
            harmonic_freqs[j] = (j+1) * f
            Spans from f to f_Nyquist = sampling_rate / 2.
        """

        samples_per_second = int(
            resolution_per_cycle * f)  # [samples/s] total samples in one second; also the FFT length N
        N = samples_per_second

        # ----------------------------------------#
        # Harmonic orders and frequencies
        # ----------------------------------------#
        # FFT bin spacing = sampling_rate / N = N / N = 1 Hz per bin
        # → FFT bin k = k Hz exactly
        # → Harmonic order j of fundamental f sits at bin j*f
        #
        # Maximum resolvable frequency = Nyquist = N / 2 Hz
        # Maximum resolvable harmonic order = Nyquist / f = (N/2) / f
        #
        # Example: resolution_per_cycle=3000, f=50
        #   N               = 150,000 samples/s
        #   Nyquist         = 75,000 Hz
        #   max_harmonic    = 75,000 / 50 = 1,500
        #   harmonic_freqs  = [50, 100, 150, ..., 75,000] Hz

        max_harmonic_order = int((N // 2) // f)  # highest harmonic order resolvable within Nyquist limit

        harmonic_orders = np.arange(1, max_harmonic_order + 1)  # [1, 2, 3, ..., max_harmonic_order]
        harmonic_freqs = harmonic_orders * f  # [f, 2f, 3f, ..., max_harmonic_order * f] [Hz]

        # ----------------------------------------#
        # Reshape into (Profile_size, N) matrix
        # ----------------------------------------#
        # Each row = one second of time-domain data
        I_matrix = I_L.reshape(Profile_size, N)  # Shape: (Profile_size, N)

        # ----------------------------------------#
        # FFT for all seconds simultaneously
        # ----------------------------------------#
        # rfft returns N//2 + 1 complex bins for a real input of length N
        # Bin k corresponds to frequency k * (sampling_rate / N) = k * 1 Hz = k Hz
        fft_vals = np.fft.rfft(I_matrix, axis=1)  # Shape: (Profile_size, N//2 + 1)

        # ----------------------------------------#
        # Extract bins at harmonic frequencies only
        # ----------------------------------------#
        # Harmonic order j sits at frequency j*f Hz = FFT bin j*f
        # We must extract ONLY these bins and discard all others.
        # Extracting all bins would include inter-harmonic noise bins,
        # which have no physical meaning but would inflate the Steinmetz sum.
        #
        # bin_indices[j] = harmonic_orders[j] * f = the FFT bin number for harmonic j
        # Example: harmonic 1 → bin 50 (50 Hz), harmonic 200 → bin 10000 (10 kHz)

        bin_indices = (harmonic_orders * int(f)).astype(int)  # FFT bin index for each harmonic order
        fft_harmonic_vals = fft_vals[:, bin_indices]  # Shape: (Profile_size, max_harmonic_order)

        # ----------------------------------------#
        # Convert FFT output to peak amplitudes
        # ----------------------------------------#
        # Factor of 2: rfft only returns positive-frequency bins; the negative-
        #   frequency mirror carries equal energy, so multiply by 2 to recover
        #   the full single-sided peak amplitude.
        # Divide by N: numpy's rfft is unnormalised (sum of inputs, not average),
        #   so dividing by N converts raw FFT magnitude to physical amplitude [A].
        # DC bin (bin 0) is excluded — it has no factor-of-2 correction and
        #   carries no harmonic information relevant to core losses.

        I_L_peak_harmonics = (2 * np.abs(
            fft_harmonic_vals)) / N  # Shape: (Profile_size, max_harmonic_order); Peak amplitude [A]

        return I_L_peak_harmonics, harmonic_orders, harmonic_freqs

    @staticmethod
    def calculate_inductor_core_losses(I_peak_harmonics, harmonic_freqs, mu_0, N, lg, le, mu_r, k, a, b, Ve):
        """
        Compute the total core loss in the inductor L1 for each second of the mission profile using Steinmetz's equation
         applied to all harmonics of the inductor current (Eq. 8, Martin-Arroyo et al., ICREPQ 2022).

        Parameters
        ----------
        I_peak_harmonics : np.ndarray, shape (Profile_size, N//2)
            Peak current amplitude [A] at each harmonic order for each second
            of the mission profile. Row i = second i, column j = harmonic j+1.
        harmonic_freqs : np.ndarray, shape (N//2,)
            Frequency [Hz] of each harmonic order.
        mu_0 : float
            Permeability of free space [H/m]. Physical constant = 4π × 10⁻⁷.
        N : int
            Number of turns in the winding [-].
        lg : float
            Air gap length [m].
        le : float
            Effective magnetic path length of the core [m].
        mu_r : float
            Relative permeability of the core material [-].
        k : float
            Steinmetz loss coefficient [W/m³].
        a : float
            Steinmetz frequency exponent alpha [-].
        b : float
            Steinmetz flux density exponent beta [-].
        Ve : float
            Effective core volume [m³].

        Returns
        -------
        P_c : np.ndarray, shape (Profile_size,)
            Total core loss [W] for each second of the mission profile.
        P_c_matrix : np.ndarray, shape (Profile_size, N//2)
            Core loss contribution [W] per second per harmonic.
        """

        # ----------------------------------------#
        # Step 1 — Peak flux density per harmonic
        # ----------------------------------------#
        # Ampere's law for a gapped core:
        #   B_j = (mu_0 * N * I_j_peak) / (lg + le/mu_r)
        # Shape: (Profile_size, N//2)
        B_j = (mu_0 * N * I_peak_harmonics) / (lg + le / mu_r)

        # ----------------------------------------#
        # Step 2 — Core loss per harmonic
        # ----------------------------------------#
        # Steinmetz equation per harmonic:
        #   P_c_j = k * f_j^alpha * B_j^beta * Ve
        # harmonic_freqs shape (N//2,) broadcasts across (Profile_size, N//2)
        # Shape: (Profile_size, N//2)
        P_c_matrix = k * (harmonic_freqs ** a) * (B_j ** b) * Ve

        # ----------------------------------------#
        # Step 3 — Sum over all harmonics
        # ----------------------------------------#
        # Shape: (Profile_size,)
        P_c = np.sum(P_c_matrix, axis=1)

        return P_c, P_c_matrix

    @staticmethod
    def calculate_winding_losses(I_L, Rdc, resolution_per_cycle, f, Profile_size):
        """
        Calculate DC copper winding losses for each second of the mission profile.

        P_w = Rdc × I_RMS²

        Parameters
        ----------
        I_L                 : np.ndarray  instantaneous inductor current [A]
        Rdc                  : float       DC winding resistance          [Ω]
        resolution_per_cycle : int         samples per fundamental cycle  [-]
        f                    : float       fundamental frequency          [Hz]
        Profile_size         : int         mission profile length         [s]

        Returns
        -------
        P_w    : np.ndarray, shape (Profile_size,)  copper losses per second  [W]
        I_RMS  : np.ndarray, shape (Profile_size,)  RMS current per second    [A]
        """

        samples_per_second = int(resolution_per_cycle * f)
        I_matrix = I_L.reshape(Profile_size, samples_per_second)
        I_RMS = np.sqrt(np.mean(I_matrix ** 2, axis=1))
        P_w = Rdc * I_RMS ** 2

        return P_w, I_RMS

    @staticmethod
    def calculate_inductor_thermal_resistance(method, R_th_user=None, A_surface=None, heat_transfer_coefficient=None, Ve_m3=None):

        """
        Calculate or supply the thermal resistance of the inductor core to ambient.

        Three methods are supported:

        Method 1 — "user":
            User provides R_th directly from a datasheet or measurement.

        Method 2 — "surface_area":
            R_th is computed from the core surface area and convective heat
            transfer coefficient:
                R_th = 1 / (h * A_surface)
            Reference: Incropera, F.P., DeWitt, D.P., Bergman, T.L., Lavine, A.S.,
                       "Fundamentals of Heat and Mass Transfer",
                       7th Ed., Wiley, 2011, Table 1.1

        Method 3 — "empirical":
            R_th is estimated from core volume using the Kazimierczuk empirical
            formula for naturally cooled magnetic cores:
                R_th = 14.5 / Ve^0.37   [K/W], Ve in cm³
            Reference: Kazimierczuk, M.K., "High-Frequency Magnetic Components",
                       2nd Ed., Wiley-IEEE Press, 2014, Chapter 1, Eq. (1.186)

        Parameters
        ----------
        method : str
            Calculation method. One of: "user", "surface_area", "empirical".

        R_th_user : float, optional
            User-supplied thermal resistance [K/W].
            Required when method = "user".

        A_surface : float, optional
            Total exposed surface area of the core [m²].
            Required when method = "surface_area".

        heat_transfer_coefficient : float, optional
            Convective heat transfer coefficient [W/(m²·K)].
            Required when method = "surface_area".
            Typical values:
                10  W/(m²·K) — natural convection, still air
                50  W/(m²·K) — moderate forced air cooling
                250 W/(m²·K) — high-velocity forced air
                500 W/(m²·K) — Liquid Cooling
            Source: Incropera et al., Table 1.1

        Ve_m3  : float, optional
            Effective core volume [m³].
            Required when method = "empirical".

        Returns
        -------
        R_th : float
            Thermal resistance from core to ambient [K/W].
        """

        if method == "user":
            if R_th_user is None:
                raise ValueError("method='user' requires R_th_user to be provided.")
            R_th = R_th_user

        elif method == "surface_area":
            if A_surface is None or heat_transfer_coefficient is None:
                raise ValueError("method='surface_area' requires A_surface and heat_transfer_coefficient.")
            # Single-node lumped thermal model: R_th computed from core surface area only.
            # Core and winding are assumed to be at the same temperature. where a single Rth represents the combined thermal
            # path of the entire inductor assembly to ambient, noting that Rth includes the thermal resistance between the
            # core and ambient, the thermal resistance between the conductor and ambient, and the conduction and radiation
            # thermal resistances between the conductor surface and the core.
            # [1] S. Martín-Arroyo et al., "Core losses analysis of the LCL filter inductor for SiC-based inverter",
            # Renewable Energy and Power Quality Journal (RE&PQJ), Vol. 20, September
            R_th = 1 / (heat_transfer_coefficient * A_surface)

        elif method == "empirical":
            if Ve_m3 is None:
                raise ValueError("method='empirical' requires Ve_cm3 to be provided.")
            Ve_cm3 = Ve_m3 * 1e6  # [cm³]
            R_th = 14.5 / (Ve_cm3 ** 0.37)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose from: 'user', 'surface_area', 'empirical'.")

        return R_th

    @staticmethod
    def calculate_inductor_temperature(T_amb, R_th, P_total):
        """
        Calculate inductor temperature for each second of the mission profile.

        Single-node lumped thermal model:
            T = T_amb + R_th × P_total

        Consistent with Martín-Arroyo et al., ICREPQ 2022, Eq. (1):
            ΔT = (Pw + Pc) · Rth

        Parameters
        ----------
        T_amb   : np.ndarray  ambient temperature          [K]
        R_th    : float       thermal resistance           [K/W]
        P_total : np.ndarray  total losses per second      [W]

        Returns
        -------
        T_inductor : np.ndarray  inductor temperature per second [K]
        """

        T_inductor = T_amb + R_th * P_total

        return T_inductor

    @staticmethod
    def check_insulation_voltage_stress(V_L, N_turns, V_bd, resolution_per_cycle, f, Profile_size):
        """
        Compute peak turn-to-turn voltage and verify it does not exceed
        the insulation breakdown voltage.

        The enamel insulation between two adjacent turns sees approximately
        one turn's worth of the total inductor voltage:
            V_turn = V_L / N_turns

        A well-designed inductor should have V_turn << V_bd.

        Parameters
        ----------
        V_L                  : np.ndarray  instantaneous inductor voltage  [V]
        N_turns              : int         number of winding turns         [-]
        V_bd                 : float       insulation breakdown voltage    [V]
                                           Source: Elektrisola datasheet page 4
                                           Grade 1 = 2400 V, Grade 2 = 4600 V
        resolution_per_cycle : int         samples per fundamental cycle   [-]
        f                    : float       fundamental frequency           [Hz]
        Profile_size         : int         mission profile length          [s]

        Returns
        -------
        V_turn_peak    : np.ndarray, shape (Profile_size,)
                         Peak turn-to-turn voltage per profile second [V]
        V_stress_ratio : np.ndarray, shape (Profile_size,)
                         V_turn_peak / V_bd per profile second        [-]

        Raises
        ------
        ValueError
            If any V_stress_ratio >= 1.0 — insulation will fail immediately.
        """

        # ── Turn-to-turn voltage ──────────────────────────────────────────────
        samples_per_second = int(resolution_per_cycle * f)
        V_matrix = V_L.reshape(Profile_size, samples_per_second)
        V_turn_peak = np.max(np.abs(V_matrix), axis=1) / N_turns

        # ── Stress ratio ──────────────────────────────────────────────────────
        V_stress_ratio = V_turn_peak / V_bd

        # ── Hard failure check ────────────────────────────────────────────────
        if np.any(V_stress_ratio >= 1.0):
            raise ValueError(
                f"\nSAFETY CHECK FAILED: Turn-to-turn voltage exceeds breakdown voltage.\n"
                f"  Peak V_turn  = {np.max(V_turn_peak):.2f} V\n"
                f"  V_bd         = {V_bd:.2f} V\n"
                f"  Stress ratio = {np.max(V_stress_ratio):.4f}\n"
                f"\nRecommendations:\n"
                f"  (1) Increase N to reduce voltage per turn, or\n")

    @staticmethod
    def calculate_inductor_lifetime(T_operating, T_rated, L_rated, Ea, kb, L_max_years):

        """
        Calculate winding insulation lifetime using the Arrhenius thermal aging model.

        Parameters
        ----------
        T_operating : np.ndarray or float
            Operating temperature of the inductor [K].
            Use T_inductor_L1 from the thermal model.
        T_rated : float
            Rated temperature index of the insulation [K].
            Source: Elektrisola datasheet page 3 — 210°C = 483 K for Amidester 200.
        L_rated : float
            Reference lifetime at T_rated [h].
            Source: IEC 60172 — 20,000 h reference point.
        Ea : float
            Activation energy of insulation degradation [J].
        kb : float
            Boltzmann constant [J/K]
        L_max_years : float
            Max Life [years]

        Returns
        -------
        L : np.ndarray or float
            Predicted insulation lifetime [Years] at each operating temperature.
        """

        L = L_rated * np.exp((Ea / kb) * (1 / T_operating - 1 / T_rated))
        L = L / (365 * 24)
        #L = np.minimum(L, L_max_years)  # cap at 30 years
        return L

    @staticmethod
    def miners_rule_modified(L_per_second, seconds_per_sample):
        """
        Apply Miner's cumulative damage rule to compute expected total lifetime
        and the fraction of life consumed by one mission profile cycle.

        Miner's rule states that failure occurs when cumulative damage D = 1:
            D = sum( dt_i / L_i )
        where dt_i is the time spent at condition i and L_i is the lifetime
        at that condition. The expected total lifetime is then:
            L_total = Profile_duration / D_cycle

        Reference
        ---------
        Miner, M.A., "Cumulative damage in fatigue",
        Journal of Applied Mechanics, 1945.
        IEC 60216-1: Electrical insulating materials — Thermal endurance properties.

        Parameters
        ----------
        L_per_second : np.ndarray, shape (Profile_size,)
            Predicted insulation lifetime [years] at each second of the mission
            profile, computed from the Arrhenius thermal aging model.
            L_per_second[i] = lifetime [years] if the inductor operated forever
            at the temperature of second i.

        seconds_per_sample : float or np.ndarray, optional
            Duration [s] that each entry of L_per_second represents.
            Default 1.0 (one entry = one second, original behaviour).
            Pass a scalar (e.g. 86400 for daily points) or an array of
            per-sample durations for an irregular profile.

        Returns
        -------
        L_total : float
            Total predicted insulation lifetime [years].
            Defined as the time until cumulative damage D reaches 1.0.
            Computed by repeating the mission profile until failure:
                L_total = Profile_duration / D_cycle

        life_consumed_percent : float
            Fraction of total insulation life consumed by ONE complete run
            of the mission profile [%].
            Range: 0% to 100%. Failure occurs when cumulative total reaches 100%.
            life_consumed_percent = (Profile_duration / L_total) * 100


        """
        sec_per_year = 365 * 24 * 3600
        dt_years = seconds_per_sample / sec_per_year  # scalar or array, [years] per sample
        d_i = dt_years / L_per_second  # fractional damage per sample
        D_cycle = np.sum(d_i)
        Profile_duration = np.sum(np.broadcast_to(dt_years, np.shape(L_per_second)))
        L_total = Profile_duration / D_cycle
        life_consumed_percent = D_cycle * 100
        if life_consumed_percent > 100:
            life_consumed_percent = 100.0
        return L_total, life_consumed_percent, D_cycle

    @staticmethod
    def miners_rule_modified_capacitor(L_per_second, seconds_per_sample, C_0 , D_cum_previous, k_capacitance):
        """
        Apply Miner's cumulative damage rule to compute expected total lifetime,
        the fraction of life consumed by one mission profile cycle, and
        optionally the degraded capacitance at the end of that cycle.

        Miner's rule states that failure occurs when cumulative damage D = 1:
            D = sum( dt_i / L_i )
        where dt_i is the time spent at condition i and L_i is the lifetime
        at that condition. The expected total lifetime is then:
            L_total = Profile_duration / D_cycle

        For a film capacitor the accumulated damage is additionally mapped to a
        loss of capacitance by the linear convention

            C_new = C_0 * (1 - k_capacitance * D_cumulative)

        so that a cumulative damage of unity corresponds to a capacitance loss of
        k_capacitance. The default k_capacitance = 0.20 follows MIL-C-62F, under
        which a capacitor is considered unhealthy once its capacitance has fallen
        20% below its pristine value. The mapping is applied to the pristine C_0
        rather than to the previous value, so that the result depends only on the
        cumulative damage and not on how the mission profile is divided in time.

        Reference
        ---------
        Miner, M.A., "Cumulative damage in fatigue",
        Journal of Applied Mechanics, 1945.
        IEC 60216-1: Electrical insulating materials — Thermal endurance properties.
        MIL-PRF-19978 / MIL-C-62F, U.S. Department of Defense, 2008.

        Parameters
        ----------
        L_per_second : np.ndarray, shape (Profile_size,)
            Predicted insulation lifetime [years] at each sample of the mission
            profile, computed from the Arrhenius thermal aging model.
            L_per_second[i] = lifetime [years] if the component operated forever
            at the condition of sample i.

        seconds_per_sample : float or np.ndarray, optional
            Duration [s] that each entry of L_per_second represents.
            Default 1.0 (one entry = one second, original behaviour).
            Pass a scalar (e.g. 86400 for daily points) or an array of
            per-sample durations for an irregular profile.

        C_0 : float or None, optional
            Pristine (design) capacitance [F]. If None, no capacitance is
            computed and C_new is returned as None. Use None for the inductors.

        D_cum_previous : float, optional
            Cumulative Miner damage accrued before this mission profile run [-].
            Zero for the first year. Default 0.0.

        k_capacitance : float, optional
            Fractional capacitance loss corresponding to a cumulative damage of
            unity [-]. Default 0.20 (MIL-C-62F).

        Returns
        -------
        L_total : float
            Total predicted lifetime [years], the time until cumulative damage
            reaches 1.0, obtained by repeating the mission profile until failure:
                L_total = Profile_duration / D_cycle

        life_consumed_percent : float
            Fraction of total life consumed by ONE complete run of the mission
            profile [%], clipped to 100. Failure occurs at 100%.

        C_new : float or None
            Capacitance [F] at the end of this mission profile run, given the
            cumulative damage D_cum_previous + D_cycle. Returned as None when
            C_0 is None. Clamped to the range [(1 - k_capacitance) * C_0, C_0].
        """
        sec_per_year = 365 * 24 * 3600
        dt_years = seconds_per_sample / sec_per_year  # scalar or array, [years] per sample
        d_i = dt_years / L_per_second  # fractional damage per sample
        D_cycle = np.sum(d_i)
        Profile_duration = np.sum(np.broadcast_to(dt_years, np.shape(L_per_second)))
        L_total = Profile_duration / D_cycle

        life_consumed_percent = D_cycle * 100
        if life_consumed_percent > 100:
            life_consumed_percent = 100.0

        if C_0 is None:
            C_new = None
            D_cum = D_cum_previous
        else:
            D_cum = min(D_cum_previous + D_cycle, 1.0)
            C_new = C_0 * (1.0 - k_capacitance * D_cum)

        return L_total, life_consumed_percent, C_new, D_cum, D_cycle


    @staticmethod
    def calculate_capacitor_thermal_resistance(method, case_shape=None,
                                               D_case=None, H_case=None,
                                               W_case=None, L_case=None,
                                               heat_transfer_coefficient=None,
                                               R_th_user=None):
        """
        Calculate thermal resistance of the capacitor to ambient.

        Two methods are supported:

        Method 1 — "user":
            User provides R_th directly from datasheet or measurement.

        Method 2 — "surface_area":
            R_th computed from the capacitor case geometry and the convective
            heat-transfer coefficient. The case may be a cylindrical can or a
            rectangular box, selected via case_shape:

            case_shape = "cylinder":
                A_surface = pi*D*H + 2*pi*(D/2)^2     [m²]
                            (lateral side + two circular end caps)
                Requires: D_case, H_case
                Source: TDK B3236X datasheet (cylindrical can), column D, H.

            case_shape = "box":
                A_surface = 2*(W*H + W*L + H*L)        [m²]
                            (all six rectangular faces)
                Requires: W_case, H_case, L_case
                Source: TDK B32354S datasheet, dimensions w x h x l.

            In both cases:
                R_th = 1 / (h * A_surface)             [K/W]
            Reference: Incropera et al., "Fundamentals of Heat and Mass Transfer",
                       7th Ed., Wiley, 2011, Table 1.1

        Parameters
        ----------
        method : str
            Calculation method. One of: "user", "surface_area".
        case_shape : str, optional
            Capacitor case shape when method = "surface_area".
            One of: "cylinder", "box".
        D_case : float, optional
            Outer diameter of cylindrical can [m]. Required for case_shape="cylinder".
        H_case : float, optional
            Height of the case [m]. Required for both shapes.
        W_case : float, optional
            Width of rectangular box [m]. Required for case_shape="box".
        L_case : float, optional
            Length (depth) of rectangular box [m]. Required for case_shape="box".
        heat_transfer_coefficient : float, optional
            Convective heat-transfer coefficient [W/(m²·K)].
            Required when method = "surface_area".
            Typical values:
                10  W/(m²·K) — natural convection, still air
                50  W/(m²·K) — moderate forced air cooling
                250 W/(m²·K) — high-velocity forced air
            Source: Incropera et al., Table 1.1
        R_th_user : float, optional
            User-supplied thermal resistance [K/W]. Required when method = "user".

        Returns
        -------
        R_th : float
            Thermal resistance from capacitor hotspot to ambient [K/W].
        """

        if method == "user":
            if R_th_user is None:
                raise ValueError("method='user' requires R_th_user.")
            R_th = R_th_user

        elif method == "surface_area":
            if heat_transfer_coefficient is None:
                raise ValueError("method='surface_area' requires heat_transfer_coefficient.")
            if case_shape is None:
                raise ValueError("method='surface_area' requires case_shape "
                                 "('cylinder' or 'box').")

            if case_shape == "cylinder":
                if D_case is None or H_case is None:
                    raise ValueError("case_shape='cylinder' requires D_case and H_case.")
                A_lateral = np.pi * D_case * H_case  # [m²] cylindrical side
                A_end_caps = 2 * np.pi * (D_case / 2) ** 2  # [m²] two circular ends
                A_surface = A_lateral + A_end_caps  # [m²] total surface area

            elif case_shape == "box":
                if W_case is None or H_case is None or L_case is None:
                    raise ValueError("case_shape='box' requires W_case, H_case, and L_case.")
                A_surface = 2 * (W_case * H_case
                                 + W_case * L_case
                                 + H_case * L_case)  # [m²] six rectangular faces

            else:
                raise ValueError(f"Unknown case_shape '{case_shape}'. "
                                 "Choose from: 'cylinder', 'box'.")

            R_th = 1 / (heat_transfer_coefficient * A_surface)  # [K/W]

        else:
            raise ValueError(f"Unknown method '{method}'. Choose from: 'user', 'surface_area'.")

        return R_th

    @staticmethod
    def calculate_tan_delta_0(tan_delta_measured, Rs, C, f_measured):

        """
        Calculate the dielectric loss tangent tan_delta_0 of a polypropylene
        film capacitor from the measured total dissipation factor.

        Physical origin
        ---------------
        The total dissipation factor of a capacitor has two contributions (TDK B3236X datasheet, page 16):
            tan_delta(f) = tan_delta_0 + Rs * omega * C
        Where:
            tan_delta_0   = dielectric loss tangent (frequency-independent material property)
            Rs * omega * C = resistive loss contribution (frequency-dependent)
        Rearranging for tan_delta_0:
            tan_delta_0 = tan_delta(f) - Rs * omega * C
        Reference
        ---------
        TDK Electronics, "FilterCap MKP AC — Single phase, B3236X Series",
        Datasheet version 08, 2022-05-03:
            - Equation: page 16
            - tan_delta measured value: page 3, row "Dissipation factor tan δ at 100 Hz"
            - Rs value: page 10, product table, column Rs

        Parameters
        ----------
        tan_delta_measured : float
            Total measured dissipation factor [-] at frequency f_measured.
            Source: TDK B3236X datasheet page 3 — tan δ ≤ 1.0×10⁻³ at 100 Hz.
            Use the maximum specified value (1.0e-3) as a conservative estimate.

        Rs : float
            Series resistance of the capacitor [Ohm].
            Source: TDK B3236X datasheet page 10, product table, column Rs.
            For B32362A4157J080: Rs = 1.9e-3 Ohm.

        C : float
            Capacitance [F].
            Source: TDK B3236X datasheet page 10, product table, column C_R.
            For B32362A4157J080: C = 150e-6 F.

        f_measured : float, optional
            Frequency at which tan_delta_measured was specified [Hz].
            Default = 100 Hz.
            Source: TDK B3236X datasheet page 3 — measurement frequency is 100 Hz.

        Returns
        -------
        tan_delta_0 : float
            Dielectric loss tangent of the polypropylene film [-].
            Frequency-independent material property of the dielectric.
            Used in capacitor power loss calculations:
                P_D = V_peak² * pi * f * C * tan_delta_0
        """

        omega = 2 * np.pi * f_measured  # [rad/s] angular frequency
        resistive_term = Rs * omega * C  # [-]     resistive contribution
        tan_delta_0 = tan_delta_measured - resistive_term  # [-]     dielectric contribution

        if tan_delta_0 <= 0:
            raise ValueError(
                f"\ntan_delta_0 came out negative or zero: {tan_delta_0:.4e}\n"
                f"This means the resistive term Rs*omega*C = {resistive_term:.4e}\n"
                f"exceeds the measured tan_delta = {tan_delta_measured:.4e}.\n"
                f"Check that Rs, C, and f_measured are consistent with "
                f"the measurement conditions.")

        return tan_delta_0

    @staticmethod
    def calculate_capacitor_lifetime_analytical(T_operating, V_C_RMS, V_C_RMS_Rated, t1, T1, A, n):
        """
        Estimate capacitor lifetime using the TDK analytical lifetime formula.

        Formula (TDK B3236X datasheet, page 19):
            t2 = t1 * exp((T1 - T2) / A) * (V1 / V2)^n

        This formula models two independent degradation mechanisms:
            1. Thermal aging    → exponential term exp((T1-T2)/A)
            2. Voltage stress   → power law term (V1/V2)^n

        Parameters
        ----------
        T_operating : np.ndarray, shape (Profile_size,)
            Capacitor hotspot temperature at each second of mission profile [K].
            Converted internally to °C for the formula.

        V_C_RMS : np.ndarray, shape (Profile_size,)
            RMS capacitor voltage at each second of mission profile [V].

        V_C_RMS_Rated : float
            Rated RMS voltage of the capacitor [V].
            V1 in the formula.
            Source: TDK B3236X datasheet page 10 — 480 V for B32362A4157J080.

        t1 : float, optional
            Reference lifetime at T1 and V1 [hours].
            Default = 100,000 hours.
            Source: TDK B3236X datasheet page 3 — life expectancy at V_RMS, |ΔC/C| ≤ 3%.

        T1 : float, optional
            Reference temperature [K].
            Source: TDK B3236X datasheet page 19 — example reference temperature.

        A : float, optional
            Thermal acceleration factor [°C].
            Default = 8.5°C.
            Source: Standard value for metallized polypropylene film capacitors.
                    Not specified in TDK B3236X datasheet — literature value.
                    Reference: Kemet, "Film Capacitor Lifetime Estimation", 2018.

        n : float, optional
            Voltage acceleration factor [-].
            Default = 9.4.
            Source: Standard value for metallized polypropylene (MKP) film capacitors.
                    Not specified in TDK B3236X datasheet — literature value.
                    Reference: IEC 61071, MKP industry standard.

        Returns
        -------
        t2_years : np.ndarray, shape (Profile_size,)
            Estimated capacitor lifetime [years] at each second of the mission
            profile, given the operating temperature and voltage at that second.
        """

        # Convert operating temperature from Kelvin to Celsius
        T2 = T_operating - 273.15  # [°C]
        T1 = T1 - 273.15  # [°C]

        # Voltage stress ratio V1/V2
        # V1 = V_C_RMS_Rated (reference rated voltage)
        # V2 = V_C_RMS       (actual operating voltage)
        # If V2 > V1 → ratio < 1 → lifetime decreases
        # If V2 < V1 → ratio > 1 → lifetime increases
        voltage_ratio = V_C_RMS_Rated / V_C_RMS  # [-]  V1/V2

        # Thermal acceleration term
        thermal_term = np.exp((T1 - T2) / A)  # [-]

        # Voltage acceleration term
        voltage_term = voltage_ratio ** n  # [-]

        # Estimated lifetime in hours
        t2_hours = t1 * thermal_term * voltage_term  # [hours]

        # Convert to years
        t2_years = t2_hours / (365 * 24)  # [years]

        return t2_years

    @staticmethod
    def check_within_tolerance(values, tol=0.10):
        """
        Verify each user-chosen component value is within `tol` (fractional) of the
        design-function optimum.

        Parameters
        ----------
        values : dict   name -> (actual, optimum)
        tol    : float  allowed fractional deviation (0.10 = 10%)

        Raises
        ------
        ValueError if any |actual - optimum| / |optimum| > tol.
        """
        offending = []
        for name, (actual, optimum) in values.items():
            actual, optimum = float(actual), float(optimum)
            dev = abs(actual - optimum) / abs(optimum) if optimum != 0 else float('inf')
            if dev > tol:
                offending.append((name, actual, optimum, dev))

        if offending:
            lines = [f"\nComponent value(s) outside ±{tol * 100:.0f}% of the design optimum:\n"]
            lines.append(f"  {'Component':<16}{'Actual':>16}{'Optimum':>16}{'Deviation':>12}")
            lines.append("  " + "-" * 60)
            for name, actual, optimum, dev in offending:
                lines.append(f"  {name:<16}{actual:>16.6e}{optimum:>16.6e}{dev * 100:>11.2f}%")
            lines.append(
                "\nEither bring the chosen values closer to the optimum, "
                "or relax `tol` if the deviation is intentional."
            )
            raise ValueError("\n".join(lines))
    '''
    @staticmethod
    def Three_phase_switching_output(t, Vs_ref, Vo, Tsw, f, Profile_size, sample="center"):
        """
        Produce the switched phase-A inverter voltage Vs from its reference Vs_ref,
        for a three-phase, three-wire inverter, using exact switching instants rendered
        onto the existing (coarse) time grid by volt-second-preserving averaging.

        Method
        ------
        1. Build three modulating references at 0, -120, +120 deg from Vs_ref
           (analytic time-shift, not Hilbert).
        2. Add min-max zero-sequence injection (SVPWM).
        3. For each of the three legs, use SYMMETRIC REGULAR SAMPLING: sample the
           modulating signal once per carrier period (at the carrier centre), and
           compute the two switching instants analytically from the carrier geometry.
           This places every edge exactly, with no dependence on grid resolution.
        4. Render each leg onto the coarse grid by assigning every sample the
           *time-average* of the pole voltage over its dt window (volt-second exact,
           anti-aliased -- not point-sampled, so no jitter).
        5. Remove common mode: Vs = Vao - (Vao + Vbo + Vco)/3  (the phase voltage a
           3-wire load actually sees).

        The carrier is the same "V"-shaped triangle used previously:
            c(tau) = 4*|tau - 0.5| - 1,   tau = (t mod Tsw)/Tsw
        A leg is ON (+Vo) when its modulating signal m >= c, which occurs on the
        interval [tau1, tau2] with
            tau1 = (1 - m)/4,  tau2 = (m + 3)/4,  duty = tau2 - tau1 = (m + 1)/2.

        Parameters
        ----------
        t : np.ndarray
            Uniform time grid [s], length Profile_size * samples_per_second.
        Vs_ref : np.ndarray
            Phase-A reference voltage [V], same length as t.
        Vo : float or np.ndarray
            Pole-voltage amplitude (Vdc/2) [V]. Scalar, or per-second profile of
            length Profile_size.
        Tsw : float
            Switching period [s].
        f : float
            Fundamental frequency [Hz].
        Profile_size : int
            Number of 1-second mission-profile points.
        sample : {"center", "natural"}
            "center"  -> symmetric regular sampling (one m per carrier period, taken at the period centre). Recommended: exact, jitter-free.
            "natural" -> natural sampling (m taken at the analytic crossing). Almost identical here because f << fsw.

        Returns
        -------
        Vs : np.ndarray
            Switched phase-A voltage [V] on the grid t (volt-second averaged).
        """

        t = np.asarray(t, dtype=float)
        Vs_ref = np.asarray(Vs_ref, dtype=float)
        if t.shape != Vs_ref.shape:
            raise ValueError("t and Vs_ref must have the same shape.")

        n = t.size
        dt = float(t[1] - t[0])
        samples_per_second = n // int(Profile_size)
        if samples_per_second * int(Profile_size) != n:
            raise ValueError("len(t) must be Profile_size * samples_per_second.")

        # Per-sample pole amplitude Vo(t)
        Vo_arr = np.atleast_1d(np.asarray(Vo, dtype=float))
        if Vo_arr.size == 1:
            Vo_t = np.full(n, Vo_arr[0])
        elif Vo_arr.size == int(Profile_size):
            Vo_t = np.repeat(Vo_arr, samples_per_second)
        else:
            raise ValueError("Vo must be scalar or length Profile_size.")
        if np.any(Vo_t <= 0):
            raise ValueError("Vo must be positive everywhere.")

        # ------------------------------------------------------------------ #
        # 1) Three references at 0, -120, +120 deg via per-second time shift
        # ------------------------------------------------------------------ #
        period_samples = int(round((1.0 / f) / dt))  # samples in one fundamental cycle
        if period_samples % 3 != 0:
            # not fatal, but the 120 deg shift is only exact if divisible by 3
            pass
        shift = period_samples // 3

        block = Vs_ref.reshape(int(Profile_size), samples_per_second)
        ref_a = Vs_ref
        ref_b = np.roll(block, +shift, axis=1).reshape(n)  # lags a by 120 deg
        ref_c = np.roll(block, +2 * shift, axis=1).reshape(n)  # leads a by 120 deg

        m_a = ref_a / Vo_t
        m_b = ref_b / Vo_t
        m_c = ref_c / Vo_t

        # ------------------------------------------------------------------ #
        # 2) Min-max zero-sequence injection (SVPWM)
        # ------------------------------------------------------------------ #
        m_stack = np.vstack([m_a, m_b, m_c])
        v_zs = -0.5 * (m_stack.max(axis=0) + m_stack.min(axis=0))
        m_a = np.clip(m_a + v_zs, -1.0, 1.0)
        m_b = np.clip(m_b + v_zs, -1.0, 1.0)
        m_c = np.clip(m_c + v_zs, -1.0, 1.0)

        # ------------------------------------------------------------------ #
        # 3+4) Exact edges + volt-second rendering, per leg
        # ------------------------------------------------------------------ #
        t_end = t[-1] + dt
        n_periods = int(np.ceil(t_end / Tsw))
        k = np.arange(n_periods)
        t_k0 = k * Tsw  # start time of each carrier period
        t_kc = (k + 0.5) * Tsw  # centre time of each carrier period

        # cell boundaries: A(t) evaluated here, differenced to get per-cell on-time
        bounds = np.empty(n + 1)
        bounds[:n] = t
        bounds[n] = t_end

        def render_leg(m_leg):
            # Sample the modulating signal once per carrier period
            if sample == "center":
                m_k = np.interp(t_kc, t, m_leg)
            elif sample == "natural":
                # natural sampling differs only at second order when f << fsw;
                # use the centre sample as the crossing estimate
                m_k = np.interp(t_kc, t, m_leg)
            else:
                raise ValueError("sample must be 'center' or 'natural'.")
            m_k = np.clip(m_k, -1.0, 1.0)

            # Exact switching instants within each period
            tau1 = (1.0 - m_k) / 4.0
            tau2 = (m_k + 3.0) / 4.0
            R = t_k0 + tau1 * Tsw  # rising edge (leg turns ON)
            F = t_k0 + tau2 * Tsw  # falling edge (leg turns OFF)
            dur = F - R  # ON duration this period

            # Cumulative ON-time function A(t), piecewise linear through knots.
            cum_before = np.concatenate(([0.0], np.cumsum(dur)[:-1]))
            knot_t = np.empty(2 * n_periods)
            knot_A = np.empty(2 * n_periods)
            knot_t[0::2] = R
            knot_t[1::2] = F
            knot_A[0::2] = cum_before
            knot_A[1::2] = cum_before + dur

            # A at cell boundaries -> on-time per cell -> fraction on
            A_bounds = np.interp(bounds, knot_t, knot_A, left=knot_A[0], right=knot_A[-1])
            on_time = np.diff(A_bounds)
            frac_on = on_time / dt

            # Pole voltage averaged over each cell: +Vo when on, -Vo when off
            return Vo_t * (2.0 * frac_on - 1.0)

        Vao = render_leg(m_a)
        Vbo = render_leg(m_b)
        Vco = render_leg(m_c)

        # ------------------------------------------------------------------ #
        # 5) Common-mode removal -> phase-A voltage seen by the LCL
        # ------------------------------------------------------------------ #
        V_cm = (Vao + Vbo + Vco) / 3.0
        Vs = Vao - V_cm
        return Vs
    '''

    @staticmethod
    def Three_phase_switching_output(t, Vs_ref, Vo, Tsw, f, Profile_size, modulation, sample="center" ):
        """
        Produce the switched phase-A inverter voltage Vs from its reference Vs_ref,
        for a three-phase, three-wire inverter, using exact switching instants rendered
        onto the existing (coarse) time grid by volt-second-preserving averaging.

        modulation : {"spwm", "svm"}
            "spwm" -> sinusoidal PWM, no zero-sequence injection
            "svm"  -> min-max zero-sequence injection (space vector modulation)
        """

        t = np.asarray(t, dtype=float)
        Vs_ref = np.asarray(Vs_ref, dtype=float)
        if t.shape != Vs_ref.shape:
            raise ValueError("t and Vs_ref must have the same shape.")

        n = t.size
        dt = float(t[1] - t[0])
        samples_per_second = n // int(Profile_size)
        if samples_per_second * int(Profile_size) != n:
            raise ValueError("len(t) must be Profile_size * samples_per_second.")

        # Per-sample pole amplitude Vo(t)
        Vo_arr = np.atleast_1d(np.asarray(Vo, dtype=float))
        if Vo_arr.size == 1:
            Vo_t = np.full(n, Vo_arr[0])
        elif Vo_arr.size == int(Profile_size):
            Vo_t = np.repeat(Vo_arr, samples_per_second)
        else:
            raise ValueError("Vo must be scalar or length Profile_size.")
        if np.any(Vo_t <= 0):
            raise ValueError("Vo must be positive everywhere.")

        # ------------------------------------------------------------------ #
        # 1) Three references at 0, -120, +120 deg via per-second time shift
        # ------------------------------------------------------------------ #
        period_samples = int(round((1.0 / f) / dt))
        shift = period_samples // 3

        block = Vs_ref.reshape(int(Profile_size), samples_per_second)
        ref_a = Vs_ref
        ref_b = np.roll(block, +shift, axis=1).reshape(n)
        ref_c = np.roll(block, +2 * shift, axis=1).reshape(n)

        m_a = ref_a / Vo_t
        m_b = ref_b / Vo_t
        m_c = ref_c / Vo_t

        # ------------------------------------------------------------------ #
        # 2) Zero-sequence injection, only for SVM
        # ------------------------------------------------------------------ #
        if modulation == "svm":
            m_stack = np.vstack([m_a, m_b, m_c])
            v_zs = -0.5 * (m_stack.max(axis=0) + m_stack.min(axis=0))
            m_a = m_a + v_zs
            m_b = m_b + v_zs
            m_c = m_c + v_zs
        elif modulation != "spwm":
            raise ValueError("modulation must be 'spwm' or 'svm'.")

        m_a = np.clip(m_a, -1.0, 1.0)
        m_b = np.clip(m_b, -1.0, 1.0)
        m_c = np.clip(m_c, -1.0, 1.0)

        # ------------------------------------------------------------------ #
        # 3+4) Exact edges + volt-second rendering, per leg
        # ------------------------------------------------------------------ #
        t_end = t[-1] + dt
        n_periods = int(np.ceil(t_end / Tsw))
        k = np.arange(n_periods)
        t_k0 = k * Tsw
        t_kc = (k + 0.5) * Tsw

        bounds = np.empty(n + 1)
        bounds[:n] = t
        bounds[n] = t_end

        def render_leg(m_leg):
            if sample not in ("center", "natural"):
                raise ValueError("sample must be 'center' or 'natural'.")
            m_k = np.interp(t_kc, t, m_leg)
            m_k = np.clip(m_k, -1.0, 1.0)

            tau1 = (1.0 - m_k) / 4.0
            tau2 = (m_k + 3.0) / 4.0
            R = t_k0 + tau1 * Tsw
            F = t_k0 + tau2 * Tsw
            dur = F - R

            cum_before = np.concatenate(([0.0], np.cumsum(dur)[:-1]))
            knot_t = np.empty(2 * n_periods)
            knot_A = np.empty(2 * n_periods)
            knot_t[0::2] = R
            knot_t[1::2] = F
            knot_A[0::2] = cum_before
            knot_A[1::2] = cum_before + dur

            A_bounds = np.interp(bounds, knot_t, knot_A,
                                 left=knot_A[0], right=knot_A[-1])
            on_time = np.diff(A_bounds)
            frac_on = on_time / dt

            return Vo_t * (2.0 * frac_on - 1.0)

        Vao = render_leg(m_a)
        Vbo = render_leg(m_b)
        Vco = render_leg(m_c)

        # ------------------------------------------------------------------ #
        # 5) Common-mode removal -> phase-A voltage seen by the LCL
        # ------------------------------------------------------------------ #
        V_cm = (Vao + Vbo + Vco) / 3.0
        Vs = Vao - V_cm
        return Vs

    @staticmethod
    def check_Vs_quality(t, Vs, Vs_ref, f, fsw, Profile_size, amp_tol=0.03, phase_tol_deg=1.0, avg_rms_tol=0.02, baseband_tol=0.01, carrier_guard=20, raise_on_fail=True):

        """
        Validate that a switched inverter voltage Vs faithfully represents its reference Vs_ref, via three checks. Analysis
        is done per mission-profile second (1 s blocks -> 1 Hz FFT bins -> harmonic order j sits at bin j*f), and the WORST
         second is used for the pass/fail decision.

        Check 1 - Fundamental match
            FFT both signals, compare the f-Hz component.
            Pass if  |amp_err| <= amp_tol  AND  |phase_err| <= phase_tol_deg.

        Check 2 - Volt-second tracking
            Moving-average Vs over one switching period Tsw and compare to Vs_ref.
            Pass if  RMS(avg - ref) / fundamental_peak <= avg_rms_tol.

        Check 3 - Baseband cleanliness
            Inspect harmonics from order 2 up to just below the carrier (fsw/f).
            In ideal linear-region PWM these are ~0 (energy lives in carrier groups);
            anything large here is numerical error.
            Pass if  largest single baseband harmonic / fundamental <= baseband_tol.

        Parameters
        ----------
        t, Vs, Vs_ref : np.ndarray   time grid and the two voltages [V]
        f, fsw        : float        fundamental and switching frequency [Hz]
        Profile_size  : int          number of 1-second blocks
        amp_tol       : float        fundamental amplitude tolerance (0.03 = 3%)
        phase_tol_deg : float        fundamental phase tolerance [deg]
        avg_rms_tol   : float        volt-second RMS tolerance, fraction of peak
        baseband_tol  : float        max single baseband harmonic, fraction of fund
        carrier_guard : float The safety margin (in harmonic orders)
        raise_on_fail : bool         raise ValueError if any check fails

        Returns
        -------
        result : dict   {'passed': bool, 'check1', 'check2', 'check3', per-second arrays}
        """

        t = np.asarray(t, float);
        Vs = np.asarray(Vs, float);
        Vs_ref = np.asarray(Vs_ref, float)
        n = t.size
        dt = float(t[1] - t[0])
        P = int(Profile_size)
        N = n // P
        if N * P != n:
            raise ValueError("len(t) must be Profile_size * samples_per_second.")

        f_bin = int(round(f))
        mf = fsw / f
        base_top = max(3, int(np.floor(mf)) - int(carrier_guard))
        base_orders = np.arange(2, base_top)
        base_bins = (base_orders * f_bin).astype(int)
        w = max(int(round((1.0 / fsw) / dt)), 1)

        Vs_b = Vs.reshape(P, N)
        ref_b = Vs_ref.reshape(P, N)

        X = np.fft.rfft(Vs_b, axis=1)
        Xr = np.fft.rfft(ref_b, axis=1)

        # Check 1 - fundamental
        a_vs = 2 * np.abs(X[:, f_bin]) / N
        a_rf = 2 * np.abs(Xr[:, f_bin]) / N
        amp_err = np.where(a_rf != 0, (a_vs - a_rf) / a_rf, np.inf)
        d = np.angle(X[:, f_bin]) - np.angle(Xr[:, f_bin])
        ph_err = np.degrees((d + np.pi) % (2 * np.pi) - np.pi)

        # Check 3 - baseband
        base_amp = 2 * np.abs(X[:, base_bins]) / N
        base_peak = base_amp.max(axis=1) / a_rf
        base_thd = np.sqrt((base_amp ** 2).sum(axis=1)) / a_rf

        # Check 2 - volt-second tracking (centred Tsw moving average vs ref)
        cs = np.concatenate([np.zeros((P, 1)), np.cumsum(Vs_b, axis=1)], axis=1)
        ma = (cs[:, w:] - cs[:, :-w]) / w  # window [i, i+w), shape (P, N-w+1)
        off = w // 2
        ref_aligned = ref_b[:, off:off + ma.shape[1]]
        avg_rms = np.sqrt(np.mean((ma - ref_aligned) ** 2, axis=1)) / a_rf

        # worst-second summary
        c1_amp = np.abs(amp_err).max();
        c1_ph = np.abs(ph_err).max()
        c2 = avg_rms.max();
        c3_peak = base_peak.max();
        c3_thd = base_thd.max()

        amp_ok = c1_amp <= amp_tol
        phase_ok = c1_ph <= phase_tol_deg
        p1 = amp_ok and phase_ok
        p2 = c2 <= avg_rms_tol
        p3 = c3_peak <= baseband_tol
        passed = p1 and p2 and p3

        # specific failure reasons
        reasons = []
        if not amp_ok:
            reasons.append(f"Check 1 (fundamental amplitude): {c1_amp * 100:.3f}% > tol {amp_tol * 100:.1f}%")
        if not phase_ok:
            reasons.append(f"Check 1 (fundamental phase): {c1_ph:.3f} deg > tol {phase_tol_deg:.1f} deg")
        if not p2:
            reasons.append(f"Check 2 (volt-second tracking): {c2 * 100:.3f}% > tol {avg_rms_tol * 100:.1f}%")
        if not p3:
            reasons.append(f"Check 3 (baseband cleanliness): {c3_peak * 100:.3f}% "
                           f"(orders 2..{base_orders[-1]}) > tol {baseband_tol * 100:.1f}%")

        result = {
            'passed': passed,
            'reasons': reasons,
            'check1': {'amp_err': c1_amp, 'phase_err_deg': c1_ph, 'pass': p1},
            'check2': {'avg_rms_rel': c2, 'pass': p2},
            'check3': {'baseband_peak': c3_peak, 'baseband_thd': c3_thd, 'pass': p3},
            'per_second': {'amp_err': amp_err, 'phase_err_deg': ph_err,
                           'avg_rms_rel': avg_rms, 'baseband_peak': base_peak},
        }

        # Print + raise ONLY on failure; silent on success.
        if not passed:
            def tag(ok):
                return "PASS" if ok else "FAIL"

            print("=" * 60)
            print("Vs QUALITY CHECK   (worst of {} second(s))".format(P))
            print("=" * 60)
            print(f"[{tag(p1)}] 1. Fundamental match")
            print(f"        amplitude error : {c1_amp * 100:6.3f} %   (tol {amp_tol * 100:.1f} %)")
            print(f"        phase error     : {c1_ph:6.3f} deg (tol {phase_tol_deg:.1f} deg)")
            print(f"[{tag(p2)}] 2. Volt-second tracking (Tsw moving avg)")
            print(f"        RMS error / peak: {c2 * 100:6.3f} %   (tol {avg_rms_tol * 100:.1f} %)")
            print(f"[{tag(p3)}] 3. Baseband cleanliness (orders 2..{base_orders[-1]})")
            print(f"        largest harmonic: {c3_peak * 100:6.3f} %   (tol {baseband_tol * 100:.1f} %)")
            print(f"        baseband THD    : {c3_thd * 100:6.3f} %   (info only)")
            print("-" * 60)
            print("OVERALL: FAIL")
            print("=" * 60)
            if raise_on_fail:
                raise ValueError("Vs quality check FAILED - simulation stopped. Reason(s):\n"
                                 + "\n".join("  - " + r for r in reasons))

        return result

    @staticmethod
    def last_of_column(val, Profile_size):
        col = np.full(Profile_size, np.nan)
        col[-1] = val
        return col

    @staticmethod
    def create_simulation_folders(base="Results"):
        """
        Creates:
            Results/
                Simulation_N/        <- auto-incremented
                    Dataframes/
                    Figures/

        Returns:
            sim_dir, dataframes_dir, figures_dir
        """

        base_dir = Path(base)
        base_dir.mkdir(exist_ok=True)

        # --- detect existing Simulation_N folders ---
        existing = []
        for p in base_dir.iterdir():
            if p.is_dir() and p.name.startswith("Simulation_"):
                try:
                    existing.append(int(p.name.split("_")[1]))
                except (IndexError, ValueError):
                    pass

        next_n = max(existing) + 1 if existing else 1

        # --- create Simulation_N and its two subfolders ---
        sim_dir = base_dir / f"Simulation_{next_n}"
        sim_dir.mkdir(exist_ok=True)

        dataframes_dir = sim_dir / "Dataframes"
        figures_dir = sim_dir / "Figures"
        dataframes_dir.mkdir(exist_ok=True)
        figures_dir.mkdir(exist_ok=True)

        return sim_dir, dataframes_dir, figures_dir

    @staticmethod
    def distort_Vs_ref(Vs_ref, t, omega, harmonics=None, noise_level=0.0, seed=None):
        """
        Inject controllable harmonic distortion into the inverter voltage reference.

        Distorts Vs_ref by adding low-order harmonics (as a fraction of the
        fundamental amplitude) and/or random noise, so the resulting switched
        voltage Vs has a higher, controlled THD. Used to study how the LCL filter
        attenuates voltage distortion into the grid-side current I_L2.

        Parameters
        ----------
        Vs_ref : np.ndarray
            Clean phase-A voltage reference [V] (one fundamental block).
        t : np.ndarray
            Time vector [s], same length as Vs_ref.
        omega : float
            Fundamental angular frequency [rad/s] (2*pi*f).
        harmonics : dict, optional
            Mapping {harmonic_order : relative_amplitude}, where the amplitude is a
            fraction of the fundamental peak. Example: {5: 0.05, 7: 0.03} adds a 5th
            harmonic at 5% and a 7th at 3%. In a 3-wire system the 3rd is suppressed
            by the connection, so 5th/7th/11th/13th are the realistic choices.
            Default None = no harmonic injection.
        noise_level : float, optional
            Std-dev of additive Gaussian noise as a fraction of the fundamental peak.
            Default 0.0 = no noise.
        seed : int, optional
            RNG seed for reproducible noise. Default None.

        Returns
        -------
        Vs_ref_distorted : np.ndarray
            Distorted reference [V], same shape as Vs_ref.
        """
        Vs_ref = np.asarray(Vs_ref, dtype=float)
        t = np.asarray(t, dtype=float)
        if Vs_ref.shape != t.shape:
            raise ValueError("Vs_ref and t must have the same shape.")

        # fundamental peak, used as the reference amplitude for relative injection
        V1_peak = np.sqrt(2.0) * np.sqrt(np.mean(Vs_ref ** 2))
        if V1_peak == 0:
            return Vs_ref.copy()

        out = Vs_ref.copy()

        # --- harmonic injection ---
        if harmonics:
            for order, rel_amp in harmonics.items():
                if order <= 1:
                    raise ValueError(f"Harmonic order must be >= 2, got {order}.")
                out = out + rel_amp * V1_peak * np.sin(order * omega * t)

        # --- additive noise ---
        if noise_level > 0.0:
            rng = np.random.default_rng(seed)
            out = out + rng.normal(0.0, noise_level * V1_peak, size=out.shape)

        return out

    @staticmethod
    def compute_THD_Vs(Vs, Vs_ref, resolution_per_cycle, n_cycles=1, max_harmonic=None, printing=False):
        """
        Compute the THD of the switched inverter voltage Vs over the last n_cycles
        fundamental periods.

        The analysis window is selected by integer sample count, so every harmonic
        lands exactly on an FFT bin and single-bin extraction is leakage-free.
        Harmonic content is measured against the fundamental of Vs itself; Vs_ref is
        used only to report the fundamental tracking amplitude/phase for reference.

        Parameters
        ----------
        Vs : array                 Switched inverter voltage [V].
        Vs_ref : array             Reference (intended) inverter voltage [V].
        resolution_per_cycle : int Samples per fundamental cycle [-].
        n_cycles : int, optional   Number of trailing cycles to analyse. Default 1.
        max_harmonic : int, optional
            Highest harmonic order included in THD. Default None = up to Nyquist.
            For baseband distortion studies, cap this (e.g. 50) so the switching
            harmonics around fsw do not dominate the metric.
        printing : bool, optional  Print a summary. Default False.

        Returns
        -------
        THD_percent : float        THD of Vs [%].
        """

        Vs = np.asarray(Vs, dtype=float)
        Vs_ref = np.asarray(Vs_ref, dtype=float)

        # --- exact-integer-cycle window (last n_cycles periods) ---
        spc = int(round(resolution_per_cycle))  # samples per cycle
        win = n_cycles * spc
        if win > len(Vs):
            raise ValueError(
                f"Window of {win} samples exceeds signal length {len(Vs)}.")

        last = slice(-win, None)
        Vs_w = Vs[last]
        Vs_ref_w = Vs_ref[last]

        # ── RMS ───────────────────────────────────────────────────
        Vs_RMS = np.sqrt(np.mean(Vs_w ** 2))
        Vs_ref_RMS = np.sqrt(np.mean(Vs_ref_w ** 2))

        # ── FFT (DC removed) ──────────────────────────────────────
        N = len(Vs_w)
        fft_Vs = np.fft.rfft(Vs_w - np.mean(Vs_w))
        fft_ref = np.fft.rfft(Vs_ref_w - np.mean(Vs_ref_w))

        # fundamental sits at bin = n_cycles (n_cycles periods in the window)
        idx_f = n_cycles

        # RMS amplitude of fundamental: sqrt(2)|X|/N
        amp_Vs = np.sqrt(2) * np.abs(fft_Vs[idx_f]) / N
        amp_ref = np.sqrt(2) * np.abs(fft_ref[idx_f]) / N

        # ── THD of Vs (harmonics 2..max up to Nyquist) ────────────
        # harmonic h sits exactly on bin h*n_cycles
        nyq_order = (len(fft_Vs) - 1) // n_cycles
        top = nyq_order if max_harmonic is None else min(max_harmonic, nyq_order)
        h_orders = np.arange(2, top + 1)
        h_bins = h_orders * n_cycles

        P_harmonics = np.sum(np.abs(fft_Vs[h_bins]) ** 2)
        P_fundamental = np.abs(fft_Vs[idx_f]) ** 2
        THD = np.sqrt(P_harmonics / P_fundamental)
        THD_percent = THD * 100

        if printing:
            print("=" * 46)
            print(f"  Vs_ref RMS          : {Vs_ref_RMS:>10.4f}  V")
            print(f"  Vs     RMS          : {Vs_RMS:>10.4f}  V")
            print("-" * 46)
            print(f"  Fundamental amp ref : {amp_ref:>10.4f}  V (RMS)")
            print(f"  Fundamental amp Vs  : {amp_Vs:>10.4f}  V (RMS)")
            print(f"  Highest order incl. : {top:>10d}")
            print("-" * 46)
            print(f"  THD of Vs           : {THD_percent:>10.4f}  %")
            print("=" * 46)

        return THD_percent

    @staticmethod
    def compare_components(C, L1, L2, profile_index=-1, K_to_C=273.15):
        """
        Print a side-by-side comparison of the capacitor (C) and the two inductors (L1, L2).

        Each argument is a dict of already-computed results. Values may be scalars or
        per-second arrays; arrays are reduced to the second given by `profile_index`
        (default -1 = last second of the mission profile).

        Expected keys
        -------------
        C  : C, R_th, V_RMS, V_RMS_rated, I_RMS, P_total, T, T_rated, Lifetime
        L1 / L2 : L, N, lg, B_peak, B_max, Bsat, Ae, le, Ve, A_surface,
                  I_RMS, V_RMS, Rdc, N_parallel, A_wire, l_turn,
                  P_core, P_winding, P_total, R_th, T, T_rated,
                  Lifetime, Lifetime_consumed

        Missing keys print as a dash, so the function still works on partially-filled dicts.
        """

        LBL, UNT, COL, RAT = 24, 8, 15, 10
        line = "-" * (LBL + UNT + 2 * COL + RAT)

        def _val(x, profile_index=-1):
            if x is None:
                return None
            arr = np.atleast_1d(x)
            if arr.size == 1:
                return float(arr[0])
            return float(arr[profile_index])

        def v(d, key):
            return _val(d.get(key), profile_index)

        def row(label, unit, a, b, scale=1.0, offset=0.0, nfmt="{:.4f}", ratio=False):
            s1 = nfmt.format(a * scale + offset) if a is not None else "-"
            s2 = nfmt.format(b * scale + offset) if b is not None else "-"
            if ratio and a not in (None, 0) and b not in (None, 0):
                rr = "{:.2f}".format(a / b)
            else:
                rr = ""
            print(f"{label:<{LBL}}{unit:<{UNT}}{s1:>{COL}}{s2:>{COL}}{rr:>{RAT}}")

        # ------------------------------------------------------------------ #
        # Header
        # ------------------------------------------------------------------ #
        print("=" * len(line))
        print(f"COMPONENT COMPARISON   (mission-profile second index = {profile_index})")
        print("=" * len(line))

        # ------------------------------------------------------------------ #
        # Capacitor (standalone block)
        # ------------------------------------------------------------------ #
        print("\nCAPACITOR")
        print(line)
        C_C = v(C, "C")
        vrms, vrated = v(C, "V_RMS"), v(C, "V_RMS_rated")
        tc, trated = v(C, "T"), v(C, "T_rated")
        print(f"  Capacitance        : {C_C * 1e6:.4f} µF" if C_C is not None else "  Capacitance        : -")
        if v(C, "R_th") is not None:
            print(f"  Thermal resistance : {v(C, 'R_th'):.4f} K/W")
        if vrms is not None:
            ratio_txt = f"   (rated {vrated:.0f} V, ratio {vrms / vrated:.2f})" if vrated else ""
            print(f"  V_RMS              : {vrms:.2f} V{ratio_txt}")
        if v(C, "I_RMS") is not None:
            print(f"  I_RMS              : {v(C, 'I_RMS'):.2f} A")
        if v(C, "P_total") is not None:
            print(f"  P_total            : {v(C, 'P_total'):.4f} W")
        if tc is not None:
            margin = f"   (rated {trated - K_to_C:.0f} °C)" if trated else ""
            print(f"  T_hotspot          : {tc - K_to_C:.2f} °C{margin}")
        if v(C, "Lifetime") is not None:
            print(f"  Lifetime           : {v(C, 'Lifetime'):.2f} years")
        if v(C, "Lifetime_consumed_C") is not None:
            print(f"  Lifetime consumed  : {v(C, 'Lifetime_consumed_C'):.2f} %")



        # ------------------------------------------------------------------ #
        # Inductors L1 vs L2 (aligned columns)
        # ------------------------------------------------------------------ #
        print("\nINDUCTORS  —  L1 (inverter side)  vs  L2 (grid side)")
        print(line)
        print(f"{'Quantity':<{LBL}}{'Unit':<{UNT}}{'L1':>{COL}}{'L2':>{COL}}{'L1/L2':>{RAT}}")
        print(line)

        #print("[ Core geometry ]")
        #row("Inductance", "[µH]", v(L1, "L"), v(L2, "L"), scale=1e6, nfmt="{:.3f}", ratio=True)
        #row("Turns N", "[-]", v(L1, "N"), v(L2, "N"), nfmt="{:.0f}", ratio=True)
        #row("Air gap", "[mm]", v(L1, "lg"), v(L2, "lg"), scale=1e3, nfmt="{:.3f}", ratio=True)
        #row("B_peak", "[T]", v(L1, "B_peak"), v(L2, "B_peak"), nfmt="{:.4f}", ratio=True)
        # B_peak / Bsat as a percentage
        #bp1 = v(L1, "B_peak");
        #bs1 = v(L1, "Bsat")
        #bp2 = v(L2, "B_peak");
        #bs2 = v(L2, "Bsat")
        #pct1 = (bp1 / bs1) if (bp1 is not None and bs1) else None
        #pct2 = (bp2 / bs2) if (bp2 is not None and bs2) else None
        #row("B_peak / Bsat", "[%]", pct1, pct2, scale=100.0, nfmt="{:.2f}")
        #row("Ae (eff. area)", "[mm²]", v(L1, "Ae"), v(L2, "Ae"), scale=1e6, nfmt="{:.2f}", ratio=True)
        #row("le (path len)", "[mm]", v(L1, "le"), v(L2, "le"), scale=1e3, nfmt="{:.2f}", ratio=True)
        #row("Ve (volume)", "[cm³]", v(L1, "Ve"), v(L2, "Ve"), scale=1e6, nfmt="{:.2f}", ratio=True)
        #row("A_surface", "[cm²]", v(L1, "A_surface"), v(L2, "A_surface"), scale=1e4, nfmt="{:.2f}", ratio=True)

        #print("[ Winding ]")
        #row("I_RMS", "[A]", v(L1, "I_RMS"), v(L2, "I_RMS"), nfmt="{:.2f}", ratio=True)
        #row("V_RMS", "[V]", v(L1, "V_RMS"), v(L2, "V_RMS"), nfmt="{:.4f}", ratio=True)
        #row("Rdc", "[mΩ]", v(L1, "Rdc"), v(L2, "Rdc"), scale=1e3, nfmt="{:.4f}", ratio=True)
        #row("Parallel strands", "[-]", v(L1, "N_parallel"), v(L2, "N_parallel"), nfmt="{:.0f}", ratio=True)
        #row("A_wire actual", "[mm²]", v(L1, "A_wire"), v(L2, "A_wire"), scale=1e6, nfmt="{:.2f}", ratio=True)
        #row("Mean turn len", "[mm]", v(L1, "l_turn"), v(L2, "l_turn"), scale=1e3, nfmt="{:.2f}", ratio=True)

        #print("[ Power losses ]")
        #row("Core loss  P_c", "[W]", v(L1, "P_core"), v(L2, "P_core"), nfmt="{:.4f}", ratio=True)
        #row("Winding loss P_w", "[W]", v(L1, "P_winding"), v(L2, "P_winding"), nfmt="{:.4f}", ratio=True)
        #row("Total loss P_tot", "[W]", v(L1, "P_total"), v(L2, "P_total"), nfmt="{:.4f}", ratio=True)

        print("[ Thermal ]")
        #row("Thermal R_th", "[K/W]", v(L1, "R_th"), v(L2, "R_th"), nfmt="{:.4f}", ratio=True)
        row("Temperature", "[°C]", v(L1, "T"), v(L2, "T"), offset=-K_to_C, nfmt="{:.2f}")
        #row("Rated temp", "[°C]", v(L1, "T_rated"), v(L2, "T_rated"), offset=-K_to_C, nfmt="{:.2f}")
        # Margin = T_rated - T_operating
        #m1 = (v(L1, "T_rated") - v(L1, "T")) if (v(L1, "T_rated") is not None and v(L1, "T") is not None) else None
        #m2 = (v(L2, "T_rated") - v(L2, "T")) if (v(L2, "T_rated") is not None and v(L2, "T") is not None) else None
        #row("Margin below rated", "[K]", m1, m2, nfmt="{:.2f}")

        print("[ Lifetime ]")
        row("Lifetime", "[yr]", v(L1, "Lifetime"), v(L2, "Lifetime"), nfmt="{:.4f}", ratio=True)
        row("Lifetime consumed", "[%]", v(L1, "Lifetime_consumed"), v(L2, "Lifetime_consumed"))

        print("=" * len(line))



    @staticmethod
    def normal_distribution_function(value, frac_sigma, n_samples, rng):
        """Draw n_samples ~ N(value, (frac_sigma*|value|)^2)."""
        sigma = frac_sigma * abs(value)
        return rng.normal(value, sigma, n_samples)

    @staticmethod
    def build_lifetime_curves_samples(lifetime_curves, frac_sigma, number_of_samples, rng):
        """
        Build a list of `number_of_samples` perturbed copies of the lifetime-curves
        dictionary. Keys (voltage ratios) and T-arrays are kept fixed as the grid;
        only the L-values are randomized, each point ~ N(L, (frac_sigma*|L|)^2).

        Returns
        -------
        list of dict, length = number_of_samples
            Each element has the same structure as `lifetime_curves`.
        """
        samples = []
        for _ in range(number_of_samples):
            perturbed = {}
            for ratio, curve in lifetime_curves.items():
                T_arr = curve["T"]  # grid — keep fixed
                L_arr = curve["L"]
                sigma = frac_sigma * np.abs(L_arr)
                L_pert = rng.normal(L_arr, sigma)  # jitter each L point
                L_pert = np.clip(L_pert, 1e-6, None)  # keep physical (no neg lifetime)
                perturbed[ratio] = {"T": T_arr.copy(), "L": L_pert}
            samples.append(perturbed)
        return samples

    @staticmethod
    def equivalent_temperature(L_eq_years, T_rated, L_rated_hours, Ea, kb):
        """
        Constant temperature T_eq that reproduces the Miner-aggregated lifetime
        via the Arrhenius model:  L = L_rated * exp[(Ea/kb)(1/T - 1/T_rated)]
        Inverted: 1/T_eq = 1/T_rated + (kb/Ea) ln(L_eq / L_rated)
        Units: L_eq and L_rated MUST be in the same unit -> convert years to hours.
        """
        L_eq_hours = L_eq_years * 365 * 24
        inv_T = 1.0 / T_rated + (kb / Ea) * np.log(L_eq_hours / L_rated_hours)
        return 1.0 / inv_T

    @staticmethod
    def equivalent_temperature_capacitor(L_eq_years, V_C_RMS, V_C_RMS_Rated, t1, T1, A, n):
        """
        Constant temperature T_eq [K] that reproduces the Miner-aggregated capacitor
        lifetime via the TDK analytical model:
            t2 = t1 * exp((T1 - T2)/A) * (V1/V2)^n
        Inverted for T2 at a fixed operating voltage V2 = V_C_RMS:
            T2 = T1 - A * ln( t2 / ( t1 * (V1/V2)^n ) )

        Units:
            - L_eq_years -> converted to HOURS (t1 is in hours)
            - T1 passed in KELVIN, converted to °C internally (formula is in °C),
              result converted back to KELVIN.
        """
        # match the forward function's internal unit handling
        T1_C = T1 - 273.15  # K -> °C
        L_eq_hours = L_eq_years * 365 * 24  # years -> hours (t1 is hours)

        voltage_ratio = V_C_RMS_Rated / V_C_RMS  # V1 / V2  (fixed operating voltage)
        voltage_term = voltage_ratio ** n

        # invert the thermal term
        T2_C = T1_C - A * np.log(L_eq_hours / (t1 * voltage_term))

        return T2_C + 273.15  # °C -> K

    @staticmethod
    def compute_THD_v2(Signal, resolution_per_cycle, n_cycles):
        """
        Compute the THD of a signal over the last n_cycles fundamental periods.

        Definition
        ----------
            THD = harmonic_RMS / fundamental_RMS
        where the fundamental is the single exact FFT bin at the fundamental
        frequency and the harmonic content is ALL remaining (non-DC, non-
        fundamental) energy, obtained by Parseval:
            harmonic_RMS = sqrt(total_RMS^2 - DC^2 - fundamental_RMS^2)

        This captures every non-fundamental component, including the switching-
        frequency sidebands, without relying on an explicit harmonic-bin list.
        The integer-cycle window guarantees the fundamental lands exactly on a
        bin, so the single-bin extraction is leakage-free.

        Parameters
        ----------
        Signal : array
            Time-domain signal (e.g. grid-side current) [A].
        resolution_per_cycle : int
            Samples per fundamental cycle [-].
        n_cycles : int, optional
            Number of trailing fundamental cycles to analyse. Default 1.

        Returns
        -------
        THD_percent : float
            THD of Signal [%].
        """
        x = np.asarray(Signal, dtype=float)
        x = x[np.isfinite(x)]

        spc = int(round(resolution_per_cycle))
        win = n_cycles * spc
        if win > len(x):
            raise ValueError(
                f"Window of {win} samples exceeds signal length {len(x)}.")

        x = x[-win:]
        N = len(x)

        # total and DC
        total_rms = np.sqrt(np.mean(x ** 2))
        dc = np.mean(x)

        # fundamental sits exactly on bin k = n_cycles
        k = n_cycles
        n = np.arange(N)
        basis = np.exp(-1j * 2 * np.pi * k * n / N)
        X = np.dot(x, basis)
        fund_rms = (2.0 * np.abs(X) / N) / np.sqrt(2)

        # harmonic content = everything except DC and fundamental (Parseval)
        harmonic_rms = np.sqrt(max(total_rms ** 2 - dc ** 2 - fund_rms ** 2, 0.0))

        THD_percent = 100.0 * harmonic_rms / fund_rms
        return THD_percent

    @staticmethod
    def spectral_split_IL2_Ig(I_L2, Ig_ref, resolution_per_cycle, n_cycles):
        """
        Fundamental / harmonic spectral split of I_L2 and Ig_ref over the last
        n_cycles fundamental periods.

        For each signal, over the trailing window:
          - fundamental_RMS = RMS of the single exact FFT bin at the fundamental.
          - harmonic_RMS    = all remaining (non-DC, non-fundamental) energy,
                              via Parseval: sqrt(total^2 - DC^2 - fund^2).

        The bin index is derived from the actual window length (k = round(N/spc)),
        so the extraction stays leakage-free even if the trailing window is not
        perfectly period-aligned. Use n_cycles large enough (e.g. n_cycles = f)
        that the window spans many whole periods.

        Parameters
        ----------
        I_L2, Ig_ref : array
            Grid-side current and reference grid current [A].
        resolution_per_cycle : int
            Samples per fundamental cycle [-].
        n_cycles : int
            Number of trailing fundamental cycles to analyse.

        Returns
        -------
        dict with keys:
            IL2_fund, IL2_hrm, IL2_total,
            Ig_fund,  Ig_hrm,  Ig_total,
            THD_IL2_self  : harmonic_RMS(I_L2) / fundamental_RMS(I_L2)  [%]
            THD_IL2_ref   : harmonic_RMS(I_L2) / fundamental_RMS(Ig_ref) [%]
        """
        spc = int(round(resolution_per_cycle))
        win = n_cycles * spc

        def _split(a):
            a = np.asarray(a, dtype=float)
            a = a[np.isfinite(a)]
            if win > len(a):
                raise ValueError(
                    f"Window of {win} samples exceeds signal length {len(a)}.")
            a = a[-win:]
            N = len(a)

            k = int(round(N / spc))  # cycles in window = fund bin
            n = np.arange(N)
            X = np.dot(a, np.exp(-1j * 2 * np.pi * k * n / N))
            fund_rms = (2.0 * np.abs(X) / N) / np.sqrt(2)

            total_rms = np.sqrt(np.mean(a ** 2))
            dc = np.mean(a)
            hrm_rms = np.sqrt(max(total_rms ** 2 - dc ** 2 - fund_rms ** 2, 0.0))

            return fund_rms, hrm_rms, total_rms

        IL2_fund, IL2_hrm, IL2_total = _split(I_L2)
        Ig_fund, Ig_hrm, Ig_total = _split(Ig_ref)

        return {
            "IL2_fund": IL2_fund,
            "IL2_hrm": IL2_hrm,
            "IL2_total": IL2_total,
            "Ig_fund": Ig_fund,
            "Ig_hrm": Ig_hrm,
            "Ig_total": Ig_total,
            "THD_IL2_self": 100.0 * IL2_hrm / IL2_fund,
            "THD_IL2_ref": 100.0 * IL2_hrm / Ig_fund,
        }

    @staticmethod
    def compute_I_L2_RMS_per_harmonic(I_L2, f, fsw,resolution_per_cycle, Profile_size):

        """
        Decompose the grid-side current I_L2 into RMS values per harmonic order
        for each second of the mission profile, for use in the transformer
        harmonic loss factor calculation of IEEE C57.110.

        Harmonic orders are the physically significant harmonics of a
        three-phase grid-connected inverter:
        - Low-order characteristic harmonics: 1, 5, 7, 11, 13, 17, 19
        - First switching band:               mf-2 ... mf+2
        - Second switching band:              2*mf-2 ... 2*mf+2

        Parameters
        ----------
        I_L2 : np.ndarray
            Time-domain grid-side current signal,
            length = Profile_size * resolution_per_cycle * f
        f : float
            Fundamental frequency [Hz]
        fsw : float
            Inverter switching frequency [Hz]
        resolution_per_cycle : int
            Number of samples per fundamental cycle
        Profile_size : int
            Number of seconds in the mission profile

        Returns
        -------
        I_L2_RMS_harmonics : np.ndarray
            Shape: (Profile_size, n_harmonics)
            I_L2_RMS_harmonics[i, n] = RMS current at harmonic order n during second i
        harmonic_orders : np.ndarray
            Shape: (n_harmonics,)
            The harmonic order h corresponding to each column. Required by the
            harmonic loss factor, since the orders are not contiguous.
        """

        # ----------------------------------------#
        # Harmonic orders
        # ----------------------------------------#
        mf = int(round(fsw / f))  # frequency modulation ratio

        low_order_harmonics = [1, 5, 7, 11, 13, 17, 19]
        switching_band_1 = list(range(mf - 2, mf + 3))
        switching_band_2 = list(range(2 * mf - 2, 2 * mf + 3))

        harmonic_orders = np.array(low_order_harmonics
                                   + switching_band_1
                                   + switching_band_2)

        # ----------------------------------------#
        # FFT, all seconds at once
        # ----------------------------------------#
        samples_per_second = int(resolution_per_cycle * f)
        N = samples_per_second

        I_matrix = np.asarray(I_L2, dtype=float).reshape(Profile_size, N)

        fft_vals = np.fft.rfft(I_matrix, axis=1)
        fft_freq = np.fft.rfftfreq(N, d=1.0 / samples_per_second)

        target_freqs = harmonic_orders * f

        bin_indices = np.argmin(
            np.abs(fft_freq[:, np.newaxis] - target_freqs[np.newaxis, :]), axis=0)

        amplitudes_peak = (2 * np.abs(fft_vals[:, bin_indices])) / N
        I_L2_RMS_harmonics = amplitudes_peak / np.sqrt(2)

        return I_L2_RMS_harmonics, harmonic_orders

    @staticmethod
    def compute_I_C_RMS_per_harmonic(I_C, f, fsw, resolution_per_cycle, Profile_size):
        """
        Decompose the grid-side current I_C into RMS values per harmonic order
        for each second of the mission profile, for use in the transformer
        harmonic loss factor calculation of IEEE C57.110.

        Harmonic orders are the physically significant harmonics of a
        three-phase grid-connected inverter:
        - Low-order characteristic harmonics: 1, 5, 7, 11, 13, 17, 19
        - First switching band:               mf-2 ... mf+2
        - Second switching band:              2*mf-2 ... 2*mf+2

        Parameters
        ----------
        I_C : np.ndarray
            Time-domain grid-side current signal,
            length = Profile_size * resolution_per_cycle * f
        f : float
            Fundamental frequency [Hz]
        fsw : float
            Inverter switching frequency [Hz]
        resolution_per_cycle : int
            Number of samples per fundamental cycle
        Profile_size : int
            Number of seconds in the mission profile

        Returns
        -------
        I_C_RMS_harmonics : np.ndarray
            Shape: (Profile_size, n_harmonics)
            I_C_RMS_harmonics[i, n] = RMS current at harmonic order n during second i
        harmonic_orders : np.ndarray
            Shape: (n_harmonics,)
            The harmonic order h corresponding to each column. Required by the
            harmonic loss factor, since the orders are not contiguous.
        """

        # ----------------------------------------#
        # Harmonic orders
        # ----------------------------------------#
        mf = int(round(fsw / f))  # frequency modulation ratio

        low_order_harmonics = [1, 5, 7, 11, 13, 17, 19]
        switching_band_1 = list(range(mf - 2, mf + 3))
        switching_band_2 = list(range(2 * mf - 2, 2 * mf + 3))

        harmonic_orders = np.array(low_order_harmonics + switching_band_1 + switching_band_2)

        # ----------------------------------------#
        # FFT, all seconds at once
        # ----------------------------------------#
        samples_per_second = int(resolution_per_cycle * f)
        N = samples_per_second

        I_matrix = np.asarray(I_C, dtype=float).reshape(Profile_size, N)

        fft_vals = np.fft.rfft(I_matrix, axis=1)
        fft_freq = np.fft.rfftfreq(N, d=1.0 / samples_per_second)

        target_freqs = harmonic_orders * f

        bin_indices = np.argmin(
            np.abs(fft_freq[:, np.newaxis] - target_freqs[np.newaxis, :]), axis=0)

        amplitudes_peak = (2 * np.abs(fft_vals[:, bin_indices])) / N
        I_C_RMS_harmonics = amplitudes_peak / np.sqrt(2)

        return I_C_RMS_harmonics, harmonic_orders

    @staticmethod
    def compute_RMS_per_harmonic(I, f, resolution_per_cycle, Profile_size, h_max):

        """
        Decompose a current signal into RMS values for every harmonic order
        from 1 to h_max, for each second of the mission profile. Applicable to
        any branch current of the LCL filter, namely I_L1, I_C and I_L2.

        The full spectrum is retained rather than a selected set of orders, so
        that quantities such as the harmonic loss factor of IEEE C57.110 can be
        evaluated over any chosen upper limit without re-running the simulation.

        Parameters
        ----------
        I : np.ndarray
            Time-domain current signal,
            length = Profile_size * resolution_per_cycle * f
        f : float
            Fundamental frequency [Hz]
        resolution_per_cycle : int
            Number of samples per fundamental cycle
        Profile_size : int
            Number of seconds in the mission profile
        h_max : int
            Highest harmonic order retained. Must satisfy
            h_max <= resolution_per_cycle / 2 to respect Nyquist.

        Returns
        -------
        I_RMS_harmonics : np.ndarray
            Shape: (Profile_size, h_max)
            I_RMS_harmonics[i, n] = RMS current at harmonic order n+1
            during second i
        harmonic_orders : np.ndarray
            Shape: (h_max,), the harmonic orders 1 ... h_max
        """

        samples_per_second = int(resolution_per_cycle * f)
        N = samples_per_second

        if h_max > resolution_per_cycle // 2:
            raise ValueError(
                f"h_max = {h_max} exceeds the Nyquist limit of "
                f"{resolution_per_cycle // 2} for the given resolution.")

        # ----------------------------------------#
        # FFT, all seconds at once
        # ----------------------------------------#
        I_matrix = np.asarray(I, dtype=float).reshape(Profile_size, N)

        fft_vals = np.fft.rfft(I_matrix, axis=1)

        # Over a one second window the signal completes f cycles, so harmonic
        # order h lands in FFT bin h * f.
        harmonic_orders = np.arange(1, h_max + 1)
        bin_indices = harmonic_orders * int(f)

        amplitudes_peak = (2 * np.abs(fft_vals[:, bin_indices])) / N
        I_RMS_harmonics = amplitudes_peak / np.sqrt(2)

        return I_RMS_harmonics, harmonic_orders