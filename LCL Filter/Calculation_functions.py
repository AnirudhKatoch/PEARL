import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import cont2discrete
from scipy.interpolate import interp1d

plt.rcParams.update({"font.size": 15, "font.family": "Times New Roman", "axes.labelsize": 15, "axes.titlesize": 15,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15})

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
    def validate_ac_rms_voltage_limit(Vdc_RMS, M, inverter_phases, modulation_scheme,
                                      single_phase_inverter_topology, Vg_RMS):
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
    def compute_THD(t, signal, f, Location, IL2_THD_plotting, max_plot_harmonic, cycle_start=None):

        """
        Parameters
        ----------
        t : array
            Time vector [s]
        signal : array
            Signal waveform to be analyzed, usually grid-side current I_L2 [A]
        f : float
            Fundamental frequency [Hz]
        Location : str
            File path where the harmonic spectrum plot is saved
        IL2_THD_plotting : bool
            If True, plot and save the harmonic RMS spectrum
        max_plot_harmonic : int
            Maximum harmonic order shown in the plot
        cycle_start : float, optional
            Start time of the cycle used for THD calculation [s].
            If None, the last full fundamental cycle is used.

        Returns
        -------
        THD : float
            Total Harmonic Distortion in per-unit [-]
        THD_percent : float
            Total Harmonic Distortion in percent [%]
        I1_rms : float
            RMS value of the fundamental component [A]
        freqs : array
            FFT frequency vector [Hz]
        mag_rms : array
            RMS magnitude spectrum of the signal
        """

        t = np.asarray(t)
        signal = np.asarray(signal)

        T = 1 / f
        dt = t[1] - t[0]

        if cycle_start is None:
            cycle_start = t[-1] - T

        cycle_end = cycle_start + T

        mask = (t >= cycle_start) & (t < cycle_end)

        y = signal[mask]

        # Remove DC offset
        y = y - np.mean(y)

        N = len(y)

        # FFT
        Y = np.fft.rfft(y)

        # RMS spectrum
        mag_rms = np.abs(Y) * np.sqrt(2) / N
        mag_rms[0] = np.abs(Y[0]) / N
        freqs = np.fft.rfftfreq(N, d=dt)

        # ----------------------------------------
        # Harmonic extraction WITHOUT for-loops
        # ----------------------------------------
        max_harmonic = int(freqs[-1] // f)
        harmonic_orders = np.arange(1, max_harmonic + 1)
        harmonic_freqs = harmonic_orders * f

        # Find nearest FFT bin for each harmonic
        indices = np.abs(freqs[:, None] - harmonic_freqs).argmin(axis=0)

        # Extract harmonic RMS values
        harmonic_rms = mag_rms[indices]

        # Fundamental RMS
        I1_rms = harmonic_rms[0]

        # THD calculation
        harmonic_rms_sq = np.sum(harmonic_rms[1:] ** 2)
        THD = np.sqrt(harmonic_rms_sq) / I1_rms
        THD_percent = THD * 100

        # ----------------------------------------
        # Plotting
        # ----------------------------------------

        harmonic_orders_plot = harmonic_orders[:max_plot_harmonic]
        harmonic_rms_plot = harmonic_rms[:max_plot_harmonic]

        if IL2_THD_plotting:
            plt.figure(figsize=(6.4, 4.8))
            plt.bar(harmonic_orders_plot, harmonic_rms_plot)
            plt.xlabel("Harmonic Order")
            plt.ylabel("RMS Current [A]")
            plt.title(f"THD = {THD_percent:.2f}%")
            plt.grid(True)
            plt.savefig(Location)
            plt.close()

        return THD, THD_percent, I1_rms, freqs, mag_rms

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
    def validate_pwm_pulse_amplitude_profile(Vdc_RMS, inverter_phases,
                                             single_phase_inverter_topology,
                                             waveform_voltage_definition, Vo):
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
    def compute_Vs_ref_phasor(t, f, Ig_RMS, Vg_RMS, phase_shift,
                              L1, L2, C, R1, R2, R3,
                              Profile_size, samples_per_second):
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

################################################################
# Extra functions
##################################################################

    @staticmethod
    def compare_I_ref_and_I_L2(t, I_L2, Ig_ref, f, dt, resolution_per_cycle, save_path):
        """
        Compare the reference grid current Ig_ref against the simulated LCL
        output current I_L2. Prints a summary of key metrics and saves a plot.

        Parameters
        ----------
        t : array
            Time vector [s]
        I_L2 : array
            Simulated grid-side inductor current [A]
        Ig_ref : array
            Reference grid current waveform [A]
        f : float
            Fundamental frequency [Hz]
        dt : float
            Simulation time step [s]
        resolution_per_cycle : int
            Number of samples per fundamental cycle
        save_path : str, optional
            File path for the saved figure
        """

        last = slice(-resolution_per_cycle, None)

        I_L2_w = I_L2[last]
        Ig_ref_w = Ig_ref[last]
        t_w = t[last]

        # ── 1. RMS ────────────────────────────────────────────────────────────────
        I_L2_RMS = np.sqrt(np.mean(I_L2_w ** 2))
        Ig_ref_RMS = np.sqrt(np.mean(Ig_ref_w ** 2))

        # ── 2. Tracking error ─────────────────────────────────────────────────────
        error = I_L2_w - Ig_ref_w
        error_RMS = np.sqrt(np.mean(error ** 2))
        error_peak = np.max(np.abs(error))
        NRMSE = error_RMS / Ig_ref_RMS * 100

        # ── 3. Fundamental amplitude and phase (via FFT) ──────────────────────────
        N = len(I_L2_w)
        freqs = np.fft.rfftfreq(N, d=dt)
        idx_f = np.argmin(np.abs(freqs - f))

        fft_L2 = np.fft.rfft(I_L2_w)
        fft_ref = np.fft.rfft(Ig_ref_w)

        amp_L2 = 2 * np.abs(fft_L2[idx_f]) / N * np.sqrt(2)
        amp_ref = 2 * np.abs(fft_ref[idx_f]) / N * np.sqrt(2)

        phase_L2 = np.angle(fft_L2[idx_f], deg=True)
        phase_ref = np.angle(fft_ref[idx_f], deg=True)
        phase_err = phase_L2 - phase_ref

        # ── 4. THD of I_L2 ────────────────────────────────────────────────────────
        harmonics = np.arange(2, 21) * f
        harmonic_idxs = [np.argmin(np.abs(freqs - h)) for h in harmonics]
        P_harmonics = np.sum([np.abs(fft_L2[i]) ** 2 for i in harmonic_idxs])
        P_fundamental = np.abs(fft_L2[idx_f]) ** 2
        THD = np.sqrt(P_harmonics / P_fundamental) * 100

        # ── Print summary ─────────────────────────────────────────────────────────
        print("=" * 46)
        print(f"  Ig_ref RMS          : {Ig_ref_RMS:>10.4f}  A")
        print(f"  I_L2   RMS          : {I_L2_RMS:>10.4f}  A")
        print("-" * 46)
        print(f"  Tracking error RMS  : {error_RMS:>10.4f}  A")
        print(f"  Tracking error peak : {error_peak:>10.4f}  A")
        print(f"  NRMSE               : {NRMSE:>10.4f}  %")
        print("-" * 46)
        print(f"  Fundamental amp ref : {amp_ref:>10.4f}  A (RMS)")
        print(f"  Fundamental amp L2  : {amp_L2:>10.4f}  A (RMS)")
        print(f"  Phase ref           : {phase_ref:>10.4f}  deg")
        print(f"  Phase I_L2          : {phase_L2:>10.4f}  deg")
        print(f"  Phase error         : {phase_err:>10.4f}  deg")
        print("-" * 46)
        print(f"  THD of I_L2         : {THD:>10.4f}  %")
        print("=" * 46)

        # ── Plot ──────────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        axes[0].plot(t_w, Ig_ref_w, label='Ig_ref', linewidth=1.5)
        axes[0].plot(t_w, I_L2_w, label='I_L2', linewidth=1.0, linestyle='--')
        axes[0].set_ylabel("Current [A]")
        axes[0].legend()
        axes[0].set_title("Waveform comparison")

        axes[1].plot(t_w, error, color='red', linewidth=1.0, label='Error (I_L2 − Ig_ref)')
        axes[1].axhline(0, color='k', linewidth=0.5)
        axes[1].set_ylabel("Error [A]")
        axes[1].set_xlabel("Time [s]")
        axes[1].legend()
        axes[1].set_title(f"Tracking error  |  NRMSE = {NRMSE:.3f}%  |  THD = {THD:.3f}%")

        plt.xlim(t_w[0], t_w[-1])
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

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
    def Capacitor_total_power_losses(fsw, f, I_C_RMS_harmonics, tan_delta_0, C, Rs):

        '''
        Compute the total power dissipation in the LCL filter capacitor across all relevant harmonics.

        Parameters
        ----------
        fsw : float
            Inverter switching frequency [Hz].
        f : float
            Fundamental grid frequency [Hz].
        I_C_RMS_harmonics : ndarray, shape (Profile_size, num_harmonics)
            RMS current through the capacitor at each harmonic frequency, for each second of the mission profile. Rows = mission-profile seconds, columns = harmonic orders.
        tan_delta_0 : float
            Dielectric dissipation factor of the polypropylene film [-].
        C : float
            Capacitance of the filter capacitor [F].
        Rs : float
            Series resistance of the capacitor [Ω].

        Returns
        -------
        P_total_C : ndarray, shape (Profile_size,)
            Total power dissipation in the capacitor [W] for each second of the mission profile.


        '''

        mf = int(round(fsw / f))  # Frequency modulation ratio: e.g. 200 for fsw=10kHz, f=50Hz

        # Define the harmonic orders to include in the loss calculation
        low_order_harmonics = [1, 5, 7, 11, 13, 17, 19]  # Dominant low-order grid harmonics
        switching_band_1 = list(range(mf - 2, mf + 3))  # First switching-frequency sideband, e.g. [198,199,200,201,202]
        switching_band_2 = list(
            range(2 * mf - 2, 2 * mf + 3))  # Second switching-frequency sideband, e.g. [398,399,400,401,402]
        harmonic_orders = np.array(low_order_harmonics + switching_band_1 + switching_band_2)

        f_i = harmonic_orders * f  # Absolute frequency of each harmonic [Hz]

        # Dielectric losses at each harmonic: PD(fi) = I(fi)^2 * tan_delta_0 / (2*pi*fi*C)
        P_D = I_C_RMS_harmonics ** 2 * tan_delta_0 / (2 * np.pi * f_i * C)

        # Resistive losses at each harmonic: PR(fi) = I(fi)^2 * Rs
        P_R = I_C_RMS_harmonics ** 2 * Rs

        P_total_per_harmonic = P_D + P_R  # Total loss contribution per harmonic [W], shape (Profile_size, num_harmonics)
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
    def Capacitor_voltage_RMS(V_C, resolution_per_cycle, f):
        """
        Compute the RMS voltage across the LCL filter capacitor for each second of the mission profile.

        Parameters
        ----------
        V_C : ndarray, shape (Profile_size * samples_per_second,)
            Instantaneous capacitor voltage waveform over the full mission profile [V].
        resolution_per_cycle : int
            Number of simulation samples per fundamental AC cycle [-].
        f : float
            Fundamental grid frequency [Hz].

        Returns
        -------
        V_C_RMS : ndarray, shape (Profile_size,)
            RMS voltage across the capacitor for each mission-profile second [V].

        """

        samples_per_second = resolution_per_cycle * f  # Number of simulation samples in one second
        samples_per_second = int(samples_per_second)

        Profile_size = len(V_C) // samples_per_second  # Recover the number of mission-profile seconds

        V_C_reshaped = V_C.reshape(Profile_size,
                                   samples_per_second)  # Reshape into (Profile_size, samples_per_second) so each row = one second of data

        V_C_RMS = np.sqrt(
            np.mean(V_C_reshaped ** 2, axis=1))  # Compute RMS over each row (each mission-profile second) [V]

        return V_C_RMS

    @staticmethod
    def validate_capacitor_operating_limits(T_C, V_C_RMS, V_C, T_C_Rated, V_C_RMS_Rated, V_C_Peak_Rated):
        """
        Validate that the LCL filter capacitor operates within its rated limits at all times.


        Parameters
        ----------
        T_C : ndarray, shape (Profile_size,)
            Capacitor hotspot temperature for each mission-profile second [K].
        V_C_RMS : ndarray, shape (Profile_size,)
            RMS voltage across the capacitor for each mission-profile second [V].
        V_C : ndarray, shape (Profile_size * samples_per_second,)
            Instantaneous capacitor voltage over the full mission profile [V].
        T_C_Rated : float, optional
            Maximum allowable capacitor hotspot temperature [K]. Default: 273 + 85 = 358 K.
        V_C_RMS_Rated : float, optional
            Maximum allowable RMS voltage across the capacitor [V]. Default: 530 V.
        V_C_Peak_Rated : float, optional
            Maximum allowable instantaneous peak voltage across the capacitor [V]. Default: 750 V.

        Returns
        -------
        None
            Returns nothing if all conditions are satisfied.

        Raises
        ------
        ValueError
            If any of the three operating limits are exceeded, with a detailed message
            identifying the violated condition, the worst-case value, and the mission-profile
            second or sample index where the violation occurs.
        """

        errors = []

        # ----------------------------------------#
        # Check 1: Hotspot temperature limit
        # ----------------------------------------#

        if np.any(T_C > T_C_Rated):
            violated_seconds = np.where(T_C > T_C_Rated)[0] + 1  # 1-indexed mission-profile seconds for readability
            worst_index = np.argmax(T_C)
            worst_value = T_C[worst_index]
            errors.append(
                f"\nCondition 1 FAILED: Capacitor hotspot temperature exceeds rated limit.\n"
                f"  Limit              : {T_C_Rated} K ({T_C_Rated - 273:.0f} °C)\n"
                f"  Worst value        : {worst_value:.2f} K ({worst_value - 273:.2f} °C) "
                f"at mission-profile second {worst_index + 1}\n"
                f"  Exceeded at second : {violated_seconds.tolist()}\n")

        # ----------------------------------------#
        # Check 2: RMS voltage limit
        # ----------------------------------------#

        if np.any(V_C_RMS > V_C_RMS_Rated):
            violated_seconds = np.where(V_C_RMS > V_C_RMS_Rated)[
                                   0] + 1  # 1-indexed mission-profile seconds for readability
            worst_index = np.argmax(V_C_RMS)
            worst_value = V_C_RMS[worst_index]
            errors.append(
                f"\nCondition 2 FAILED: Capacitor RMS voltage exceeds rated limit.\n"
                f"  Limit              : {V_C_RMS_Rated} V\n"
                f"  Worst value        : {worst_value:.2f} V "
                f"at mission-profile second {worst_index + 1}\n"
                f"  Exceeded at second : {violated_seconds.tolist()}\n")

        # ----------------------------------------#
        # Check 3: Instantaneous peak voltage limit
        # ----------------------------------------#

        if np.any(np.abs(V_C) > V_C_Peak_Rated):
            violated_samples = np.where(np.abs(V_C) > V_C_Peak_Rated)[0]  # Sample indices where the limit is exceeded
            worst_index = np.argmax(np.abs(V_C))
            worst_value = V_C[worst_index]
            errors.append(
                f"\nCondition 3 FAILED: Capacitor instantaneous voltage exceeds rated peak limit.\n"
                f"  Limit              : ±{V_C_Peak_Rated} V\n"
                f"  Worst value        : {worst_value:.2f} V at sample index {worst_index}\n"
                f"  Number of violated samples : {len(violated_samples)}\n")

        # ----------------------------------------#
        # Raise all errors together if any occurred
        # ----------------------------------------#

        if errors:
            raise ValueError(
                "\n" + "=" * 60 +
                "\nCAPACITOR OPERATING LIMIT VIOLATION DETECTED" +
                "\n" + "=" * 60 +
                "".join(errors) +
                "\n" + "=" * 60)

    @staticmethod
    def Capacitor_lifetime(T_C, V_C_RMS, V_C_RMS_Rated, lifetime_curves):
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

########################################################################################################################
# If you have time in the end make the PMW better which is basically production of Vs from Vs_ref
########################################################################################################################

    @staticmethod
    def Sinusoidal_Pulse_Width_Modulation_One_Phase(P_RMS, t, Vo, Vs_ref, Tsw):

        """
        Generate single-phase sinusoidal PWM (SPWM) voltage waveform.

        Parameters
        ----------
        P_RMS : array
        RMS Active Power [s]
        t : array
        Time vector [s]
        Vo : array
        PWM pulse amplitude [V]
        Tsw : float
        Switching period [s]
        Vs_ref : array
        Reference voltage for inverter PMW output[V]

        Returns
        -------
        V_s : array
        PWM output voltage waveform [V]
        """

        t_profile = np.arange(len(P_RMS))
        Vo_t = np.interp(t, t_profile, Vo)
        m_ref = Vs_ref / Vo_t

        def v_carrier(t, Tsw):
            tau = (t % Tsw) / Tsw
            return 4.0 * np.abs(tau - 0.5) - 1.0

        carrier = v_carrier(t, Tsw)

        Vs = np.where(m_ref >= carrier, Vo_t, -Vo_t)

        return Vs

    @staticmethod
    def Sinusoidal_Pulse_Width_Modulation_One_Phase_updated(P_RMS, t, Vo, Vs_ref, Tsw, f):

        """
        Generate the effective phase-A voltage of a three-phase SPWM inverter.

        This function internally creates three 120-degree shifted PWM pole voltages,
        removes the common-mode voltage, and returns the phase-A voltage seen by
        the LCL filter.

        Parameters
        ----------
        P_RMS : array
            RMS active power profile. Used to define the mission-profile time axis
            for interpolating Vo.

        t : array
            Time vector [s].

        Vo : array
            PWM pole-voltage amplitude profile [V], usually Vdc/2.

        Tsw : float
            Switching period [s].

        Vs_ref : array
            Phase-A reference voltage for inverter PWM output [V].

        f : float
            Fundamental frequency [Hz].

        Returns
        -------
        V_s : array
            Effective phase-A inverter voltage waveform after common-mode removal [V].
        """

        from scipy.signal import hilbert

        t = np.asarray(t)
        Vs_ref = np.asarray(Vs_ref)

        if len(t) != len(Vs_ref):
            raise ValueError("t and Vs_ref must have the same length.")

        t_profile = np.arange(len(P_RMS))
        Vo_t = np.interp(t, t_profile, Vo)

        if np.any(Vo_t <= 0):
            raise ValueError("All interpolated Vo values must be positive.")

        # ----------------------------------------#
        # Problem 1 fix: proper 120° phase shifts
        # using Hilbert transform so m_b, m_c carry
        # the same harmonic content as m_a
        # ----------------------------------------#
        analytic_signal = hilbert(Vs_ref)
        Vs_ref_b = np.real(analytic_signal * np.exp(-1j * 2 * np.pi / 3))
        Vs_ref_c = np.real(analytic_signal * np.exp(+1j * 2 * np.pi / 3))

        m_a = Vs_ref / Vo_t
        m_b = Vs_ref_b / Vo_t
        m_c = Vs_ref_c / Vo_t

        # ----------------------------------------#
        # Problem 2 fix: min-max zero-sequence
        # injection (SVM equivalent)
        # ----------------------------------------#
        m_stack = np.vstack([m_a, m_b, m_c])
        v_zs = -0.5 * (np.max(m_stack, axis=0) + np.min(m_stack, axis=0))

        m_a = m_a + v_zs
        m_b = m_b + v_zs
        m_c = m_c + v_zs

        if np.any(np.abs(m_a) > 1):
            raise ValueError("Phase-A modulation reference exceeds [-1, 1] after injection.")

        # ----------------------------------------#
        # Carrier and switching
        # ----------------------------------------#
        def v_carrier(t, Tsw):
            tau = (t % Tsw) / Tsw
            return 4.0 * np.abs(tau - 0.5) - 1.0

        carrier = v_carrier(t, Tsw)

        Vao = np.where(m_a >= carrier, Vo_t, -Vo_t)
        Vbo = np.where(m_b >= carrier, Vo_t, -Vo_t)
        Vco = np.where(m_c >= carrier, Vo_t, -Vo_t)

        V_common = (Vao + Vbo + Vco) / 3.0
        V_s = Vao - V_common

        return V_s

    @staticmethod
    def plot_LCL_signals(t, V_L1, I_L1, V_C, I_C, V_L2, I_L2,
                         resolution_per_cycle,
                         save_path):
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