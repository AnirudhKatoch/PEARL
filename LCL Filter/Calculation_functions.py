import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import cont2discrete


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
    def validate_ac_rms_voltage_limit(Vdc_rated, M, inverter_phases, modulation_scheme,single_phase_inverter_topology, Vg_RMS):

        """
        Validate that the requested inverter AC RMS output voltage does not exceed
        the theoretical maximum achievable RMS voltage for the selected inverter
        configuration.

        Parameters
        ----------
        Vdc_rated : float
            Rated DC bus voltage [V]

        M : float
            Modulation index [-]

        inverter_phases : {1, 3}
            Number of inverter phases

        modulation_scheme : {"spwm", "svm"}
            PWM modulation scheme

        single_phase_inverter_topology : {"half", "full"}, optional
            Required when inverter_phases == 1

        Vg_RMS : array_like
            Requested inverter AC RMS output voltage profile [V]

        """

        if Vdc_rated <= 0:
            raise ValueError("Vdc_rated must be positive.")

        if np.any(Vg_RMS < 0):
            raise ValueError("M must be non-negative.")

        if inverter_phases not in (1, 3):
            raise ValueError("inverter_phases must be 1 or 3.")

        if modulation_scheme not in ("spwm", "svm"):
            raise ValueError("modulation_scheme must be 'spwm' or 'svm'.")
        
        if modulation_scheme == "svm":
            raise ValueError(f"Current framework only supports 'spwm' modulation. ")

        if inverter_phases == 1:

            if single_phase_inverter_topology not in ("half", "full"):
                raise ValueError("For single-phase inverter, "
                    "single_phase_inverter_topology must be 'half' or 'full'.")

            if single_phase_inverter_topology == "full":
                Vg_RMS_max_theoretical = (M * Vdc_rated) / np.sqrt(2.0)

            else:  # half bridge
                Vg_RMS_max_theoretical = (M * Vdc_rated) / (2.0 * np.sqrt(2.0))

        else:  # three-phase

            if modulation_scheme == "svm":
                Vg_RMS_max_theoretical = (M * Vdc_rated) / np.sqrt(6.0)

            else:  # spwm
                Vg_RMS_max_theoretical = (M * Vdc_rated) / (2.0 * np.sqrt(2.0))

        if np.any(Vg_RMS < 0):
            raise ValueError(f"Invalid Vg_RMS detected. All values must be non-negative. "
                             f"Minimum detected value is {np.min(Vg_RMS):.3f} V.")

        if np.any(Vg_RMS > Vg_RMS_max_theoretical):
            violation_idx = np.where(Vg_RMS > Vg_RMS_max_theoretical)[0]
            raise ValueError(f"Invalid Vg_RMS detected. Vg_RMS must be less than or equal to "
                             f"Vg_RMS_max_theoretical at every profile second. "
                             f"First violation at index {violation_idx[0]}: "
                             f"Vg_RMS={Vg_RMS[violation_idx[0]]:.3f} V, "
                             f"Vg_RMS_max_theoretical={Vg_RMS_max_theoretical[violation_idx[0]]:.3f} V.")

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
                             f" period are required to accurately resolve PWM switching events."
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

            raise ValueError(
                "Not feasible: required inverter voltage exceeds capability.\n"
                f"Required peak = {Vs_peak:.2f} V, "
                f"Available = {available:.2f} V"
            )

    @staticmethod
    def Inverse_LCL_Filter_Grid_Connected_for_Vs(t, V_g, I_L2, L1, L2, C, R1, R2):

        """
        Reverse-calculate LCL filter quantities from known grid voltage and desired grid current.

        Known:
        ------
        V_g  : grid voltage [V]
        I_L2 : desired grid-side current [A]

        Calculates:
        -----------
        V_L2 : grid-side inductor voltage [V]
        V_C  : capacitor voltage [V]
        I_C  : capacitor current [A]
        I_L1 : inverter-side inductor current [A]
        V_L1 : inverter-side inductor voltage [V]
        V_s  : required inverter voltage [V]

        Explanation

        LCL Filter Differential Equations (Grid-Connected Inverter)

        This system models the dynamics of an LCL filter connecting an inverter to the grid.

        The circuit structure is:
            Vs → R1 → L1 → Vc → R2 → L2 → Vg
                              │
                              C
                              │
                            ground

        Where:
            Vs   : inverter output voltage
            Vg   : grid voltage
            Vc   : capacitor node voltage
            IL1  : current through inverter-side inductor (L1)
            IL2  : current through grid-side inductor (L2)
            Ic   : capacitor current
            R1,R2: series resistances
            L1,L2: inductances
            C    : capacitance

        ------------------------------------------------------------
        1) LEFT SIDE (Inverter → L1 → Capacitor node)

        Apply KVL:
            Vs = Vc + R1*IL1 + L1*dIL1/dt
        Rearranged into differential form:
            dIL1/dt = (Vs - Vc - R1*IL1) / L1

        ------------------------------------------------------------
        2) RIGHT SIDE (Capacitor node → L2 → Grid)

        Apply KVL:
            Vc = Vg + R2*IL2 + L2*dIL2/dt
        Rearranged into differential form:
            dIL2/dt = (Vc - Vg - R2*IL2) / L2

        ------------------------------------------------------------
        3) CAPACITOR NODE (KCL)

        At node Vc:
            IL1 = Ic + IL2
        Capacitor equation:
            Ic = C * dVc/dt
        Combine:
            C * dVc/dt = IL1 - IL2
        Rearranged:
            dVc/dt = (IL1 - IL2) / C

        ------------------------------------------------------------
        Summary of state equations:
        dIL1/dt = (Vs - Vc - R1*IL1) / L1
        dIL2/dt = (Vc - Vg - R2*IL2) / L2
        dVc/dt  = (IL1 - IL2) / C

        """

        # Function starts here
        t = np.asarray(t)
        V_g = np.asarray(V_g)
        I_L2 = np.asarray(I_L2)

        if t.ndim != 1 or V_g.ndim != 1 or I_L2.ndim != 1:
            raise ValueError("t, V_L2, and I_L2 must be 1D arrays.")

        if not (len(t) == len(V_g) == len(I_L2)):
            raise ValueError("t, V_L2, and I_L2 must have the same length.")

        if len(t) < 2:
            raise ValueError("t must contain at least two time points.")

        if L1 <= 0 or L2 <= 0 or C <= 0:
            raise ValueError("L1, L2, and C must be positive.")

        # From: Right Side
        dI_L2_dt = np.gradient(I_L2, t)  # Taking derivative of L2 inductor current
        V_C = V_g + (R2 * I_L2) + (L2 * dI_L2_dt)

        # From: Middle
        # I_C = C*dV_C/dt
        dV_C_dt = np.gradient(V_C, t)  # Taking derivative of C capacitor Voltage
        I_C = C * dV_C_dt
        # KCL: I_L1 = I_C + I_L2
        I_L1 = I_C + I_L2

        # From: Left
        # dIL1/dt
        dI_L1_dt = np.gradient(I_L1, t)  # Taking derivative of L1 inductor current
        # Required inverter voltage:
        # V_s = V_C + R1*I_L1 + V_L1
        V_s = V_C + (R1 * I_L1) + (L1 * dI_L1_dt)

        return V_s

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
    def LCL_Filter_Grid_Connected(t, Vs, Vg, L1, L2, C, R1, R2):

        """
        Solve the time-domain response of an LCL filter connected between an inverter and the grid.

        The function computes the inductor currents, capacitor voltage, and related voltages
        based on the differential equations of the LCL filter.

        Differential equations
        ----------------------
        dI_L1_dt = (Vs - V_C - (R1 * I_L1)) / L1
        dI_L2_dt = (V_C - V_g - R2 * I_L2) / L2
        dV_C_dt = (I_L1 - I_L2) / C

        Naming convention
        -----------------
        Vs  : inverter voltage input
        Vg  : known grid voltage
        I_L1 : current through left inductor
        V_L1 : voltage across left inductor
        I_L2 : current through right inductor
        V_L2 : voltage across right inductor
        V_C  : capacitor voltage
        I_C  : capacitor current
        R1  : Series resistance of inverter-side inductor
        R2  : Series resistance of grid-side inductor

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

        if len(t) != len(Vs) or len(t) != len(Vg):
            raise ValueError("t, Vs, and Vg must have the same length.")

        dt_array = np.diff(t)
        if not np.allclose(dt_array, dt_array[0]):
            raise ValueError("This discrete state-space method requires a fixed time step.")

        dt = dt_array[0]
        n = len(t)

        # State vector:
        # x = [I_L1, I_L2, V_C]
        A = np.array([[-R1 / L1, 0.0, -1.0 / L1],
                      [0.0, -R2 / L2, 1.0 / L2],
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
        V_L1 = Vs - V_C - R1 * I_L1
        V_L2 = V_C - Vg - R2 * I_L2

        # Optional consistency checks
        kcl_ok = np.allclose(I_L1, I_C + I_L2)
        kvl_left_ok = np.allclose(Vs, R1 * I_L1 + V_L1 + V_C)
        kvl_right_ok = np.allclose(V_C, R2 * I_L2 + V_L2 + Vg)

        if not (kcl_ok and kvl_left_ok and kvl_right_ok):
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

        return THD, THD_percent, I1_rms, freqs, mag_rms
