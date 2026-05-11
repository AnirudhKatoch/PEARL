import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete

plt.rcParams.update({"font.size": 15, "font.family": "Times New Roman", "axes.labelsize": 15, "axes.titlesize": 15,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15})

class All_the_functions_class:

    @staticmethod
    def validate_or_set_pulse_amplitude(Vdc_rated,inverter_phases,single_phase_inverter_topology=None,waveform_voltage_definition="pole_voltage",Vo=None):

        """
        Validate or derive the instantaneous PWM pulse amplitude Vo from Vdc_rated.

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
                interpreted here as pole voltage unless you explicitly model
                line-to-line switched voltages elsewhere.
        Vo : float or None
            Requested pulse amplitude [V]. If None, the function returns the
            maximum allowed Vo.

        Returns
        -------
        Vo_rated : float
            Validated or derived pulse amplitude [V]
        Vo_theoretical_max : float
            Maximum physically allowed pulse amplitude [V]
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
            # For standard 2-level three-phase leg voltages
            Vo_theoretical_max = Vdc_rated / 2.0

        if Vo is None:
            return Vo_theoretical_max, Vo_theoretical_max

        if Vo < 0:
            raise ValueError("Vo must be non-negative.")

        if Vo > Vo_theoretical_max:
            raise ValueError(
                f"Invalid Vo={Vo:.3f} V. Maximum allowed Vo is {Vo_theoretical_max:.3f} V "
                f"for Vdc_rated={Vdc_rated:.3f} V with the selected inverter configuration.")

        Vo_rated = Vo

        return Vo_rated, Vo_theoretical_max

    @staticmethod
    def compute_theoretical_fundamental_rms_limit(Vdc_rated,M,inverter_phases,modulation_scheme,single_phase_inverter_topology=None):
        """
        Compute the theoretical RMS limit of the fundamental AC-side phase voltage.

        Parameters
        ----------
        Vdc_rated : float
            Rated DC bus voltage [V]
        M : float
            Modulation index [-]
        inverter_phases : {1, 3}
            Number of inverter phases
        modulation_scheme : {"spwm", "svm"}
            Modulation scheme
        single_phase_inverter_topology : {"half", "full"}, optional
            Required when inverter_phases == 1

        Returns
        -------
        Vs_theoretical : float
            Maximum fundamental RMS phase voltage [V]
        """

        if Vdc_rated <= 0:
            raise ValueError("Vdc_rated must be positive.")
        if M < 0:
            raise ValueError("M must be non-negative.")
        if inverter_phases not in (1, 3):
            raise ValueError("inverter_phases must be 1 or 3.")
        if modulation_scheme not in ("spwm", "svm"):
            raise ValueError("modulation_scheme must be 'spwm' or 'svm'.")

        if inverter_phases == 1:
            if single_phase_inverter_topology not in ("half", "full"):
                raise ValueError(
                    "For single-phase inverter, single_phase_inverter_topology must be 'half' or 'full'."
                )

            if single_phase_inverter_topology == "full":
                Vs_theoretical = (M * Vdc_rated) / np.sqrt(2.0)
            else:  # half
                Vs_theoretical = (M * Vdc_rated) / (2.0 * np.sqrt(2.0))

        else:  # 3-phase
            if modulation_scheme == "svm":
                Vs_theoretical = (M * Vdc_rated) / np.sqrt(6.0)
            else:  # spwm
                Vs_theoretical = (M * Vdc_rated) / (2.0 * np.sqrt(2.0))

        Vs_RMS_max_theoretical = Vs_theoretical

        return Vs_RMS_max_theoretical

    @staticmethod
    def Sinusoidal_Pulse_Width_Modulation_One_Phase_old(t, M, f, Tsw, Vo, T, Vdc):

        """
        Generate single-phase sinusoidal PWM (SPWM) voltage waveform.

        Parameters
        ----------
        t : array
        Time vector [s]
        M : array
        Modulation index [-]
        f : float
        Fundamental frequency [Hz]
        Tsw : float
        Switching period [s]
        Vo : array
        PWM pulse amplitude [V]
        T : float
        Fundamental period [s]
        Vdc : array
        DC bus voltage [V]

        Returns
        -------
        V_s : array
        PWM output voltage waveform [V]
        """

        t_profile = np.arange(len(Vdc))  # Time vector [s] with 1-second resolution, same length as Vdc mission profile
        if len(M) == len(Vdc):
            M_t = np.interp(t, t_profile, M)
        else:
            M_t = M

        Vo_t = np.interp(t, t_profile, Vo)

        def v_ref(t, M_t):
            return M_t * np.sin(2 * np.pi * f * t)

        def v_carrier(t, Tsw):
            tau = (t % Tsw) / Tsw
            return 1.0 - np.abs(2 * tau - 1.0)


        def vs_half_spwm(t, M_t, Vo_t):
            return np.where(v_ref(t, M_t) >= v_carrier(t, Tsw), Vo_t, 0.0)

        def vs_full_spwm(t, M_t, Vo_t):
            tt = t % T
            half_T = T / 2  # Half-cycle duration [s]
            return np.where(tt < half_T, vs_half_spwm(t, M_t, Vo_t), -vs_half_spwm(tt - half_T, M_t, Vo_t))

        V_s = vs_full_spwm(t, M_t, Vo_t)

        del t_profile, M_t, Vo_t

        return V_s


    @staticmethod
    def Solving_LCL_Filter_Grid_Connected(t, V_s, V_g, L1, L2, C, R1, R2):

        """
        Solve the time-domain response of an LCL filter connected between an inverter and the grid.

        The function computes the inductor currents, capacitor voltage, and related voltages
        based on the differential equations of the LCL filter.

        Differential equations
        ----------------------
        dI_L1/dt = (V_s - V_C) / L1
        dI_L2/dt = (V_C - V_g) / L2
        dV_C/dt  = (I_L1 - I_L2) / C

        Naming convention
        -----------------
        V_s  : inverter voltage input
        V_g  : known grid voltage
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
        V_s : array
        Inverter output voltage [V]
        V_g : array
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
        V_s = np.asarray(V_s)
        V_g = np.asarray(V_g)

        if t.ndim != 1 or V_s.ndim != 1 or V_g.ndim != 1:
            raise ValueError("t, V_s, and V_g must be 1D arrays.")

        if not (len(t) == len(V_s) == len(V_g)):
            raise ValueError("t, V_s, and V_g must have the same length.")

        if len(t) < 2:
            raise ValueError("t must contain at least two time points.")

        if L1 <= 0 or L2 <= 0 or C <= 0:
            raise ValueError("L1, L2, and C must be positive.")

        def lcl_ode(t_now, x, t_grid, V_s_grid, V_g_grid, L1, L2, C):
            I_L1, I_L2, V_C = x

            V_s_now = np.interp(t_now, t_grid, V_s_grid)
            V_g_now = np.interp(t_now, t_grid, V_g_grid)

            #dI_L1_dt = (V_s_now - V_C) / L1
            #dI_L2_dt = (V_C - V_g_now) / L2
            #dV_C_dt = (I_L1 - I_L2) / C

            dI_L1_dt = (V_s_now - V_C - (R1 * I_L1)) / L1
            dI_L2_dt = (V_C - V_g_now - R2 * I_L2) / L2
            dV_C_dt = (I_L1 - I_L2) / C

            return [dI_L1_dt, dI_L2_dt, dV_C_dt]

        # Initial conditions: [I_L1(0), I_L2(0), V_C(0)]
        x0 = [0.0, 0.0, 0.0]

        sol = solve_ivp(fun=lambda t_now, x: lcl_ode(t_now, x, t, V_s, V_g, L1, L2, C), t_span=(t[0], t[-1]), y0=x0,
                        t_eval=t, method="RK45")

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        # State variables
        I_L1 = sol.y[0]
        I_L2 = sol.y[1]
        V_C = sol.y[2]

        # Derived quantities
        #V_L1 = V_s - V_C
        #V_L2 = V_C - V_g
        #I_C = I_L1 - I_L2

        V_L1 = V_s - V_C - R1 * I_L1
        V_L2 = V_C - V_g - R2 * I_L2
        I_C = I_L1 - I_L2

        # Optional consistency checks
        kcl_ok = np.allclose(I_L1, I_C + I_L2)
        kvl_left_ok = np.allclose(V_s, R1 * I_L1 + V_L1 + V_C)
        kvl_right_ok = np.allclose(V_C, R2 * I_L2 + V_L2 + V_g)

        if not (kcl_ok and kvl_left_ok and kvl_right_ok):
            print("Warning: one or more KCL/KVL checks are not within tolerance.")

        return V_L1, I_L1, V_C, I_C, V_L2, I_L2


    @staticmethod
    def Solving_LCL_Filter_Grid_Connected_Known_IL2(t, V_s, V_g, I_L2_known, L1, C, R1, R2, L2=None):

        """
        Solve the time-domain response of an LCL filter when grid-side current I_L2 is known.

        In this formulation, I_L2 is treated as a known input rather than a state.
        The solved states are:
            - I_L1
            - V_C

        Governing equations
        -------------------
        dI_L1/dt = (V_s - V_C - R1*I_L1) / L1
        dV_C/dt  = (I_L1 - I_L2_known) / C

        Derived quantities
        ------------------
        I_C  = I_L1 - I_L2_known
        V_L1 = V_s - V_C - R1*I_L1
        V_L2 = V_C - V_g - R2*I_L2_known

        If L2 is provided, an optional consistency check is also performed:
            V_L2 ?= L2 * dI_L2_known/dt

        Parameters
        ----------
        t : array
            Time vector [s]
        V_s : array
            Inverter output voltage [V]
        V_g : array
            Grid voltage [V]
        I_L2_known : array
            Known grid-side inductor current [A]
        L1 : float
            Inverter-side inductance [H]
        C : float
            Filter capacitance [F]
        R1 : float
            Series resistance of inverter-side inductor [Ohm]
        R2 : float
            Series resistance of grid-side inductor [Ohm]
        L2 : float or None, optional
            Grid-side inductance [H]. Only needed for optional consistency check.

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
            Known grid-side inductor current [A]
        """

        import numpy as np
        from scipy.integrate import solve_ivp

        t = np.asarray(t, dtype=float)
        V_s = np.asarray(V_s, dtype=float)
        V_g = np.asarray(V_g, dtype=float)
        I_L2_known = np.asarray(I_L2_known, dtype=float)

        if t.ndim != 1 or V_s.ndim != 1 or V_g.ndim != 1 or I_L2_known.ndim != 1:
            raise ValueError("t, V_s, V_g, and I_L2_known must be 1D arrays.")

        if not (len(t) == len(V_s) == len(V_g) == len(I_L2_known)):
            raise ValueError("t, V_s, V_g, and I_L2_known must have the same length.")

        if len(t) < 2:
            raise ValueError("t must contain at least two time points.")

        if L1 <= 0 or C <= 0:
            raise ValueError("L1 and C must be positive.")

        if np.any(np.diff(t) <= 0):
            raise ValueError("t must be strictly increasing.")

        def lcl_reduced_ode(t_now, x, t_grid, V_s_grid, I_L2_grid, L1, C, R1):

            I_L1, V_C = x

            V_s_now = np.interp(t_now, t_grid, V_s_grid)
            I_L2_now = np.interp(t_now, t_grid, I_L2_grid)

            dI_L1_dt = (V_s_now - V_C - (R1 * I_L1)) / L1
            dV_C_dt = (I_L1 - I_L2_now) / C

            return [dI_L1_dt, dV_C_dt]

        # Initial conditions: [I_L1(0), V_C(0)]
        x0 = [0.0, 0.0]

        sol = solve_ivp(
            fun=lambda t_now, x: lcl_reduced_ode(t_now, x, t, V_s, I_L2_known, L1, C, R1),
            t_span=(t[0], t[-1]),
            y0=x0,
            t_eval=t,
            method="RK45"
        )

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        # State variables
        I_L1 = sol.y[0]
        V_C = sol.y[1]

        # Known / derived quantities
        I_L2 = I_L2_known
        I_C = I_L1 - I_L2
        V_L1 = V_s - V_C - R1 * I_L1
        V_L2 = V_C - V_g - R2 * I_L2

        # Optional consistency checks
        kcl_ok = np.allclose(I_L1, I_C + I_L2)

        kvl_left_ok = np.allclose(V_s, R1 * I_L1 + V_L1 + V_C)

        kvl_right_ok = np.allclose(V_C, R2 * I_L2 + V_L2 + V_g)

        if not (kcl_ok and kvl_left_ok and kvl_right_ok):
            print("Warning: one or more KCL/KVL checks are not within tolerance.")

        if L2 is not None:
            dI_L2_dt = np.gradient(I_L2, t)
            v_l2_from_inductor_law = L2 * dI_L2_dt
            l2_ok = np.allclose(V_L2, v_l2_from_inductor_law, rtol=1e-3, atol=1e-3)

            if not l2_ok:
                #print("Warning: V_L2 is not fully consistent with L2 * dI_L2/dt for the prescribed I_L2.")
                None

        return V_L1, I_L1, V_C, I_C, V_L2, I_L2



    @staticmethod
    def Plotting_Grid_Connected_LCL_filter(t, V_L1, I_L1, V_C, I_C, V_L2, I_L2, f):

        mask = t >= (t[-1] - (2/f))


        plt.figure(figsize=(6.4 * 2, 4.8))
        plt.plot(t[mask], V_L1[mask], label="V_L1", linewidth=1.2)
        plt.plot(t[mask], V_C[mask], label="V_C", linewidth=1.2)
        plt.plot(t[mask], V_L2[mask], label="V_L2", linewidth=1.2)
        plt.title("Voltages in L-C-L filter")
        plt.xlabel("Time [s]")
        plt.ylabel("Voltage [V]")
        plt.xlim([min(t[mask]), max(t[mask])])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("Figures/Voltage.png")
        plt.close()

        plt.figure(figsize=(6.4 * 2, 4.8))
        #plt.plot(t[mask], I_L1[mask], label="I_L1", linewidth=1.5)
        #plt.plot(t[mask], I_C[mask], label="I_C", linewidth=1.2)
        plt.plot(t[mask], I_L2[mask], label="I_L2", linewidth=1.2)
        plt.title("Currents in L-C-L filter")
        plt.xlabel("Time [s]")
        plt.ylabel("Current [A]")
        plt.xlim([min(t[mask]), max(t[mask])])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("Figures/Current.png")
        plt.close()

    @staticmethod
    def Plotting_PWM_Output_Voltage(t, V_s):


        # Time window
        t_start = 0.00
        t_end = 0.02

        # Mask for the desired interval
        mask = (t >= t_start) & (t <= t_end)

        # Plot
        plt.figure(figsize=(6.4 * 2, 4.8))
        plt.plot(t[mask], V_s[mask])
        plt.xlabel("Time [s]")
        plt.ylabel("V_s [V]")
        plt.title("PWM Output Voltage")
        plt.xlim(t_start, t_end)
        plt.grid()
        plt.savefig("Figures/PWM Output Voltage.png")

    @staticmethod
    def THD_and_harmonics(signal, t_ss):

        N = len(signal)
        dt = t_ss[1] - t_ss[0]
        fs = 1 / dt

        fft_vals = np.fft.fft(signal)
        fft_vals = np.abs(fft_vals) / N

        freqs = np.fft.fftfreq(N, d=dt)

        # keep only positive frequencies
        mask = freqs > 0
        freqs = freqs[mask]
        fft_vals = fft_vals[mask]

        f0 = 50  # or 60 depending on your setup

        fund_idx = np.argmin(np.abs(freqs - f0))
        I1 = fft_vals[fund_idx]

        harmonics = np.copy(fft_vals)
        harmonics[fund_idx] = 0

        THD = np.sqrt(np.sum(harmonics ** 2)) / I1

        print("THD (%):", THD * 100)

        num_harmonics = 20

        harmonic_numbers = np.arange(1, num_harmonics + 1)
        harmonic_amplitudes = []

        for n in harmonic_numbers:
            target_freq = n * f0
            idx = np.argmin(np.abs(freqs - target_freq))
            harmonic_amplitudes.append(fft_vals[idx])

        harmonic_amplitudes = np.array(harmonic_amplitudes)

        plt.figure(figsize=(10, 5))
        plt.bar(harmonic_numbers, harmonic_amplitudes)

        plt.title("Harmonic Spectrum of Output Current i1(t)")
        plt.xlabel("Harmonic Number (n × f0)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.savefig("Figures/Harmonic_Spectrum.png")

    @staticmethod
    def plot_voltage_current(
            t,
            Vg,
            Ig,
            t_end,
            voltage_label,
            current_label,Location,
    ):
        """
        Plot voltage and current on dual y-axis up to a given time.

        Parameters
        ----------
        t : array
            Time vector [s]
        Vg : array
            Voltage signal [V]
        Ig : array
            Current signal [A]
        t_end : float
            Time until which to plot [s]
        voltage_label : str
            Label for voltage
        current_label : str
            Label for current
        """

        import numpy as np
        import matplotlib.pyplot as plt

        mask = t <= t_end

        t_plot = t[mask]
        Vg_plot = Vg[mask]
        Ig_plot = Ig[mask]

        fig, ax1 = plt.subplots()

        # Voltage (left axis)
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Voltage [V]", color="blue")
        ax1.plot(t_plot, Vg_plot, color="blue", label=voltage_label)
        ax1.tick_params(axis='y', labelcolor="blue")

        # Current (right axis)
        ax2 = ax1.twinx()
        ax2.set_ylabel("Current [A]", color="red")
        ax2.plot(t_plot, Ig_plot, color="red", linestyle="--", label=current_label)
        ax2.tick_params(axis='y', labelcolor="red")

        plt.xlim(min(t_plot),max(t_plot))
        plt.title(f"{voltage_label} and {current_label} up to {t_end}s")
        plt.grid()
        plt.savefig(Location)

    @staticmethod
    def plot_voltage_signal(
            t,
            Vs,
            t_end,
            voltage_label,
            Location,
    ):
        """
        Plot voltage signal up to a given time and show RMS value.

        Parameters
        ----------
        t : array
            Time vector [s]
        Vs : array
            Voltage signal [V]
        t_end : float
            Time until which to plot [s]
        voltage_label : str
            Label for voltage
        Location : str
            Save location for plot
        """

        t = np.asarray(t)
        Vs = np.asarray(Vs)

        mask = t <= t_end

        t_plot = t[mask]
        Vs_plot = Vs[mask]

        Vs_rms = np.sqrt(np.mean(Vs_plot ** 2))

        fig, ax1 = plt.subplots()

        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Voltage [V]", color="blue")
        ax1.plot(t_plot, Vs_plot, color="blue", label=voltage_label)
        ax1.tick_params(axis='y', labelcolor="blue")

        ax1.axhline(Vs_rms, linestyle="--", label=f"RMS = {Vs_rms:.2f} V")
        ax1.axhline(-Vs_rms, linestyle="--")

        plt.xlim(min(t_plot), max(t_plot))
        plt.title(f"{voltage_label} up to {t_end}s | RMS = {Vs_rms:.2f} V")
        plt.grid()
        plt.legend()
        plt.savefig(Location)
        plt.close()


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
    def checking_V_s_to_Vs_ref(t, Tsw, dt, Vs, Vs_ref, t_end, Location):

        # Time window
        t_start = 0.00
        # Mask for the desired interval
        mask = (t >= t_start) & (t <= t_end)

        samples_per_switching_period = int(round(Tsw / dt))
        kernel = np.ones(samples_per_switching_period) / samples_per_switching_period
        Vs_avg = np.convolve(Vs, kernel, mode="same")
        plt.figure()
        plt.plot(t[mask], Vs_ref[mask], label="Vs_ref")
        plt.plot(t[mask], Vs_avg[mask], label="PWM moving average")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Voltage [V]")
        plt.legend()
        plt.savefig(Location)

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
    def checking_I_L2_to_Ig_ref(t, Ig_ref, I_L2, t_end, Location):

        t_start = 0.00
        mask = (t >= t_start) & (t <= t_end)

        plt.figure(figsize=(6.4, 4.8))

        plt.plot(t[mask], Ig_ref[mask], label="Ig_ref")
        plt.plot(t[mask], I_L2[mask], label="I_L2")

        plt.xlabel("Time [s]")
        plt.ylabel("Current [A]")
        plt.title("Grid Current Reference vs LCL Output Current")

        plt.legend()
        plt.grid(True)
        plt.savefig(Location)
        plt.close()

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