import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class All_the_functions_class:

    @staticmethod
    def validate_or_set_pulse_amplitude(Vdc_rated,inverter_phases,single_phase_inverter_topology=None,waveform_voltage_definition="switched_output",Vo=None):

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
    def Sinusoidal_Pulse_Width_Modulation_One_Phase(t, M, f, Tsw, Vo, T, Vdc):

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

        def v_ref(t, M_t):
            return M_t * np.sin(2 * np.pi * f * t)

        def v_carrier(t, Tsw):
            tau = (t % Tsw) / Tsw
            return 1.0 - np.abs(2 * tau - 1.0)

        Vo_t = np.interp(t, t_profile, Vo)

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
        plt.plot(t[mask], I_L1[mask], label="I_L1", linewidth=1.5)
        plt.plot(t[mask], I_C[mask], label="I_C", linewidth=1.2)
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