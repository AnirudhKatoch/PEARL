"""
Calculation_functions.py  (PEARL v3)

All PEARL v1 functions are kept unchanged (top of the class). The PEARL v2 additions are at
the end of the class: modulation, physical loss waveforms, operating-point binning, periodic
rainflow, slow rainflow, damage, rainflow matrices / cells, Monte Carlo, Weibull, folders.
"""
import numpy as np
import numexpr as ne
from pathlib import Path
import rainflow
import pandas as pd

class Calculation_functions_class:


    @staticmethod
    def compute_power_flow(P,
                           Q,
                           V_dc,
                           Vs,
                           M,
                           single_phase_inverter_topology,
                           inverter_phases,
                           modulation_scheme,
                           N_parallel):

        """
        Compute apparent power S, RMS current Is, and phase angle phi

        Parameters
        ----------
        P : array
            Active power per sec [W]
        Q : array
            Reactive power per sec [VAr]
        V_dc : array
             DC-side phase voltage per sec [V]
        Vs : array
             RMS AC-side phase voltage per sec [V]
        M : float
            Modulation index [-]
        single_phase_inverter_topology : {"half","full"}
            Inverter topology (affects Vs limit for single-phase).
        inverter_phases : {1,3}
            Number of phases. If 3, Vs is interpreted as PHASE RMS (i.e., V_ll/sqrt(3)).
        modulation_scheme : {"spwm","svm"}
            Modulation strategy used for generating inverter switching signals.

        Vs, Is, phi, V_dc, pf, M, S
        Returns
        -------

        Vs : array
             RMS AC-side phase voltage per sec [V]
        Is : array
            RMS current per sample [A].
        phi : array
            Phase angle between voltage and current per sample [rad]
        V_dc : array
            DC-side  voltage per sec [V]
        pf : array
            Power factor per sec [-].
        M : float
            Modulation index [-]
        S : array
            Apparent power per sample [VA].


        """

        P = P / N_parallel
        Q = Q / N_parallel

        pf = np.zeros_like(P, dtype=float)
        Is = np.zeros_like(P, dtype=float)  # [A] Inverter RMS current
        phi = np.zeros_like(P, dtype=float)  # [rad] Phase angle

        S = np.sqrt(P ** 2 + Q ** 2) # [VA] Inverter RMS apparent power

        # Case 1: P = 0 AND Q != 0 → pf = 0
        m_P0_Qnz = (P == 0) & (Q != 0)
        pf[m_P0_Qnz] = 0.0

        # Case 2: P != 0 AND Q = 0 → pf = ±1
        m_Pnz_Q0 = (P != 0) & (Q == 0)
        pf[m_Pnz_Q0] = np.sign(P[m_Pnz_Q0]) * 1.0

        # Case 3: General case (both P and Q nonzero)
        m_general = (P != 0) & (Q != 0)
        pf[m_general] = np.abs(P[m_general] / S[m_general])
        pf[(m_general & (Q < 0))] *= -1


        if inverter_phases == 1:
            if single_phase_inverter_topology == "full":
                Vs_theoretical = (M * V_dc) / np.sqrt(2.0)
            elif single_phase_inverter_topology == "half":
                Vs_theoretical = (M * V_dc) / (2.0 * np.sqrt(2.0))
        elif inverter_phases == 3:
            if modulation_scheme == "svm":
                # Space vector PWM (or 3rd harmonic injection)
                Vs_theoretical = (M * V_dc) / np.sqrt(6.0)  # [V RMS phase]
            elif modulation_scheme == "spwm":  # "spwm"
                # Sinusoidal PWM
                Vs_theoretical = (M * V_dc) / (2.0 * np.sqrt(2.0))

        if Vs.size == 0:
            Vs = Vs_theoretical.copy()

        else:
            indices = np.where(Vs > Vs_theoretical)[0]
            if indices.size > 0:
                raise ValueError(
                    f"Invalid input: AC phase RMS voltage exceeds the theoretical limit "
                    f"Vs must not be greater than {np.max(Vs_theoretical)}.")


        # masks
        m0 = pf == 0  # zero power factor
        mneg = pf < 0  # inductive
        mpos = pf > 0  # capacitive

        # ---- pf == 0 branch ----
        # P[i] = 0
        P[m0] = 0.0

        # S[i] = sqrt(P[i]^2 + Q[i]^2)  (with P already zeroed where m0)
        S[m0] = np.sqrt(P[m0] ** 2 + Q[m0] ** 2)

        # Is[i] = S[i] / Vs[i]
        with np.errstate(divide='ignore', invalid='ignore'):
            Is[m0] = S[m0] / (Vs[m0] if inverter_phases == 1 else (3.0 * Vs[m0]))

        # phi: 0 if S==0 else ±pi/2 depending on sign of Q
        phi[m0] = 0.0
        nz = m0 & (S != 0)
        phi[nz] = np.where(Q[nz] > 0, np.pi / 2, -np.pi / 2)

        # ---- pf != 0 branch ----
        abspf = np.abs(pf)
        mnz = ~m0  # pf != 0

        # S[i] = P[i] / abs(pf[i])
        S[mnz] = P[mnz] / abs(pf[mnz])

        # Is[i] = S[i] / Vs[i]
        with np.errstate(divide='ignore', invalid='ignore'):
            Is[mnz] = S[mnz] / (Vs[mnz] if inverter_phases == 1 else (3.0 * Vs[mnz]))

        # phi[i] = ± arccos(abs(pf[i]))
        phi[mneg] = -np.arccos(abspf[mneg])  # inductive
        phi[mpos] = np.arccos(abspf[mpos])  # capacitive

        # Q[i] = ± sqrt(S[i]^2 - P[i]^2) for pf != 0
        # (Note: numerical noise can make the radicand slightly negative; clip at 0.)
        rad = (S[mnz] ** 2 - P[mnz] ** 2)
        root = np.sqrt(rad)
        idx_mnz = np.where(mnz)[0]
        Q[idx_mnz[mneg[mnz]]] = -root[mneg[mnz]]
        Q[idx_mnz[mpos[mnz]]] = root[mpos[mnz]]

        return Vs, Is, phi, V_dc, pf, M, S

    @staticmethod
    def check_max_package_current_limit(Is,M, max_IGBT_current, max_diode_current):
        """
        Check that inverter RMS currents do not exceed device package limits.

        Parameters
        ----------
        Is : array
            RMS current on the AC side [A].
        max_IGBT_current : float
            Maximum allowable RMS current rating of the IGBT [A].
        max_diode_current : float, optional
            Maximum allowable RMS current rating of the diode [A].

        Raises
        ------
        ValueError
            If IGBT or diode RMS currents exceed their device ratings.
        """

        I_s_rms  = np.max(Is)
        Maximum_IGBT_current = 0.5 * I_s_rms * np.sqrt(1.0 + ((8.0 * M * 1.0) / (3.0 * np.pi)))
        Maximum_Diode_current = 0.5 * I_s_rms * np.sqrt(1.0 - ((8.0 * M * 1.0) / (3.0 * np.pi)))

        #print("Maximum_IGBT_current",Maximum_IGBT_current)

        # ---- IGBT CHECK ----
        if np.any(Maximum_IGBT_current > max_IGBT_current):
            raise ValueError(
                f"IGBT RMS current limit exceeded: "
                f"max allowed {max_IGBT_current} A"
            )

        if np.any(Maximum_Diode_current > max_diode_current):
                raise ValueError(
                    f"Diode RMS current limit exceeded: "
                    f"max allowed {max_diode_current} A"
                )


    @staticmethod
    def check_vce(V_dc, max_V_CE):
        """
        Check if collector–emitter voltage exceeds the maximum allowed value.

        Parameters
        ----------

        V_dc : float or np.ndarray
            DC link voltage (single value or array of values).
        max_V_CE : float
            Maximum allowed collector–emitter voltage.

        Raises
        ------
        UserWarning
            If the collector–emitter voltage exceeds max_V_CE.
        """
        V_CE = V_dc

        violations = np.where(V_CE > max_V_CE)[0]

        if violations.size > 0:
            raise ValueError(
                f"Collector–emitter voltage exceeded! "
                f"Maximum allowed: {max_V_CE}, but got values up to {V_CE.max()}. "
                "Please reduce the DC voltage or update the IGBT specifications."
            )

        return V_CE

    @staticmethod
    def Instantaneous_modulation(M, omega, t, phi):

        """
        Calculate the inverter modulation function m(t).

        Parameters
        ----------
        M : float
            Modulation index of the inverter [-] (typically 0.8–1.0 for PV inverters).
        omega : float
            Angular frequency of the AC grid [rad/s] (ω = 2πf).
        t : array
            Time instant [s].
        phi : float
            Phase angle of the inverter output current relative to the voltage [rad].

        Returns
        -------
        m : float
                Instantaneous modulation function [-].

        """

        m = (M * np.sin(omega * t + phi) + 1) / 2

        return m

    @staticmethod
    def IGBT_and_diode_current(Is, t, m, omega):

        """
        Calculate the instantaneous IGBT and diode currents in one inverter leg

        Parameters
        ----------
        Is : float
            RMS value of the inverter output current [A].
        t : array
            Time instant [s].
        m : array
            Instantaneous modulation function [-], typically between 0 and 1.

        Returns
        -------
        is_I : array
            Instantaneous IGBT current [A]. (Non-negative only, conduction blocked when negative.)
        is_D : array
            Instantaneous diode current [A]. (Non-negative only, conduction blocked when negative.)
        """

        base = np.sqrt(2) * Is * np.sin(omega * t) * m
        is_I = np.maximum(base, 0)
        is_D = np.maximum(-base, 0)

        return is_I, is_D

    @staticmethod
    def Switching_losses(V_dc, is_I, t_on, t_off, f_sw, is_D, I_ref, V_ref, Err_D):

        """
        Calculate the IGBT and diode switching power losses.

        Parameters
        ----------
        V_dc : float
            DC-link voltage of the inverter [V].
        is_I : array
            Instantaneous current through the IGBT [A].
        t_on : float
            Effective IGBT turn-on time [s].
        t_off : float
            Effective IGBT turn-off time [s].
        f_sw : float
            Inverter switching frequency [Hz].
        is_D : array
            Instantaneous current through the diode [A].
        I_ref : float
            Reference test current for diode reverse recovery [A].
        V_ref : float
            Reference test voltage for diode reverse recovery [V].
        Err_D : float
            Reverse recovery energy per switching event for the diode [J].

        Returns
        -------
        P_sw_I : array
            Instantaneous IGBT switching power loss [W].
        P_sw_D : array
            Instantaneous diode switching power loss [W].

        """

        is_I = np.ascontiguousarray(is_I, dtype=np.float64)
        is_D = np.ascontiguousarray(is_D, dtype=np.float64)

        # IGBT

        # E_on_I = ((np.sqrt(2) / (2 * np.pi)) * V_dc * is_I * t_on)
        # E_off_I = ((np.sqrt(2) / (2 * np.pi)) * V_dc * is_I * t_off)

        c1 = (np.sqrt(2) / (2 * np.pi))
        c2 = (np.sqrt(2) / np.pi)

        P_sw_I_expr = ne.evaluate("(( c1 * V_dc * is_I * t_on) + ( c1 * V_dc * is_I * t_off)) * f_sw",
                                  local_dict=dict(c1=c1, V_dc=V_dc, is_I=is_I, t_on=t_on, t_off=t_off, f_sw=f_sw), )
        P_sw_I = ne.evaluate("where(P_sw_I_expr > 0.0, P_sw_I_expr, 0.0)", local_dict=dict(P_sw_I_expr=P_sw_I_expr))

        # Diode

        P_sw_D_expr = ne.evaluate("((c2 * (is_D * V_dc) / (I_ref * V_ref)) * Err_D * f_sw)",
                                  local_dict=dict(c2=c2, V_dc=V_dc, is_D=is_D, f_sw=f_sw, I_ref=I_ref, V_ref=V_ref,
                                                  Err_D=Err_D), )
        # P_sw_D = np.maximum(P_sw_D, 0)
        P_sw_D = ne.evaluate("where(P_sw_D_expr > 0.0, P_sw_D_expr, 0.0)", local_dict=dict(P_sw_D_expr=P_sw_D_expr))

        return P_sw_I, P_sw_D

    @staticmethod
    def Conduction_losses(is_I, R_IGBT, V_0_IGBT, M, pf, is_D, R_D, V_0_D):

        """
        Calculate conduction losses of the inverter’s IGBT and diode.

        Parameters
        ----------
        is_I : array
            Instantaneous value of the inverter output current flowing through the IGBT [A].
        R_IGBT : float
            Effective on-resistance of the IGBT conduction model [Ohm].
        V_0_IGBT : float
            Effective knee voltage of the IGBT [V].
        M : float
            Modulation index of the inverter [-].
        pf : float
            Power factor of inverter output current [-].
            (negative = inductive load, current lags voltage;
             positive = capacitive load, current leads voltage).
        is_D : array
            Instantaneous value of the inverter output current flowing through the diode [A].
        R_D : float
            Effective dynamic resistance of the diode [Ohm].
        V_0_D : float
            Effective forward knee voltage of the diode [V].

        Returns
        -------
        P_con_I : array
            Instantaneous conduction loss of the IGBT [W].
        P_con_D : array
            Instantaneous conduction loss of the diode [W].
        """

        is_I = np.ascontiguousarray(is_I, dtype=np.float64)
        is_D = np.ascontiguousarray(is_D, dtype=np.float64)
        pf = np.ascontiguousarray(pf, dtype=np.float64)

        c1 = np.sqrt(2 * np.pi)
        c2 = (3 * np.pi)
        c3 = np.pi

        # IGBT

        # P_con_I = (((is_I ** 2 / 4.0) * R_IGBT) + ((is_I / np.sqrt(2 * np.pi)) * V_0_IGBT) +
        #           ((((is_I ** 2 / 4.0) * (8 * M / (3 * np.pi)) * R_IGBT) + (
        #                   (is_I / np.sqrt(2 * np.pi)) * (np.pi * M / 4.0) * V_0_IGBT)) * abs(pf)))
        # P_con_I = np.maximum(P_con_I, 0)

        P_con_I_expr = ne.evaluate("(((is_I ** 2 / 4.0) * R_IGBT) + ((is_I / c1) * V_0_IGBT) + "
                                   "((((is_I ** 2 / 4.0) * (8 * M / c2) * R_IGBT) +"
                                   " ((is_I / c1) * (c3 * M / 4.0) * V_0_IGBT)) * abs(pf)))",
                                   local_dict=dict(is_I=is_I, R_IGBT=R_IGBT, V_0_IGBT=V_0_IGBT, M=M, pf=pf, c1=c1,
                                                   c2=c2, c3=c3), )
        P_con_I = ne.evaluate("where(P_con_I_expr > 0.0, P_con_I_expr, 0.0)",
                              local_dict=dict(P_con_I_expr=P_con_I_expr))

        # Diode

        # P_con_D = ((((is_D ** 2 / 4.0) * R_D) + ((is_D / np.sqrt(2 * np.pi)) * V_0_D)) -
        #           ((((is_D ** 2 / 4.0)) * ((8 * M / (3 * np.pi)) * R_D)) + (
        #                       (is_D / np.sqrt(2 * np.pi)) * (np.pi * M / 4.0) * V_0_D)) * abs(pf))
        # P_con_D = np.maximum(P_con_D, 0)

        # Diode

        P_con_D_expr = ne.evaluate("((((is_D ** 2 / 4.0) * R_D) + ((is_D / c1) * V_0_D)) -"
                                   " ((((is_D ** 2 / 4.0)) * ((8 * M / c2) * R_D)) + "
                                   "((is_D / c1) * (c3 * M / 4.0) * V_0_D)) * abs(pf))",
                                   local_dict=dict(is_D=is_D, R_D=R_D, V_0_D=V_0_D, M=M, pf=pf, c1=c1, c2=c2, c3=c3), )
        P_con_D = ne.evaluate("where(P_con_D_expr > 0.0, P_con_D_expr, 0.0)",
                              local_dict=dict(P_con_D_expr=P_con_D_expr))

        return P_con_I, P_con_D

    @staticmethod
    def create_simulation_folders(base="Results"):
        """
        Creates:
            Dataframes/
                Simulation_N/
                    df_electrical_loss/
                    df_thermal/

        Automatically increments Simulation_N.
        Returns:
            sim_dir, df_electrical_loss_dir, df_thermal_dir
        """

        base_dir = Path(base)
        base_dir.mkdir(exist_ok=True)

        # --- detect existing Simulation_N folders ---
        existing = []
        for p in base_dir.iterdir():
            if p.is_dir() and p.name.startswith("Simulation_"):
                try:
                    n = int(p.name.split("_")[1])
                    existing.append(n)
                except (IndexError, ValueError):
                    pass

        # --- choose next folder number ---
        next_n = max(existing) + 1 if existing else 1

        # --- create Simulation_N folder ---
        sim_dir = base_dir / f"Simulation_{next_n}"
        sim_dir.mkdir(exist_ok=True)

        # --- create subfolders ---
        df_electrical_loss_dir = sim_dir / "df_electrical_loss"
        df_electrical_loss_dir.mkdir(exist_ok=True)

        df_thermal_dir = sim_dir / "df_thermal"
        df_thermal_dir.mkdir(exist_ok=True)

        Figures_dir = sim_dir / "Figures"
        Figures_dir.mkdir(exist_ok=True)

        df_lifetime_IGBT_dir = sim_dir / "df_lifetime_IGBT"
        df_lifetime_IGBT_dir.mkdir(exist_ok=True)

        df_lifetime_Diode_dir = sim_dir / "df_lifetime_Diode"
        df_lifetime_Diode_dir.mkdir(exist_ok=True)

        df_electrical_dir = sim_dir / "df_electrical"
        df_electrical_dir.mkdir(exist_ok=True)

        df_lifetime_IGBT_MC_dir = sim_dir / "df_lifetime_IGBT_MC"
        df_lifetime_IGBT_MC_dir.mkdir(exist_ok=True)

        df_lifetime_Diode_MC_dir = sim_dir / "df_lifetime_Diode_MC"
        df_lifetime_Diode_MC_dir.mkdir(exist_ok=True)

        return sim_dir, df_electrical_loss_dir, df_thermal_dir, df_lifetime_IGBT_dir,df_lifetime_Diode_dir,df_electrical_dir, Figures_dir, df_lifetime_IGBT_MC_dir, df_lifetime_Diode_MC_dir

    @staticmethod
    def rainflow_algorithm(temp, dt):
        """
        Compute ΔT, T_mean, t_on for each rainflow cycle of a temperature signal.

        Parameters
        ----------
        temp : 1D array
            Temperature signal (e.g. Tj_igbt) [K or °C].
        dt : float
            Time step [s] of the signal (e.g. 0.001).

        Returns
        -------
        dT : 1D array
            Cycle ranges ΔT.
        Tmean : 1D array
            Cycle mean temperature.
        t_on : 1D array
            Heating time per cycle [s], defined as |end_idx - start_idx| * dt.
        count : 1D array
            Cycle count (usually 0.5 or 1.0).
        """
        temp = np.asarray(temp, dtype=float)

        cycles = list(rainflow.extract_cycles(temp))
        if len(cycles) == 0:
            return (np.empty(0), np.empty(0), np.empty(0), np.empty(0))

        cycles = np.array(cycles, dtype=float)
        dT = cycles[:, 0]
        Tmean = cycles[:, 1]
        count = cycles[:, 2]
        starts = cycles[:, 3].astype(int)
        ends = cycles[:, 4].astype(int)

        # Option A: compute t_on via sample distance * dt
        thermal_cycle_period = np.abs(ends - starts) * dt

        # (equivalent to np.abs(ends - starts) / steps_per_sec)

        return dT, Tmean, thermal_cycle_period, count


    @staticmethod
    def cycles_to_failure_lesit(deltaT,  # ΔT_j   : array or scalar
                                Tmean,  # T_jm   : array or scalar (K)
                                thermal_cycle_period,  # : array or scalar (s)
                                A0,
                                A1,
                                T0_K,
                                lambda_K,  # T0_K, λ
                                alpha,
                                Ea_J,
                                kB_J_per_K,  # activation energy, Boltzmann
                                C,
                                gamma,
                                k_thickness):  # k_thickness for IGBT or diode

        """

        Parameters
        ----------
        deltaT : array
                Junction temperature swing per cycle ΔTj [K].
        Tmean : array
            Mean (medium) junction temperature per cycle Tjm [K].
        thermal_cycle_period : array
            Heating time per cycle ton [s]
        A0 : float
            Technology coefficient A0 [-]
        A1 : float
            Factor A1 for the low-ΔT extension [-].
        T0_K : float
            Reference temperature T0 for the low-ΔT extension [K].
        lambda_K : float
            Decay constant λ for the low-ΔT extension [K].
        alpha : float
            Coffin–Manson exponent α (typically negative) [-].
        Ea_J : float
            Activation energy Ea [J]. **Use Joules, not eV.**
        kB_J_per_K : float
            Boltzmann constant k_B [J/K].
        C : float
            Time-shape coefficient C [-] in the ton-scaling term.
        gamma : float
            Time-shape exponent γ [-] in the ton-scaling term.
        k_thickness : float
            Chip thickness factor kthickness [-], accounting for
            chip thickness / technology (e.g. 1.0, 0.65, 0.5, 0.33).

        Returns
        -------
        Nf : ndarray
            Cycles-to-failure estimate per cycle Nf [-].
        """

        # Make sure inputs are float64 and contiguous (good for numexpr)
        deltaT = np.ascontiguousarray(deltaT, dtype=np.float64)
        Tmean = np.ascontiguousarray(Tmean, dtype=np.float64)
        thermal_cycle_period = np.ascontiguousarray(thermal_cycle_period, dtype=np.float64)

        if np.any(Tmean <= 0):
            raise ValueError("Tmean contains 0 K or negative values, which is not physically possible.")

        # Arrhenius temperature factor: exp(Ea / (kB * Tmean))
        c_arrhenius = ne.evaluate("exp(Ea_J / (kB_J_per_K * Tmean))",
                                  local_dict=dict(Ea_J=Ea_J, kB_J_per_K=kB_J_per_K, Tmean=Tmean))

        # exp_low = exp( - (ΔT - T0_K) / λ )
        exp_low = ne.evaluate("exp(-(deltaT - T0_K) / lambda_K)",
                              local_dict=dict(deltaT=deltaT, T0_K=T0_K, lambda_K=lambda_K))

        Nf = ne.evaluate(
            "A0 * (A1 ** exp_low) * "
            "(deltaT ** (alpha - exp_low)) * "
            "c_arrhenius * "
            "((C + thermal_cycle_period**gamma) / (C + 2.0**gamma)) * "
            "k_thickness",
            local_dict=dict(A0=A0, A1=A1, alpha=alpha, C=C,
                            gamma=gamma, k_thickness=k_thickness, deltaT=deltaT, exp_low=exp_low,
                            c_arrhenius=c_arrhenius, thermal_cycle_period=thermal_cycle_period))
        return Nf


    @staticmethod
    def miners_rule(Nf, count, Is):
        Nf = np.asarray(Nf, dtype=float)
        count = np.asarray(count, dtype=float)

        # Optional: filter out invalid entries
        mask = (Nf > 0) & np.isfinite(Nf) & np.isfinite(count)
        Nf = Nf[mask]
        count = count[mask]
        # Damage sum: D = Σ (count_i / Nf_i)
        D = np.sum(count / Nf)
        # Equivalent full cycles to failure: Nf_eq = 1 / D
        Nf_eq = np.inf if D == 0 else 1.0 / D

        mission_seconds_total = len(Is)
        seconds_per_year = 365 * 24 * 3600
        lifetime_years = (Nf_eq*mission_seconds_total)/(seconds_per_year)

        # ---- From here: get LC_year and Nf_target ----
        # Yearly life consumption
        LC_year = 1.0 / lifetime_years  # because LC_year * lifetime_years = 1

        # Equivalent static: cycles per year (Table IV uses 50 Hz → *50)
        cycles_per_year = seconds_per_year * 50.0

        # Nf_target in cycles (this is what you want)
        Nf_target_MC = cycles_per_year / LC_year  # same as cycles_per_year * lifetime_years


        return Nf_eq, lifetime_years, Nf_target_MC

    @staticmethod
    def read_datafames(df_dir):
        df_dir = Path(df_dir)

        # If df_dir is relative, make it relative to this file's directory
        if not df_dir.is_absolute():
            base_dir = Path(__file__).resolve().parent
            df_dir = base_dir / df_dir

        if not df_dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {df_dir}")

        all_files = sorted(df_dir.glob("df_*.parquet"))
        if not all_files:
            raise ValueError(f"No parquet files matching 'df_*.parquet' in {df_dir}")

        df_list = [pd.read_parquet(f) for f in all_files]
        full_df = pd.concat(df_list, ignore_index=True)
        return full_df

    @staticmethod
    def check_igbt_diode_temp_limits(Tj_igbt, Tj_diode,max_IGBT_temperature,max_Diode_temperature):

        if  (np.max(Tj_igbt) > max_IGBT_temperature):
            raise ValueError(f"IGBT junction temperature exceeded! "
                             f"Max temperature is {np.max(Tj_igbt - 273.15)} Celsius.")

        if  (np.max(Tj_diode) > max_Diode_temperature):
            raise ValueError(f"Diode  junction temperature exceeded! "
                             f"Max temperature is {np.max(Tj_diode - 273.15)} Celsius.")


    @staticmethod
    def residual_deltaT_MC(deltaT, Nf_target, Tmean, thermal_cycle_period,
                           A0, A1, T0_K, lambda_K, alpha, Ea_J, kB_J_per_K,
                           C, gamma, k_thickness):

        # Arrhenius temperature factor: exp(Ea / (kB * Tmean))
        c_arrhenius = ne.evaluate("exp(Ea_J / (kB_J_per_K * Tmean))",
                                  local_dict=dict(Ea_J=Ea_J, kB_J_per_K=kB_J_per_K, Tmean=Tmean))

        # exp_low = exp( - (ΔT - T0_K) / λ )
        exp_low = ne.evaluate("exp(-(deltaT - T0_K) / lambda_K)",
                              local_dict=dict(deltaT=deltaT, T0_K=T0_K, lambda_K=lambda_K))

        Nf = ne.evaluate(
            "A0 * (A1 ** exp_low) * "
            "(deltaT ** (alpha - exp_low)) * "
            "c_arrhenius * "
            "((C + thermal_cycle_period**gamma) / (C + 2.0**gamma)) * "
            "k_thickness",
            local_dict=dict(A0=A0, A1=A1, alpha=alpha, C=C,
                            gamma=gamma, k_thickness=k_thickness, deltaT=deltaT, exp_low=exp_low,
                            c_arrhenius=c_arrhenius, thermal_cycle_period=thermal_cycle_period))

        return Nf - Nf_target

    @staticmethod
    def normal_distributio_MC(Tj_mean_MC, deltaT_MC, thermal_cycle_period_MC, A0, A1, T0_K, lambda_K, alpha, Ea_J,
                              kB_J_per_K, C, gamma, k_thickness, normal_distribution, number_of_samples):

        """
        Create Monte Carlo samples for ALL stress parameters and ALL
        lifetime-model parameters using a normal distribution.

        Parameters are interpreted as the *nominal values* around which
        the samples are drawn.
        """

        def normal_distribution_function(variable, normal_distribution, number_of_samples):
            sigma = normal_distribution * abs(variable)
            samples = np.random.normal(variable, sigma, number_of_samples)
            return samples

        # --------------------------------------------
        # Generate samples for equivalent stress values
        # --------------------------------------------

        Tj_mean_samples = normal_distribution_function(Tj_mean_MC, normal_distribution, number_of_samples)
        deltaT_samples = normal_distribution_function(deltaT_MC, normal_distribution, number_of_samples)
        thermal_period_samples = normal_distribution_function(thermal_cycle_period_MC, normal_distribution,
                                                              number_of_samples)

        # --------------------------------------------
        # Generate samples for lifetime-model parameters
        # --------------------------------------------
        A0_samples = normal_distribution_function(A0, normal_distribution, number_of_samples)
        A1_samples = normal_distribution_function(A1, normal_distribution, number_of_samples)
        T0_K_samples = normal_distribution_function(T0_K, normal_distribution, number_of_samples)
        lambda_K_samples = normal_distribution_function(lambda_K, normal_distribution, number_of_samples)
        alpha_samples = normal_distribution_function(alpha, normal_distribution, number_of_samples)
        C_samples = normal_distribution_function(C, normal_distribution, number_of_samples)
        gamma_samples = normal_distribution_function(gamma, normal_distribution, number_of_samples)
        k_th_samples = normal_distribution_function(k_thickness, normal_distribution, number_of_samples)

        Ea_J_samples = np.full(number_of_samples, Ea_J)
        kB_samples = np.full(number_of_samples, kB_J_per_K)

        return {"Tj_mean": Tj_mean_samples, "deltaT": deltaT_samples, "thermal_period": thermal_period_samples,
                "A0": A0_samples, "A1": A1_samples, "T0_K": T0_K_samples, "lambda_K": lambda_K_samples,
                "alpha": alpha_samples, "Ea_J": Ea_J_samples, "kB_J_per_K": kB_samples, "C": C_samples,
                "gamma": gamma_samples, "k_thickness": k_th_samples}
    # ==================================================================================
    # ==================================================================================
    #  PEARL v2 additions
    # ==================================================================================
    # ==================================================================================

    # ---------------------------------------------------------------- modulation -------
    @staticmethod
    def modulation_amplitude(Vs, V_dc, inverter_phases, single_phase_inverter_topology):
        """Peak of the phase-voltage reference divided by the voltage the leg can swing around
        its midpoint (Vdc/2 for a three-phase or half-bridge leg, Vdc for a full bridge)."""
        Vs = np.asarray(Vs, dtype=np.float64); V_dc = np.asarray(V_dc, dtype=np.float64)
        if int(inverter_phases) == 1 and single_phase_inverter_topology == "full":
            return np.sqrt(2.0) * Vs / V_dc
        return np.sqrt(2.0) * Vs / (V_dc / 2.0)

    @staticmethod
    def check_modulation(ma, inverter_phases, modulation_scheme):
        """Linear-modulation limit: 2/sqrt(3) with SVPWM (three-phase), 1 otherwise."""
        limit = 2.0 / np.sqrt(3.0) if (int(inverter_phases) == 3 and modulation_scheme == "svm") else 1.0
        worst = float(np.max(ma))
        if worst > limit + 1e-9:
            raise ValueError(f"Modulation amplitude {worst:.3f} exceeds the linear limit {limit:.3f} "
                             f"(increase V_dc, lower Vs, or use 'svm' for three-phase).")
        return limit

    @staticmethod
    def duty_cycle(theta_v, ma, zero_sequence):
        """Local duty cycle of the upper switch, theta_v = omega*t + phi.
        zero_sequence=True: centred SVPWM (min/max zero-sequence injection)."""
        va = ma * np.sin(theta_v)
        if zero_sequence:
            vb = ma * np.sin(theta_v - 2.0 * np.pi / 3.0)
            vc = ma * np.sin(theta_v + 2.0 * np.pi / 3.0)
            va = va - 0.5 * (np.maximum(np.maximum(va, vb), vc) + np.minimum(np.minimum(va, vb), vc))
        return np.clip(0.5 * (1.0 + va), 0.0, 1.0)

    # ------------------------------------------------------------------- losses -------
    @staticmethod
    def loss_waveforms_physical(Is, phi, V_dc, ma, Np, zero_sequence, f_sw, t_on, t_off, Err_D, I_ref, V_ref,
                                R_IGBT, V_0_IGBT, R_D, V_0_D):
        """
        Switching-period-averaged instantaneous losses of the upper IGBT and upper diode of one
        leg over ONE fundamental period (rows = operating points, Np samples per period).
            i(t)  = sqrt(2) * Is * sin(omega t)                (phase current)
            d(t)  = duty cycle of the upper switch
            IGBT  (i > 0): f_sw * 0.5*Vdc*i*(t_on+t_off)  +  d * (V0*i + R*i^2)
            Diode (i < 0): f_sw * Err*(|i|/I_ref)*(Vdc/V_ref) +  d * (V0D*|i| + RD*i^2)
        The period averages equal the textbook average-loss formulas.
        """
        Is = np.asarray(Is, dtype=np.float64)[:, None]; phi = np.asarray(phi, dtype=np.float64)[:, None]
        V_dc = np.asarray(V_dc, dtype=np.float64)[:, None]; ma = np.asarray(ma, dtype=np.float64)[:, None]
        th = 2.0 * np.pi * np.arange(Np)[None, :] / Np
        i = np.sqrt(2.0) * Is * np.sin(th)
        ip, im = np.maximum(i, 0.0), np.maximum(-i, 0.0)
        d = Calculation_functions_class.duty_cycle(th + phi, ma, zero_sequence)
        P_I = 0.5 * V_dc * (t_on + t_off) * f_sw * ip + d * (V_0_IGBT * ip + R_IGBT * ip ** 2)
        P_D = f_sw * Err_D * (im / I_ref) * (V_dc / V_ref) + d * (V_0_D * im + R_D * im ** 2)
        return P_I, P_D

    @staticmethod
    def loss_waveforms_legacy(Is, phi, V_dc, Np, M, f_sw, t_on, t_off, Err_D, I_ref, V_ref, R_IGBT, V_0_IGBT, R_D, V_0_D):
        """PEARL v1 loss formulas over one period (kept only to reproduce old results)."""
        CFc = Calculation_functions_class
        Is = np.asarray(Is, dtype=np.float64)[:, None]; phi = np.asarray(phi, dtype=np.float64)[:, None]
        V_dc = np.asarray(V_dc, dtype=np.float64)[:, None]
        shape = (Is.shape[0], Np)
        omega = 1.0
        t = 2.0 * np.pi * np.arange(Np)[None, :] / Np
        m = CFc.Instantaneous_modulation(M=M, omega=omega, t=t, phi=phi)
        is_I, is_D = CFc.IGBT_and_diode_current(Is=Is, t=t, m=m, omega=omega)
        is_I = np.ascontiguousarray(np.broadcast_to(is_I, shape)); is_D = np.ascontiguousarray(np.broadcast_to(is_D, shape))
        Vf = np.ascontiguousarray(np.broadcast_to(V_dc, shape))
        pf = np.ascontiguousarray(np.broadcast_to(np.abs(np.cos(phi)), shape))
        sI, sD = CFc.Switching_losses(V_dc=Vf, is_I=is_I, t_on=t_on, t_off=t_off, f_sw=f_sw, is_D=is_D,
                                      I_ref=I_ref, V_ref=V_ref, Err_D=Err_D)
        cI, cD = CFc.Conduction_losses(is_I=is_I, R_IGBT=R_IGBT, V_0_IGBT=V_0_IGBT, M=M, pf=pf, is_D=is_D,
                                       R_D=R_D, V_0_D=V_0_D)
        return np.maximum(sI + cI, 0.0), np.maximum(sD + cD, 0.0)

    # ------------------------------------------------------- operating-point bins ----
    @staticmethod
    def bin_operating_points(Is, phi, V_dc, ma, resolution):
        """Group seconds with (almost) the same operating point.
        resolution = (dIs [A], dphi [rad]) or None (only exact duplicates are merged).
        Returns dict with representative Is, phi, V_dc, ma per bin and 'index' (second -> bin)."""
        Is = np.asarray(Is, dtype=np.float64); phi = np.asarray(phi, dtype=np.float64)
        V_dc = np.asarray(V_dc, dtype=np.float64); ma = np.asarray(ma, dtype=np.float64)
        if resolution is None:
            _, first, inv = np.unique(np.stack([Is, phi, V_dc, ma], 1), axis=0, return_index=True, return_inverse=True)
            rIs, rphi = Is[first], phi[first]
        else:
            dI, dphi = resolution
            cols = [np.rint(Is / dI), np.rint((phi + np.pi) / dphi), np.rint(V_dc / 0.5), np.rint(ma / 1e-4)]
            cols = [c.astype(np.int64) - c.min().astype(np.int64) for c in cols]
            dims = [int(c.max()) + 1 for c in cols]
            if np.prod([float(x) for x in dims]) < 2 ** 62:
                _, first, inv = np.unique(np.ravel_multi_index(cols, dims), return_index=True, return_inverse=True)
            else:
                _, first, inv = np.unique(np.stack(cols, 1), axis=0, return_index=True, return_inverse=True)
            rIs = np.rint(Is[first] / dI) * dI
            rphi = np.rint((phi[first] + np.pi) / dphi) * dphi - np.pi
        inv = inv.ravel()
        return {"Is": rIs, "phi": rphi, "V_dc": V_dc[first], "ma": ma[first],
                "index": inv.astype(np.int32 if first.size < 2 ** 31 else np.int64), "n": int(first.size)}

    # ------------------------------------------------------------- cycle counting -----
    @staticmethod
    def periodic_cycles(R, period, eps=1e-9):
        """
        Rainflow cycles of ONE closed period of a periodic signal, vectorised over rows.
        For a closed history, rainflow = repeatedly remove the neighbouring pair of turning points
        with the smallest range (each removed pair is one full cycle); the last pair is the
        max/min cycle. Returns dT, offset (cycle mid-temperature minus period mean), t_branch, count
        per period; shape (rows, K). t_branch: heating time (min -> max) for the max/min cycle,
        duration of the own branch for inner loops.
        """
        R = np.asarray(R, dtype=np.float64)
        N, Np = R.shape
        dt = period / Np
        s = np.sign(np.roll(R, -1, axis=1) - R); s[s == 0] = 1.0
        is_rev = s != np.roll(s, 1, axis=1)
        n_rev = np.count_nonzero(is_rev, axis=1)
        active = (R.max(axis=1) - R.min(axis=1)) > eps
        K = max(1, int(n_rev[active].max()) // 2) if active.any() else 1
        dT = np.zeros((N, K)); off = np.zeros((N, K)); ton = np.full((N, K), dt); cnt = np.zeros((N, K))
        for n in np.unique(n_rev[active]):
            rows = np.flatnonzero(active & (n_rev == n))
            pos = np.nonzero(is_rev[rows])[1].reshape(rows.size, n)
            val = R[rows[:, None], pos]
            ar = np.arange(rows.size); m = n; c = 0
            while m > 2:
                rng = np.abs(np.roll(val, -1, axis=1) - val)
                j = rng.argmin(axis=1); j2 = (j + 1) % m
                a, b = val[ar, j], val[ar, j2]
                dT[rows, c] = np.abs(b - a); off[rows, c] = 0.5 * (a + b)
                ton[rows, c] = ((pos[ar, j2] - pos[ar, j]) % Np) * dt; cnt[rows, c] = 1.0
                keep = (j[:, None] + 2 + np.arange(m - 2)[None, :]) % m
                val = np.take_along_axis(val, keep, 1); pos = np.take_along_axis(pos, keep, 1)
                m -= 2; c += 1
            hi, lo = val.max(axis=1), val.min(axis=1)
            p_hi, p_lo = pos[ar, val.argmax(axis=1)], pos[ar, val.argmin(axis=1)]
            dT[rows, c] = hi - lo; off[rows, c] = 0.5 * (hi + lo)
            ton[rows, c] = np.maximum((p_hi - p_lo) % Np, 1) * dt; cnt[rows, c] = 1.0
        return dT, off, ton, cnt

    @staticmethod
    def rainflow_turning_points(x, dt=1.0, gate=0.0):
        """Rainflow (ASTM E1049) on a long series. Turning points are found with numpy; an optional
        hysteresis gate [K] drops reversals smaller than `gate`.
        Returns dT, mean, t (= |i_end - i_start| * dt), count, i_start, i_end."""
        x = np.asarray(x, dtype=np.float64)
        empty = (np.empty(0),) * 4 + (np.empty(0, dtype=np.int64),) * 2
        if x.size < 2:
            return empty
        keep = np.r_[True, np.diff(x) != 0]
        idx = np.flatnonzero(keep); xv = x[idx]
        if xv.size < 2:
            return empty
        d = np.diff(xv)
        turn = np.flatnonzero(np.sign(d[1:]) != np.sign(d[:-1])) + 1
        rev = np.r_[0, turn, xv.size - 1]
        ridx, rval = idx[rev], xv[rev]
        if gate > 0 and rval.size > 2:
            ki, kv = [0], [rval[0]]
            for i in range(1, rval.size):
                v = rval[i]
                if len(kv) >= 2 and (v - kv[-1]) * (kv[-1] - kv[-2]) > 0:
                    ki[-1], kv[-1] = i, v
                elif abs(v - kv[-1]) >= gate or i == rval.size - 1:
                    ki.append(i); kv.append(v)
            ridx, rval = ridx[ki], np.asarray(kv)
        if rval.size == 2:
            cyc = np.array([[abs(rval[1] - rval[0]), 0.5 * (rval[0] + rval[1]), 0.5, 0, 1]])
        else:
            cyc = np.array([c for c in rainflow.extract_cycles(rval)], dtype=np.float64)
        if cyc.size == 0:
            return empty
        i1, i2 = ridx[cyc[:, 3].astype(int)], ridx[cyc[:, 4].astype(int)]
        return cyc[:, 0], cyc[:, 1], np.abs(i2 - i1) * dt, cyc[:, 2], i1, i2

    # ------------------------------------------------------------------ damage --------
    @staticmethod
    def effective_ton(ton, rule, ton_min=0.04):
        """'semikron': t_on = max(t_on, 40 ms), AN 21-001 (no further Nf increase below 40 ms).
        'raw': unchanged."""
        if rule == "semikron":
            return np.maximum(ton, ton_min)
        if rule == "raw":
            return np.asarray(ton, dtype=np.float64)
        raise ValueError("ton_rule must be 'semikron' or 'raw'")

    @staticmethod
    def damage_per_cycle(dT, Tjm, ton, lt, k_thickness, ton_rule, ton_min=0.04):
        """1/Nf for each cycle (0 for dT <= 0). lt = a bound model from Lifetime_models (callable),
        or (v1 style) a dict of Semikron parameters used with k_thickness / ton_rule."""
        dT = np.asarray(dT, dtype=np.float64); Tjm = np.asarray(Tjm, dtype=np.float64)
        ton = np.broadcast_to(np.asarray(ton, dtype=np.float64), dT.shape)
        out = np.zeros(dT.shape)
        ok = dT > 1e-9
        if ok.any() and callable(lt):                 # a model from Lifetime_models (applies its own t_on limits)
            Nf = lt(dT[ok], Tjm[ok], ton[ok])
            good = np.isfinite(Nf) & (Nf > 0)
            res = np.zeros(Nf.shape); res[good] = 1.0 / Nf[good]
            out[ok] = res
            return out
        if ok.any():
            Nf = Calculation_functions_class.cycles_to_failure_lesit(
                deltaT=dT[ok], Tmean=Tjm[ok],
                thermal_cycle_period=Calculation_functions_class.effective_ton(ton[ok], ton_rule, ton_min),
                k_thickness=k_thickness, **lt)
            good = np.isfinite(Nf) & (Nf > 0)
            res = np.zeros(Nf.shape); res[good] = 1.0 / Nf[good]
            out[ok] = res
        return out

    @staticmethod
    def lifetime_years(damage, mission_seconds):
        """Miner's rule: the mission causes `damage`; failure at total damage 1."""
        return np.inf if damage <= 0 else mission_seconds / damage / (365 * 24 * 3600)

    # ------------------------------------------------- cycle cells (compact storage) --
    @staticmethod
    def cycle_cells(dT, Tjm, ton, count, steps=(0.25, 1.0, 0.25)):
        """Aggregate cycles into cells (dT, Tjm, log10 t_on). Returns dict of arrays:
        count, and count-weighted mean dT, Tjm, ton for every non-empty cell."""
        dT = np.asarray(dT, float).ravel(); Tjm = np.asarray(Tjm, float).ravel()
        ton = np.asarray(ton, float).ravel(); count = np.asarray(count, float).ravel()
        m = count != 0
        dT, Tjm, ton, count = dT[m], Tjm[m], ton[m], count[m]
        if dT.size == 0:
            return {k: np.zeros(0) for k in ("count", "dT", "Tjm", "ton")}
        sd, sT, st = steps
        i1 = np.floor(dT / sd).astype(np.int64)
        i2 = np.floor((Tjm - 150.0) / sT).astype(np.int64)
        i3 = np.floor((np.log10(np.maximum(ton, 1e-6)) + 6.0) / st).astype(np.int64)
        key = (i1 * 2048 + np.clip(i2, 0, 2047)) * 512 + np.clip(i3, 0, 511)
        u, inv = np.unique(key, return_inverse=True)
        w = np.bincount(inv, weights=count)
        wa = np.abs(np.bincount(inv, weights=np.abs(count)))
        return {"count": w,
                "dT": np.bincount(inv, weights=np.abs(count) * dT) / wa,
                "Tjm": np.bincount(inv, weights=np.abs(count) * Tjm) / wa,
                "ton": 10 ** (np.bincount(inv, weights=np.abs(count) * np.log10(np.maximum(ton, 1e-6))) / wa)}

    @staticmethod
    def merge_cells(cell_list, steps=(0.25, 1.0, 0.25)):
        parts = [c for c in cell_list if c["count"].size]
        if not parts:
            return {k: np.zeros(0) for k in ("count", "dT", "Tjm", "ton")}
        cat = {k: np.concatenate([c[k] for c in parts]) for k in ("count", "dT", "Tjm", "ton")}
        return Calculation_functions_class.cycle_cells(cat["dT"], cat["Tjm"], cat["ton"], cat["count"], steps)

    @staticmethod
    def accumulate_matrix(H, x, y, w, x_edges, y_edges):
        """Add weights w to a regular 2-D histogram H (rainflow matrix)."""
        ix = np.floor((x - x_edges[0]) / (x_edges[1] - x_edges[0])).astype(np.int64)
        iy = np.floor((y - y_edges[0]) / (y_edges[1] - y_edges[0])).astype(np.int64)
        nx, ny = x_edges.size - 1, y_edges.size - 1
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (w != 0)
        H += np.bincount(ix[ok] * ny + iy[ok], weights=w[ok], minlength=nx * ny).reshape(nx, ny)

    # ---------------------------------------------------------- 50 Hz + slow damage ---
    @staticmethod
    def fast_cycle_damage(bin_index, Tj_mean, scale, cycles, f0, lt, k_thickness, ton_rule, ton_min,
                          matrix_edges, block=1 << 20):
        """
        Damage of the 50 Hz cycles, second by second.
        cycles = (dT, offset, t_branch, count) per bin (from periodic_cycles); scale = loss
        scaling per second (temperature coupling, 1 = off). Returns per-second damage (float32),
        total damage, rainflow matrices (counts, damage) and compact cycle cells for Monte Carlo.
        """
        CFc = Calculation_functions_class
        dTb, offb, tonb, cntb = cycles
        N = Tj_mean.size
        x_edges, y_edges = matrix_edges
        H_cnt = np.zeros((x_edges.size - 1, y_edges.size - 1)); H_dmg = np.zeros_like(H_cnt)
        D = np.empty(N, dtype=np.float32); total = 0.0; cells = []
        for s in range(0, N, block):
            e = min(N, s + block); ib = bin_index[s:e]; Tm = Tj_mean[s:e]; sc = scale[s:e]
            Db = np.zeros(e - s)
            for c in range(dTb.shape[1]):
                cnt = f0 * cntb[ib, c]
                if not np.any(cnt):
                    continue
                dT = dTb[ib, c] * sc
                Tjm = Tm + offb[ib, c] * sc
                ton = tonb[ib, c]
                dmg = cnt * CFc.damage_per_cycle(dT, Tjm, ton, lt, k_thickness, ton_rule, ton_min)
                Db += dmg
                CFc.accumulate_matrix(H_cnt, dT, Tjm - 273.15, cnt, x_edges, y_edges)
                CFc.accumulate_matrix(H_dmg, dT, Tjm - 273.15, dmg, x_edges, y_edges)
                cells.append(CFc.cycle_cells(dT, Tjm, ton, cnt))
            D[s:e] = Db; total += float(Db.sum())
        return {"per_second": D, "total": total, "matrix_counts": H_cnt, "matrix_damage": H_dmg,
                "cells": CFc.merge_cells(cells)}

    @staticmethod
    def slow_cycle_damage(series, ripple_up, ripple_dn, main_dmg_fn, lt, k_thickness, ton_rule, ton_min, gate,
                          matrix_edges):
        """
        Mission-profile cycles from the 1-s mean junction temperature `series`.
        A slow cycle in the real signal runs from a ripple trough at its cold end to a ripple peak
        at its hot end, and it 'absorbs' one 50 Hz cycle there (superposition rule of rainflow).
        So: damage = damage(merged cycle) - damage(absorbed 50 Hz cycle), never below zero.
        ripple_up/ripple_dn: ripple max/min above/below the mean at each series point.
        main_dmg_fn(idx): damage of one main 50 Hz cycle at series points idx.
        """
        CFc = Calculation_functions_class
        dT0, _, t, cnt, i1, i2 = CFc.rainflow_turning_points(series, 1.0, gate)
        x_edges, y_edges = matrix_edges
        H_cnt = np.zeros((x_edges.size - 1, y_edges.size - 1)); H_dmg = np.zeros_like(H_cnt)
        if dT0.size == 0:
            z = np.zeros(0)
            return {"total": 0.0, "n_cycles": 0, "matrix_counts": H_cnt, "matrix_damage": H_dmg,
                    "cells_merged": CFc.cycle_cells(z, z, z, z), "cells_absorbed": CFc.cycle_cells(z, z, z, z),
                    "end_index": np.zeros(0, int), "damage": z}
        hot = np.where(series[i1] >= series[i2], i1, i2); cold = np.where(series[i1] >= series[i2], i2, i1)
        hi = series[hot] + ripple_up[hot]; lo = series[cold] + ripple_dn[cold]
        dT, Tjm = hi - lo, 0.5 * (hi + lo)
        merged = CFc.damage_per_cycle(dT, Tjm, t, lt, k_thickness, ton_rule, ton_min)
        absorbed = 0.5 * (main_dmg_fn(hot) + main_dmg_fn(cold))
        dmg = np.maximum(merged - absorbed, 0.0) * cnt
        CFc.accumulate_matrix(H_cnt, dT, Tjm - 273.15, cnt, x_edges, y_edges)
        CFc.accumulate_matrix(H_dmg, dT, Tjm - 273.15, dmg, x_edges, y_edges)
        # absorbed 50 Hz cycles as negative counts (for Monte Carlo re-evaluation)
        a_dT = np.r_[ripple_up[hot] - ripple_dn[hot], ripple_up[cold] - ripple_dn[cold]]
        a_T = np.r_[series[hot] + 0.5 * (ripple_up[hot] + ripple_dn[hot]), series[cold] + 0.5 * (ripple_up[cold] + ripple_dn[cold])]
        return {"total": float(dmg.sum()), "n_cycles": int(dT.size), "matrix_counts": H_cnt, "matrix_damage": H_dmg,
                "cells_merged": CFc.cycle_cells(dT, Tjm, t, cnt),
                "cells_absorbed": CFc.cycle_cells(a_dT, a_T, np.full(a_dT.size, ton_min), -0.5 * np.r_[cnt, cnt]),
                "end_index": np.maximum(i1, i2), "damage": dmg}

    # -------------------------------------------------------------- Monte Carlo -------
    @staticmethod
    def monte_carlo_lifetime(cells, mission_seconds, lt, k_thickness, ton_rule, ton_min, sigma, n_samples,
                             seed=0, calibration=1.0, batch=250):
        """
        Lifetime distribution from the stored cycle cells (no new simulation needed).
        Each sample draws ONE set of uncertain factors applied to every cycle:
          stress : dT * (1 + e_dT),  Tjm + e_Tjm [K]
          model  : coefficient * (1 + e_A), slope exponent * (1 + e_alpha)   (any lifetime model)
        sigma = dict of 1-sigma values {"dT": rel, "Tjm_K": abs, "A": rel, "alpha": rel} ("A0" = old name of "A").
        `calibration` = exact damage / cell damage (removes the small cell-rounding error).
        Returns lifetimes in years (array of n_samples).
        """
        rng = np.random.default_rng(seed)
        e_dT = rng.normal(0, sigma.get("dT", 0.0), n_samples)
        e_T = rng.normal(0, sigma.get("Tjm_K", 0.0), n_samples)
        e_A0 = rng.normal(0, sigma.get("A0", 0.0), n_samples)
        e_al = rng.normal(0, sigma.get("alpha", 0.0), n_samples)
        cnt, dT, Tjm = cells["count"], cells["dT"], cells["Tjm"]
        if cnt.size == 0:
            return np.full(n_samples, np.inf)
        e_A = rng.normal(0, sigma.get("A", sigma.get("A0", 0.0)), n_samples) if "A" in sigma else e_A0
        D = np.empty(n_samples)
        if callable(lt):                              # any model from Lifetime_models
            ton = cells["ton"]
            for s in range(0, n_samples, batch):
                e = min(n_samples, s + batch)
                dTs = dT[None, :] * (1.0 + e_dT[s:e, None])
                Ts = Tjm[None, :] + e_T[s:e, None]
                Nf = lt(dTs, Ts, ton[None, :], pert={"A": e_A[s:e, None], "alpha": e_al[s:e, None]})
                with np.errstate(divide="ignore", invalid="ignore"):
                    d = np.where(np.isfinite(Nf) & (Nf > 0) & (dTs > 0), cnt[None, :] / Nf, 0.0)
                D[s:e] = np.maximum(d.sum(axis=1), 0.0) * calibration
            with np.errstate(divide="ignore"):
                return np.where(D > 0, mission_seconds / D / (365 * 24 * 3600), np.inf)
        ton = Calculation_functions_class.effective_ton(cells["ton"], ton_rule, ton_min)
        tf = (lt["C"] + ton ** lt["gamma"]) / (lt["C"] + 2.0 ** lt["gamma"])
        for s in range(0, n_samples, batch):
            e = min(n_samples, s + batch)
            dTs = dT[None, :] * (1.0 + e_dT[s:e, None])
            Ts = Tjm[None, :] + e_T[s:e, None]
            al = lt["alpha"] * (1.0 + e_al[s:e, None])
            beta = np.exp(-(dTs - lt["T0_K"]) / lt["lambda_K"])
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                Nf = (lt["A0"] * (1.0 + e_A0[s:e, None]) * lt["A1"] ** beta * dTs ** (al - beta)
                      * np.exp(lt["Ea_J"] / (lt["kB_J_per_K"] * Ts)) * tf[None, :] * k_thickness)
                d = np.where(np.isfinite(Nf) & (Nf > 0) & (dTs > 0), cnt[None, :] / Nf, 0.0)
            D[s:e] = np.maximum(d.sum(axis=1), 0.0) * calibration
        with np.errstate(divide="ignore"):
            return np.where(D > 0, mission_seconds / D / (365 * 24 * 3600), np.inf)

    @staticmethod
    def weibull_summary(samples):
        """Two-parameter Weibull fit of lifetime samples; returns shape, scale and B1/B10/B50 [years]."""
        from scipy.stats import weibull_min
        x = np.asarray(samples, dtype=np.float64); x = x[np.isfinite(x) & (x > 0)]
        if x.size < 10:
            return {"shape": np.nan, "scale": np.nan, "B1": np.nan, "B10": np.nan, "B50": np.nan}
        k, _, lam = weibull_min.fit(x, floc=0)
        Bq = lambda q: lam * (-np.log(1.0 - q)) ** (1.0 / k)
        return {"shape": float(k), "scale": float(lam), "B1": float(Bq(0.01)), "B10": float(Bq(0.10)),
                "B50": float(Bq(0.50))}

    @staticmethod
    def series_system_weibull(shape, scale, n_units):
        """n identical Weibull units in series (any unit failing = system failure):
        same shape, scale * n^(-1/shape)."""
        sc = scale * n_units ** (-1.0 / shape)
        Bq = lambda q: sc * (-np.log(1.0 - q)) ** (1.0 / shape)
        return {"shape": shape, "scale": sc, "B1": Bq(0.01), "B10": Bq(0.10), "B50": Bq(0.50), "n_units": n_units}

    # --------------------------------------------------------------- folders ----------
    @staticmethod
    def create_result_folders(base="Results"):
        """Results/Simulation_N/{Figures, Data}; N increments automatically."""
        base_dir = Path(base); base_dir.mkdir(parents=True, exist_ok=True)
        nums = [int(p.name.split("_")[1]) for p in base_dir.iterdir()
                if p.is_dir() and p.name.startswith("Simulation_") and p.name.split("_")[1].isdigit()]
        sim_dir = base_dir / f"Simulation_{max(nums) + 1 if nums else 1}"
        (sim_dir / "Figures").mkdir(parents=True, exist_ok=True)
        (sim_dir / "Data").mkdir(parents=True, exist_ok=True)
        return sim_dir, sim_dir / "Data", sim_dir / "Figures"
