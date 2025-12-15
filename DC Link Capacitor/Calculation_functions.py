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
                           modulation_scheme):

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

        pf = np.zeros_like(P, dtype=float)
        Is = np.zeros_like(P, dtype=float)  # [A] Inverter RMS current
        phi = np.zeros_like(P, dtype=float)  # [rad] Phase angle

        S = np.sqrt(P ** 2 + Q ** 2)  # [VA] Inverter RMS apparent power

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

        #if inverter_phases == 1:
        #    if single_phase_inverter_topology == "full":
        #        Vs_theoretical = (M * V_dc) / np.sqrt(2.0)
        #    elif single_phase_inverter_topology == "half":
         #       Vs_theoretical = (M * V_dc) / (2.0 * np.sqrt(2.0))
        #elif inverter_phases == 3:
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

        inverter_phases = 3

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
    def check_max_capacitor_voltage_and_current_limit(Max_voltage_datasheet_cap, Max_current_datasheet_cap, V_per_cap, I_per_cap):

        """
        Check capacitor voltage and current against datasheet limits

        Parameters
        ----------
        Max_voltage_datasheet_cap : float
            Maximum allowable capacitor voltage from datasheet [V]
        Max_current_datasheet_cap : float
            Maximum allowable capacitor RMS current from datasheet [A]
        V_per_cap : array
            Per-capacitor operating voltage per sample [V]
        I_per_cap : array
            Per-capacitor operating RMS current per sample [A]

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the per-capacitor voltage exceeds the rated voltage
        ValueError
            If the per-capacitor RMS current exceeds the rated current
        """

        # ---- IGBT CHECK ----
        if np.any(V_per_cap > Max_voltage_datasheet_cap):
            raise ValueError(
                f"Capacitor voltage limit exceeded: "
                f"max allowed {Max_voltage_datasheet_cap} A"
            )

        if np.any(I_per_cap > Max_current_datasheet_cap):
            raise ValueError(
                f"Capacitor current limit exceeded: "
                f"max allowed {Max_current_datasheet_cap} A"
            )

    @staticmethod
    def check_max_capacitor_temperature_limit(Max_temperature_cap_dict, T_core, Rated_voltage_datasheet_cap, V_per_cap):

        """
        Check capacitor core temperature against allowable derated limits
        based on voltage ratio from the datasheet graph.

        Parameters
        ----------
        Max_temperature_cap_dict : dict
            Dictionary mapping V_ratio -> Max allowed temperature [K].
        T_core : float or array
            Capacitor core temperature [K].
        V_per_cap : float or array
            Actual per-capacitor DC voltage [V].
        Rated_voltage_datasheet_cap : float
            Capacitor rated DC voltage [V].

        Raises
        ------
        ValueError
            If T_core exceeds allowed temperature for given V_ratio.
        """

        # Compute voltage ratio V/V_rated
        V_ratio = V_per_cap / float(Rated_voltage_datasheet_cap)

        # Extract breakpoint arrays from dictionary

        V_keys = np.array(list(Max_temperature_cap_dict.keys()))
        T_vals = np.array(list(Max_temperature_cap_dict.values()))  # in K

        # We need V_keys in **ascending** order for np.interp
        idx_sort = np.argsort(V_keys)
        V_keys_sorted = V_keys[idx_sort]
        T_vals_sorted = T_vals[idx_sort]


        # Interpolate
        # Values outside range get clamped to nearest endpoint automatically.
        T_allowed = np.interp(V_ratio, V_keys_sorted, T_vals_sorted)




        # Check each point
        mask_exceed = T_core > T_allowed
        if np.any(mask_exceed):
            i = np.where(mask_exceed)[0][0]
            raise ValueError(
                f"Capacitor temperature exceeds derated limit at index {i}:\n"
                f"  T_core = {T_core[i]:.2f} K\n"
                f"  Allowed = {T_allowed[i]:.2f} K\n"
                f"  V_ratio = {V_ratio[i]:.3f}"
            )



    @staticmethod
    def capacitor_RMS_ripple_current(Is, M, phi):

        """
        Compute DC-link capacitor RMS ripple current (Kolar analytical model)

        Parameters
        ----------
        Is : array
            RMS inverter phase current per sample [A]
        M : array or float
            Modulation index [-]
        phi : array
            Phase angle between inverter voltage and current per sample [rad]

        Returns
        -------
        Idcl : array
            RMS DC-link capacitor ripple current per sample [A]


        """

        Idcl = (np.sqrt(2) * Is) * np.sqrt(
            M * ((np.sqrt(3) / (4 * np.pi)) + (((np.sqrt(3) / (np.pi) - ((9 * M) / 16)) * np.cos(phi) ** 2))))

        return Idcl

    @staticmethod
    def core_temperature_calibration_factor(I_per_cap, ESR_eff, V_per_cap, minimum_insulation_resistance, T_core,T_env,Thermal_resistance):

        """
        Compute calibration factor for capacitor core-temperature model.


        Parameters
        ----------
        I_per_cap : float
            RMS ripple current per capacitor [A].
        ESR_eff : float
            Effective ESR used for power-loss calculation [Ω].
        V_per_cap : float
            Per-capacitor operating DC voltage [V].
        minimum_insulation_resistance : float
            Leakage resistance used to estimate leakage current [Ω].
        T_core : float
            Desired capacitor core/case temperature at rated conditions [°C].
        T_env : float
            Ambient temperature used in calibration [°C].

        Returns
        -------
        calibration_factor : float
            Scaling factor 'k' in the core-temperature model:
                T_core = T_env + k * Thermal_resistance * P_losses

        """

        P_ripple = (I_per_cap ** 2) * ESR_eff  # ripple-loss per cap

        I_leak = V_per_cap / minimum_insulation_resistance
        P_leak = I_leak * V_per_cap  # leakage-loss per cap

        P_losses = P_ripple + P_leak  # total losses per cap

        calibration_factor_core_temp = (T_core - T_env) / (Thermal_resistance * P_losses)

        return calibration_factor_core_temp

    @staticmethod
    def core_temperature_calculationsI_cap(I_per_cap, ESR_eff, V_per_cap, minimum_insulation_resistance, T_env,
                                               Thermal_resistance, calibration_factor_core_temp):
        """
            Compute capacitor core temperature using calibrated thermal model.

            Parameters
            ----------
            I_per_cap : float or array
                RMS ripple current per capacitor [A].
            ESR_eff : float or array
                Effective ESR used for ripple-loss calculation [Ω].
            V_per_cap : float or array
                Per-capacitor operating DC voltage [V].
            minimum_insulation_resistance : float
                Leakage resistance used to estimate leakage current [Ω].
            T_env : float or array
                Ambient temperature [°C].
            Thermal_resistance : float
                Thermal resistance of the capacitor [°C/W].
            calibration_factor_core_temp : float
                Temperature scaling factor 'k' obtained from calibration:
                    T_core = T_env + k * Thermal_resistance * P_losses

            Returns
            -------
            T_core : float or array
                Estimated capacitor core temperature [°C].

            """

        P_ripple = (I_per_cap ** 2) * ESR_eff  # ripple-loss per cap

        I_leak = V_per_cap / minimum_insulation_resistance
        P_leak = I_leak * V_per_cap  # leakage-loss per cap

        P_losses = P_ripple + P_leak  # total losses per cap

        T_core = T_env + calibration_factor_core_temp * Thermal_resistance * P_losses  # Capacitor temperature

        return T_core

    @staticmethod
    def Nichion_lifetime_model(L_r, T_r, T, Delta_t_r, I_r, K):

        """
           Compute capacitor lifetime using the given formula.

           Parameters
           ----------
           L_r : float
               Rated lifetime [hours].
           T_r : float
               Rated temperature [K].
           T : Array
               Core temperature [K].
           Delta_t_r : float
               Rated temperature rise. [K]
           I_r : float
               Ripple current. [A]
           K : float
               Model Constant. [-]

           Returns
           -------
           L : float
               Estimated lifetime [hours].
           """

        L = (L_r * (2 ** ((T_r - T) / 10)) * (2 ** (1 - (((Delta_t_r * ((I_r / T) ** 2))) / K))))

        return L


    @staticmethod
    def Rubycon_lifetime_model(L_r, T_r, T, Delta_t_r, Delta_t, V_r, V):

        """
        Compute capacitor lifetime using the combined temperature, ripple,
        and voltage-acceleration model.

        Parameters
        ----------
        L_r : float
            Rated lifetime [hours]
        T_r : float
            Rated temperature [K]
        T : array
            Operating temperature [K]
        Delta_t_r : float
            Rated ripple temperature rise [K]
        Delta_t : array
            Actual ripple temperature rise [K]
        V_r : float
            Rated voltage [V]
        V : array
            Operating voltage [V]

        Returns
        -------
        L : array
            Estimated lifetime [hours]
        """

        L = (L_r * (2 ** ((T_r - T) / 10)) * (2 ** ((Delta_t_r / 10) - (Delta_t / 10))) * ((V_r / V) ** 2.5))

        return L

    @staticmethod
    def Panasonic_lifetime_model(L_r, T_r, T_a):

        """
        Compute capacitor lifetime using the simple temperature-acceleration model.

        Parameters
        ----------
        L_r : float
            Rated lifetime [hours]
        T_r : float
            Rated temperature [K]
        T_a : array
            Core temperature [K]

        Returns
        -------
        L : array
            Estimated lifetime [hours]
        """

        L = L_r * (2 ** ((T_r - T_a) / 10))

        return L


    @staticmethod
    def Cornell_Dubilier_lifetime_model(L_r, T_r, T, V_r, F, V):

        """

        Parameters
        ----------
        L_r : float
            Rated lifetime at rated temperature and rated voltage [hours].
        T_r : float
            Rated temperature used for lifetime specification [K].
        T : float or array
            Actual operating capacitor core temperature [K].
        V_r : float
            Rated voltage from datasheet [V].
        V : float or array
            Actual operating voltage applied to the capacitor [V].
        F : float
            Cornell Dubilier Model constant.

        Returns
        -------
        L : float or array
            Estimated lifetime under operating temperature and voltage [hours].
        """

        L = (L_r * (2 ** ((T_r - T) / 10)) * ((V_r * F) / V) ** 8)

        return L


    @staticmethod
    def Faratronic_lifetime_model(L_r, T_r, V_r, V):
        """
        Compute capacitor lifetime using voltage-acceleration only.

        Parameters
        ----------
        L_r : float
            Rated lifetime [hours].
        T_r : float
            Rated temperature used for lifetime specification [K]
        V_r : float
            Rated voltage [V].
        V : float or array
            Actual operating voltage [V].

        Returns
        -------
        L : float or array
            Estimated lifetime [hours].
        """
        L = L_r * T_r * ((V_r / V) ** 9)

        return L

    @staticmethod
    def Generic_Arrhenius_lifetime_model(L_0, V, V_0, n, Ea, kB, T, T_0):
        """
        Compute lifetime using voltage and temperature acceleration.

        Parameters
        ----------
        L_0 : float
            Reference lifetime. [hours]
        V_0 : float
            Reference voltage. [V]
        n : float
            Constant dependent on type of capacitor. [-]
        Ea : float
            Activation energy. [J]
        kB : float
            Boltzmann constant. [J/K]
        T_0 : float
            Reference temperature [K].
        V : float or array
            Operating voltage [V].
        T : float or array
            Operating temperature [K].

        Returns
        -------
        L : float or array
            Estimated lifetime.
        """

        L = (L_0 * ((V / V_0) ** (-n)) * np.exp((Ea / kB) * (1 / T - 1 / T_0)))

        return L

    @staticmethod
    def get_lifetime_from_graph(T_core, V_ratio, lifetime_dict):

        """
        Vectorized 2D interpolation: lifetime(T, V_ratio) from datasheet graph.

        Parameters
        ----------
        T_core : float or array [K]
        V_ratio : float or array
        lifetime_dict : dict
            Keys = temperatures in K
            Values = {"V_ratio": [...], "L_hours": [...]}

        Returns
        -------
        L_hours : float or array
        """

        T = np.asarray(T_core)
        V = np.asarray(V_ratio)

        # ----- 1) Temperature axis keys -----
        T_keys = np.array(sorted(lifetime_dict.keys()))  # ascending

        # ----- 2) Helper: interpolate L along V_ratio for all temperature curves -----
        # Create matrix: each row corresponds to one temperature curve.
        all_L_at_T = []

        for T_key in T_keys:
            curve = lifetime_dict[T_key]
            V_curve = np.array(curve["V_ratio"])
            L_curve = np.array(curve["L_hours"])
            idx = np.argsort(V_curve)  # ensure V increasing for np.interp
            # Interpolate at ALL V_ratio values at once
            L_interp = np.interp(V, V_curve[idx], L_curve[idx], left=L_curve[idx][0], right=L_curve[idx][-1])
            all_L_at_T.append(L_interp)

        # Stack into array: shape = (n_temps, n_samples)
        L_vs_T = np.vstack(all_L_at_T)

        # ----- 3) Now interpolate in the temperature dimension -----
        # For each sample T[i], find the index where it would be inserted
        idx_high = np.searchsorted(T_keys, T)

        # Clamp extreme temperatures (below lowest or above highest)
        below = idx_high == 0
        above = idx_high == len(T_keys)

        # Prepare output
        L_out = np.zeros_like(T, dtype=float)

        # --- Case A: below lowest temperature ----
        L_out[below] = L_vs_T[0, below]

        # --- Case B: above highest temperature ---
        L_out[above] = L_vs_T[-1, above]

        # --- Case C: inside temperature range ---
        inside = ~(below | above)

        # Extract the surrounding temperatures
        iH = idx_high[inside]  # high index
        iL = iH - 1  # low index

        T_low = T_keys[iL]
        T_high = T_keys[iH]

        w = (T[inside] - T_low) / (T_high - T_low)

        L_low = L_vs_T[iL, inside]
        L_high = L_vs_T[iH, inside]

        L_out[inside] = L_low + w * (L_high - L_low)

        return L_out

    @staticmethod
    def miners_rule_lifetime(L_hours, Simulation_durations):
        """
        Apply Miner’s rule to an L_hours array over a mission profile.

        Parameters
        ----------
        L_hours : array
            Time-to-failure at each sample [hours].
        Simulation_durations : float
            Duration of each sample [s] (time resolution of profile).

        Returns
        -------
        D : float
            Damage fraction accumulated in one mission profile.
            Failure expected when D ≈ 1.
        T_profile_h : float
            Duration of one mission profile [hours].
        lifetime_equiv_h : float
            Equivalent total lifetime if this mission profile repeats
            continuously until failure.
        """

        L_hours = np.asarray(L_hours, dtype=float)

        # Time per sample in hours
        dt_h = Simulation_durations / 3600.0

        # Total profile duration
        T_profile_h = len(L_hours) * dt_h

        # Miner damage per sample
        damage_i = dt_h / L_hours  # Δt / L_i

        # Total damage for one pass of the profile
        D = np.sum(damage_i)

        # Equivalent lifetime if this profile is repeated until failure
        lifetime_equiv_h = T_profile_h / D

        lifetime_equiv_h = lifetime_equiv_h / (365 * 24)

        return D, lifetime_equiv_h

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
        dataframes_dir = sim_dir / "Dataframes"
        dataframes_dir.mkdir(exist_ok=True)

        Figures_dir = sim_dir / "Figures"
        Figures_dir.mkdir(exist_ok=True)


        return sim_dir, dataframes_dir, Figures_dir

    @staticmethod
    def normal_distribution_function(variable, normal_distribution, number_of_samples):
        sigma = normal_distribution * abs(variable)
        samples = np.random.normal(variable, sigma, number_of_samples)
        return samples

    @staticmethod
    def validate_lifetime_model(capacitor_type, model_name):

        ALLOWED_LIFETIME_MODELS = {
            "electrolytic": {
                "Nichion_lifetime",
                "Rubycon_lifetime",
                "Panasonic_lifetime",
                "Generic_Arrhenius_lifetime",
                "Graph_Based_lifetime",
            },
            "film": {
                "Cornell_Dubilier_lifetime",
                "Faratronic_lifetime",
                "Generic_Arrhenius_lifetime",
                "Graph_Based_lifetime",
            },
        }

        if capacitor_type not in ALLOWED_LIFETIME_MODELS:
            raise ValueError(
                f"Unknown capacitor_type '{capacitor_type}'. "
                f"Expected one of {list(ALLOWED_LIFETIME_MODELS.keys())}."
            )

        if model_name not in ALLOWED_LIFETIME_MODELS[capacitor_type]:
            raise ValueError(
                f"Lifetime model '{model_name}' is not valid for "
                f"capacitor type '{capacitor_type}'. "
                f"Allowed models are: "
                f"{sorted(ALLOWED_LIFETIME_MODELS[capacitor_type])}."
            )





