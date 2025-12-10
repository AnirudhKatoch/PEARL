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
    def check_max_capacitor_current_limit(Max_voltage_datasheet_cap, Max_current_datasheet_cap, V_per_cap, I_per_cap):

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

