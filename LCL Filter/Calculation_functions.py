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
                           modulation_scheme,
                           inverter_phases,
                           single_phase_inverter_topology):

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
        inverter_phases  : {"1","3"}
            Number of phases. If 3, Vs is interpreted as PHASE RMS (i.e., V_ll/sqrt(3)).
        single_phase_inverter_topology : {"half","full"}
            Inverter topology (affects Vs limit for single-phase).

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
        S[mnz] = np.abs(P[mnz]) / np.abs(pf[mnz])

        # Is[i] = S[i] / Vs[i]
        with np.errstate(divide='ignore', invalid='ignore'):
            Is[mnz] = S[mnz] / (Vs[mnz] if inverter_phases == 1 else (3.0 * Vs[mnz]))

        # phi[i] = ± arccos(abs(pf[i]))
        phi[mneg] = -np.arccos(abspf[mneg])  # inductive
        phi[mpos] = np.arccos(abspf[mpos])  # capacitive

        # Q[i] = ± sqrt(S[i]^2 - P[i]^2) for pf != 0
        # (Note: numerical noise can make the radicand slightly negative; clip at 0.)
        rad = np.clip(S[mnz] ** 2 - P[mnz] ** 2, 0.0, None)
        root = np.sqrt(rad)
        idx_mnz = np.where(mnz)[0]
        Q[idx_mnz[mneg[mnz]]] = -root[mneg[mnz]]
        Q[idx_mnz[mpos[mnz]]] = root[mpos[mnz]]

        return Vs, Is, phi, V_dc, pf, M, S



    @staticmethod
    def synthetic_profile(THD_input, dt, rms_values, phi, omega,
                          harmonic_orders=None,
                          harmonic_weights=None,
                          harmonic_phases=None):
        """
        Generic synthetic waveform generator for voltage or current.

        Parameters
        ----------
        THD_input : float or array_like
            THD of the waveform [-]
        dt : float
            Time step [s]
        rms_values : array_like
            RMS values per sample (current or voltage)
        phi : array_like
            Fundamental phase angle per sample [rad]
        omega : float
            Fundamental angular frequency [rad/s]
        harmonic_orders : array_like, optional
            Harmonic orders
        harmonic_weights : array_like, optional
            Relative harmonic RMS weights
        harmonic_phases : array_like, optional
            Harmonic phase angles [rad]

        Returns
        -------
        waveform_final : ndarray
            Synthetic instantaneous waveform
        """

        if harmonic_orders is None:
            harmonic_orders = np.array([5, 7, 11, 13])

        if harmonic_weights is None:
            harmonic_weights = np.array([0.50, 0.30, 0.15, 0.05], dtype=float)

        harmonic_weights = np.asarray(harmonic_weights, dtype=float)
        harmonic_weights = harmonic_weights / np.sum(harmonic_weights)

        if harmonic_phases is None:
            harmonic_phases = np.zeros(len(harmonic_orders))

        rms_values = np.asarray(rms_values)
        phi = np.asarray(phi)

        samples_per_main_sample = int(round(1 / dt))
        t = np.arange(0, len(rms_values), dt)

        rms_expanded = np.repeat(rms_values, samples_per_main_sample)
        phi_expanded = np.repeat(phi, samples_per_main_sample)

        n = min(len(t), len(rms_expanded), len(phi_expanded))
        t = t[:n]
        rms_expanded = rms_expanded[:n]
        phi_expanded = phi_expanded[:n]

        # Fundamental
        waveform_fundamental = np.sqrt(2) * rms_expanded * np.sin(omega * t + phi_expanded)

        # THD expansion
        if np.isscalar(THD_input):
            THD_expanded = np.full_like(rms_expanded, THD_input, dtype=float)
        else:
            THD_input = np.asarray(THD_input)
            THD_expanded = np.repeat(THD_input, samples_per_main_sample)[:n]

        # Harmonics
        h_total_rms = THD_expanded * rms_expanded
        h_rms = h_total_rms[:, np.newaxis] * np.sqrt(harmonic_weights)[np.newaxis, :]

        waveform_harmonics_total = np.zeros_like(t, dtype=float)

        for k, (h, ph) in enumerate(zip(harmonic_orders, harmonic_phases)):
            component = np.sqrt(2) * h_rms[:, k] * np.sin(h * omega * t + ph)
            waveform_harmonics_total += component

        waveform_total = waveform_fundamental + waveform_harmonics_total

        def rms_per_main_sample(signal):
            n_full = len(signal) // samples_per_main_sample
            trimmed = signal[:n_full * samples_per_main_sample]
            reshaped = trimmed.reshape(n_full, samples_per_main_sample)
            return np.sqrt(np.mean(reshaped ** 2, axis=1))

        waveform_total_rms = rms_per_main_sample(waveform_total)
        ratio = waveform_total_rms / rms_values[:len(waveform_total_rms)]
        ratio_expanded = np.repeat(ratio, samples_per_main_sample)

        waveform_final = waveform_total[:len(ratio_expanded)] / ratio_expanded
        return waveform_final
