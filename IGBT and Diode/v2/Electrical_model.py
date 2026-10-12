"""
Electrical_model.py  (PEARL v3)
===============================

Losses are needed for ONE fundamental period (20 ms at 50 Hz) per operating point, because
within one mission-profile second the inverter repeats the same period 50 times. Operating
points that are (almost) equal are grouped into bins, so each distinct point is computed once.

compute_bin_losses   : loss waveforms for every operating-point bin (new, used by mother_function)
compute_IGBT_and_Diode_power_losses : PEARL v1 function (1 s at dt), kept for compatibility
"""

import numpy as np
from functools import lru_cache
from Calculation_functions import Calculation_functions_class as CF


def compute_bin_losses(params, bins, start=0, stop=None):
    """
    Loss waveforms over one period for bins[start:stop].
    Returns P_I, P_D with shape (n_bins, Np) in W.
    """
    sl = slice(start, stop)
    Is, phi, V_dc, ma = bins["Is"][sl], bins["phi"][sl], bins["V_dc"][sl], bins["ma"][sl]
    p = params
    if p.loss_model == "physical":
        return CF.loss_waveforms_physical(
            Is=Is, phi=phi, V_dc=V_dc, ma=ma, Np=p.Np, zero_sequence=p.zero_sequence,
            f_sw=p.f_sw, t_on=p.t_on, t_off=p.t_off, Err_D=p.Err_D, I_ref=p.I_ref, V_ref=p.V_ref,
            R_IGBT=p.R_IGBT, V_0_IGBT=p.V_0_IGBT, R_D=p.R_D, V_0_D=p.V_0_D)
    if p.loss_model == "legacy":
        return CF.loss_waveforms_legacy(
            Is=Is, phi=phi, V_dc=V_dc, Np=p.Np, M=p.M, f_sw=p.f_sw, t_on=p.t_on, t_off=p.t_off,
            Err_D=p.Err_D, I_ref=p.I_ref, V_ref=p.V_ref, R_IGBT=p.R_IGBT, V_0_IGBT=p.V_0_IGBT,
            R_D=p.R_D, V_0_D=p.V_0_D)
    raise ValueError("loss_model must be 'physical' or 'legacy'")


# ----------------------------------------------------------------------------------------
# PEARL v1 (unchanged behaviour): 1-second loss waveform at time step dt
# ----------------------------------------------------------------------------------------

@lru_cache(maxsize=20000)
def _compute_IGBT_and_Diode_power_losses_cached(Is, phi, V_dc, pf, dt, M, omega, t_on, t_off, f_sw, I_ref, V_ref,
                                                Err_D, R_IGBT, V_0_IGBT, R_D, V_0_D):
    t = np.arange(0.0, 1, dt, dtype=np.float64)
    m = CF.Instantaneous_modulation(M=M, omega=omega, t=t, phi=phi)
    is_I, is_D = CF.IGBT_and_diode_current(Is=Is, t=t, m=m, omega=omega)
    P_sw_I, P_sw_D = CF.Switching_losses(V_dc=V_dc, is_I=is_I, t_on=t_on, t_off=t_off, f_sw=f_sw, is_D=is_D,
                                         I_ref=I_ref, V_ref=V_ref, Err_D=Err_D)
    P_con_I, P_con_D = CF.Conduction_losses(is_I=is_I, R_IGBT=R_IGBT, V_0_IGBT=V_0_IGBT, M=M, pf=pf, is_D=is_D,
                                            R_D=R_D, V_0_D=V_0_D)
    P_I = np.ascontiguousarray(np.maximum(P_sw_I + P_con_I, 0.0), dtype=np.float64)
    P_D = np.ascontiguousarray(np.maximum(P_sw_D + P_con_D, 0.0), dtype=np.float64)
    for arr in (P_I, P_D, is_I, is_D, P_sw_I, P_sw_D, P_con_I, P_con_D):
        arr.setflags(write=False)
    return P_I, P_D, is_I, is_D, P_sw_I, P_sw_D, P_con_I, P_con_D


def compute_IGBT_and_Diode_power_losses(Is, phi, V_dc, pf, dt, M, omega, t_on, t_off, f_sw, I_ref, V_ref, Err_D,
                                        R_IGBT, V_0_IGBT, R_D, V_0_D):
    r = lambda x: float(round(float(x), 10))
    return _compute_IGBT_and_Diode_power_losses_cached(r(Is), r(phi), r(V_dc), r(pf), r(dt), r(M), r(omega),
                                                       r(t_on), r(t_off), r(f_sw), r(I_ref), r(V_ref), r(Err_D),
                                                       r(R_IGBT), r(V_0_IGBT), r(R_D), r(V_0_D))
