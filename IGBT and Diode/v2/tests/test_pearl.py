"""
PEARL v3 test suite.  Run:  pytest tests   (or: python tests/test_pearl.py)
Each test checks one claim of the method against an independent reference. Total < 1 min.
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")
import numpy as np
import rainflow
from Input_parameters import Input_parameters_class
from Calculation_functions import Calculation_functions_class as CF
from Thermal_model import ThermalNetwork, ripple_from_losses
from Electrical_model import compute_bin_losses
from Lifetime_models import Semikron2024, fit_cma, build_lifetime_model
from mother_function import mother_function
from main import synthetic_pv_profile


def _params(**kw):
    return Input_parameters_class(np.full(3, 20000.0), np.zeros(3), np.full(3, 298.15), **kw)


def test_loss_model_matches_textbook_averages():
    p = _params(modulation_scheme="spwm", Vs_rms=190.0)
    for deg in (0, 45, 90, 135, 180):
        phi, Is, ma, Vdc = np.radians(deg), 40.0, 0.9, 600.0
        PI, PD = CF.loss_waveforms_physical(np.array([Is]), np.array([phi]), np.array([Vdc]), np.array([ma]), 20000, False,
                                            p.f_sw, p.t_on, p.t_off, p.Err_D, p.I_ref, p.V_ref, p.R_IGBT, p.V_0_IGBT, p.R_D, p.V_0_D)
        Ipk, c = np.sqrt(2) * Is, np.cos(phi)
        TI = (p.V_0_IGBT * Ipk * (1 / (2 * np.pi) + ma * c / 8) + p.R_IGBT * Ipk ** 2 * (1 / 8 + ma * c / (3 * np.pi))
              + Ipk * Vdc * (p.t_on + p.t_off) * p.f_sw / (2 * np.pi))
        TD = (p.V_0_D * Ipk * (1 / (2 * np.pi) - ma * c / 8) + p.R_D * Ipk ** 2 * (1 / 8 - ma * c / (3 * np.pi))
              + p.f_sw * p.Err_D * Ipk * Vdc / (np.pi * p.I_ref * p.V_ref))
        assert abs(PI.mean() / TI - 1) < 1e-3 and abs(PD.mean() / TD - 1) < 1e-3


def test_ripple_fft_equals_time_simulation():
    p = _params(N_parallel=1)
    net = ThermalNetwork.from_params(p)
    bins = CF.bin_operating_points(p.Is[:1], p.phi[:1], p.V_dc[:1], p.ma[:1], None)
    p.Np = 400
    PI, PD = compute_bin_losses(p, bins)
    RI, _ = ripple_from_losses(net, PI, PD, 50.0)
    z0 = net.steady_state(PI.mean(), PD.mean(), 298.15)
    Y, _ = net.simulate(np.tile(PI[0], 100), np.tile(PD[0], 100), 298.15, 0.02 / 400, z0)
    last = Y[0][-400:]
    assert abs(np.ptp(last) - np.ptp(RI)) < 0.02 * np.ptp(RI)


def test_interval_mean_is_exact_average():
    p = _params()
    net = ThermalNetwork.from_params(p)
    rng = np.random.default_rng(1)
    PI = rng.uniform(0, 40, 50); PD = rng.uniform(0, 10, 50); Ta = np.full(50, 300.0)
    z0 = net.initial_state(300.0)
    Ym, _ = net.interval_mean(PI, PD, Ta, 1.0, z0)
    Yf, _ = net.simulate(np.repeat(PI, 2000), np.repeat(PD, 2000), np.repeat(Ta, 2000), 1.0 / 2000, z0)
    assert np.max(np.abs(Ym[0] - Yf[0].reshape(50, 2000).mean(1))) < 0.01


def test_periodic_rainflow_matches_rainflow_package():
    rng = np.random.default_rng(3)
    for _ in range(50):
        k = np.arange(40); x = sum(rng.normal() * np.sin(2 * np.pi * h * k / 40 + rng.uniform(0, 6)) / h for h in (1, 2, 3))
        dT, off, ton, cnt = CF.periodic_cycles(x[None, :], 0.02)
        cyc = np.array(list(rainflow.extract_cycles(np.tile(x, 300))))
        mid = (cyc[:, 3] > 40 * 50) & (cyc[:, 4] < 40 * 250)
        ref = np.sum(cyc[mid, 2] * cyc[mid, 0] ** 4) / 200
        assert abs(np.sum(cnt * dT ** 4) / ref - 1) < 0.02


def test_damage_matches_bruteforce_reference():
    P, Q, T = synthetic_pv_profile(1)
    a, b = 11 * 3600, 11 * 3600 + 300
    Np = 200; dt = 0.02 / Np
    p = Input_parameters_class(P[a:b], Q[a:b], T[a:b], N_parallel=1, r_sink=0.72, sink_mass=0.018, Np=Np, op_resolution=None)
    net = ThermalNetwork.from_params(p)
    bins = CF.bin_operating_points(p.Is, p.phi, p.V_dc, p.ma, None)
    PIb, PDb = compute_bin_losses(p, bins)
    PI = np.tile(PIb[bins["index"]], (1, 50)).ravel(); PD = np.tile(PDb[bins["index"]], (1, 50)).ravel()
    z0 = net.steady_state(PIb[bins["index"][0]].mean(), PDb[bins["index"][0]].mean(), p.T_env[0])
    Y, _ = net.simulate(PI, PD, np.repeat(p.T_env, 50 * Np), dt, z0)
    Nf = Semikron2024(k_thickness=p.k_thickness).bind("IGBT")
    dT, Tm, ton, cnt, _, _ = CF.rainflow_turning_points(Y[0], dt)
    ref = np.sum(cnt * CF.damage_per_cycle(dT, Tm, ton, Nf, None, "semikron"))
    r = mother_function(P[a:b], Q[a:b], T[a:b], save=False, plot=False, verbose=False, N_parallel=1, r_sink=0.72,
                        sink_mass=0.018, Np=Np, op_resolution=None, MC_samples=50)
    assert abs(r["IGBT"]["damage_total"] / ref - 1) < 0.03


def test_binning_and_resolution_converged():
    P, Q, T = synthetic_pv_profile(1)
    kw = dict(save=False, plot=False, verbose=False, MC_samples=50)
    r1 = mother_function(P, Q, T, op_resolution=(0.02, 1e-3), **kw)
    r2 = mother_function(P, Q, T, op_resolution=None, **kw)
    assert abs(r1["IGBT"]["lifetime_years"] / r2["IGBT"]["lifetime_years"] - 1) < 0.01


def test_fit_cma_recovers_parameters():
    rng = np.random.default_rng(0)
    pts = [(d, t, 3e14 * d ** -5.2 * np.exp(1e-19 / (1.380649e-23 * (t + 273.15))) * np.exp(rng.normal(0, 0.03)))
           for d in (40, 60, 80, 100) for t in (100, 125, 150)]
    f = fit_cma(pts)
    assert abs(f["n"] - 5.2) < 0.1 and abs(f["Ea_J"] / 1e-19 - 1) < 0.05


def test_user_curves_reproduce_formula():
    sk = Semikron2024()
    curves = {T: {"dTj": list(np.arange(5, 125, 5.0)),
                  "Nf": list(sk.bind("IGBT")(np.arange(5, 125, 5.0), T + 273.15 - np.arange(5, 125, 5.0) / 2, np.full(24, 2.0)))}
              for T in (100, 125, 150)}
    m = build_lifetime_model({"model": "curves", "temperature": "Tj_max", "curves": {"IGBT": curves, "Diode": "IGBT"},
                              "t_on_s": 2.0, "ton_correction": {"C": 1.0, "gamma": -0.75}, "t_on_min": 0.04})
    dT = np.array([20.0, 40.0, 70.0]); Tjm = np.array([380.0, 390.0, 360.0]); ton = np.array([0.04, 1.0, 10.0])
    ratio = m.bind("IGBT")(dT, Tjm, ton) / sk.bind("IGBT")(dT, Tjm, ton)
    assert np.all(np.abs(ratio - 1) < 0.05)


def test_four_quadrant_charging_loads_the_diode():
    n = 600
    kw = dict(save=False, plot=False, verbose=False, init="periodic", MC_samples=50)
    dis = mother_function(np.full(n, 20000.0), np.zeros(n), np.full(n, 298.15), **kw)["per_second"]
    chg = mother_function(np.full(n, -20000.0), np.zeros(n), np.full(n, 298.15), **kw)["per_second"]
    assert dis["P_mean_I"][-1] > dis["P_mean_D"][-1] and chg["P_mean_D"][-1] > chg["P_mean_I"][-1]


def test_celsius_input_is_rejected():
    try:
        Input_parameters_class(np.zeros(3), np.zeros(3), np.full(3, 25.0))
    except ValueError:
        return
    raise AssertionError("T_env in degC was accepted")


if __name__ == "__main__":
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t0 = time.time(); fn(); print(f"PASS {name} ({time.time() - t0:.1f} s)")
