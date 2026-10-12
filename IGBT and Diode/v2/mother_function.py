"""
mother_function.py  (PEARL v3)
==============================

mother_function(P, Q, T_env, **overrides) -> results (dict)

  P, Q, T_env : 1-D arrays at the mission-profile resolution (default 1 s), T_env in Kelvin.
  overrides   : any attribute of Input_parameters_class, e.g. N_parallel=2, r_sink=0.72.

Pipeline (no time simulation at the switching or grid time scale anywhere):

  1. Operating points     each second -> (Is, phi, V_dc, modulation); near-equal seconds share a bin
  2. One period per bin   loss waveform over 20 ms  -> mean loss + ripple via Zth(j*omega)
                          -> the 50 Hz thermal cycles (dTj, offset, t_on) of that operating point
  3. Mean temperatures    thermal network driven by the mean losses, exact at the 1-s step
                          (+ optional electro-thermal coupling)
  4. Damage               50 Hz cycles: f cycles per second, known directly (no rainflow)
                          mission-profile cycles: rainflow on the 1-s mean temperature only
                          Semikron Danfoss power-cycle model + Miner's rule
  5. Uncertainty          Monte Carlo on the stored cycles -> Weibull, B10 for IGBT, diode,
                          switch and the whole inverter
  6. Save + plot          summary.json, hourly.parquet, rainflow matrices, Monte Carlo samples
"""

import json
import time
import warnings

import numpy as np
import pandas as pd

from Input_parameters import Input_parameters_class
from Calculation_functions import Calculation_functions_class as CF
from Electrical_model import compute_bin_losses
from Thermal_model import ThermalNetwork, ripple_from_losses
from Lifetime_models import build_lifetime_models


def _stack_cycles(parts, fill_t):
    """Concatenate per-chunk cycle arrays (dT, offset, t, count) padded to the same width."""
    K = max(c[0].shape[1] for c in parts)
    out = []
    for j in range(4):
        fill = fill_t if j == 2 else 0.0
        out.append(np.concatenate([np.pad(c[j], ((0, 0), (0, K - c[j].shape[1])), constant_values=fill)
                                   for c in parts]))
    return tuple(out)


def mother_function(P, Q, T_env, save=True, plot=None, verbose=True, **overrides):

    t_start = time.time()
    timing = {}

    # ---------------------------------------------------------------------------------- #
    # 0. Inputs
    # ---------------------------------------------------------------------------------- #
    p = Input_parameters_class(P=P, Q=Q, T_env=T_env, **overrides)
    N = p.Is.size
    dt = float(p.profile_dt)
    mission_seconds = N * dt
    f0 = float(p.f)
    lt = p.lifetime_parameters
    kth = (p.k_thickness["IGBT"], p.k_thickness["Diode"])
    names = ("IGBT", "Diode")
    plot = p.Plotting_flag if plot is None else plot
    timing["0_inputs"] = time.time() - t_start

    # ---------------------------------------------------------------------------------- #
    # 1. Operating-point bins
    # ---------------------------------------------------------------------------------- #
    t = time.time()
    bins = CF.bin_operating_points(p.Is, p.phi, p.V_dc, p.ma, p.op_resolution)
    idx, nb = bins["index"], bins["n"]
    timing["1_bins"] = time.time() - t

    # ---------------------------------------------------------------------------------- #
    # 2. One period per bin: losses, mean losses, ripple, 50 Hz cycles
    # ---------------------------------------------------------------------------------- #
    t = time.time()
    net = ThermalNetwork.from_params(p)
    Pm = np.zeros((nb, 2)); up = np.zeros((nb, 2)); dn = np.zeros((nb, 2))
    parts = {0: [], 1: []}
    for s in range(0, nb, 100_000):
        e = min(nb, s + 100_000)
        PI, PD = compute_bin_losses(p, bins, s, e)
        Pm[s:e, 0], Pm[s:e, 1] = PI.mean(axis=1), PD.mean(axis=1)
        RI, RD = ripple_from_losses(net, PI, PD, f0)
        for d, R in ((0, RI), (1, RD)):
            up[s:e, d], dn[s:e, d] = R.max(axis=1), R.min(axis=1)
            parts[d].append(CF.periodic_cycles(R, 1.0 / f0))
    cycles = {d: _stack_cycles(parts[d], 1.0 / f0 / p.Np) for d in (0, 1)}
    main_ton = {d: cycles[d][2][np.arange(nb), np.argmax(cycles[d][0], axis=1)] for d in (0, 1)}
    del parts
    timing["2_losses_ripple_cycles"] = time.time() - t

    # ---------------------------------------------------------------------------------- #
    # 3. Mean temperatures at the profile step (+ electro-thermal coupling)
    # ---------------------------------------------------------------------------------- #
    t = time.time()
    PmI, PmD = Pm[idx, 0], Pm[idx, 1]
    scale = [np.ones(N), np.ones(N)]
    coupled = bool(p.kappa_IGBT or p.kappa_Diode)
    for it in range(p.coupling_iterations if coupled else 1):
        PI_s, PD_s = PmI * scale[0], PmD * scale[1]
        if p.init == "ambient":
            z0 = net.initial_state(p.T_env[0])
        else:
            z0 = net.steady_state(PI_s[0], PD_s[0], p.T_env[0])
        if p.init == "periodic":
            for _ in range(2):
                _, z0 = net.interval_mean(PI_s, PD_s, p.T_env, dt, z0, p.quasi_static_tau)
        Y, _ = net.interval_mean(PI_s, PD_s, p.T_env, dt, z0, p.quasi_static_tau)
        if not coupled:
            break
        new = [np.clip(1.0 + p.kappa_IGBT * (Y[0] - p.kappa_T_ref), 0.5, 3.0),
               np.clip(1.0 + p.kappa_Diode * (Y[1] - p.kappa_T_ref), 0.5, 3.0)]
        change = max(np.max(np.abs(new[0] - scale[0])), np.max(np.abs(new[1] - scale[1])))
        if change < 1e-4:
            break
        scale = new
    y0 = net.W_out @ z0
    del PI_s, PD_s
    timing["3_mean_temperature"] = time.time() - t

    # ---------------------------------------------------------------------------------- #
    # 4.-5. Damage, lifetime and reliability for every selected lifetime model
    # ---------------------------------------------------------------------------------- #
    t = time.time()
    edges = (np.arange(0, 200 + p.matrix_bins[0], p.matrix_bins[0]),
             np.arange(-50, 250 + p.matrix_bins[1], p.matrix_bins[1]))
    steps_per_hour = max(1, int(round(3600.0 / dt)))
    hours = np.arange(0, N, steps_per_hour)
    hourly = {"t_start_h": hours * dt / 3600.0}
    per_second = {"Tj_mean_I": Y[0], "Tj_mean_D": Y[1], "T_case": Y[2].astype(np.float32),
                  "T_sink": Y[3].astype(np.float32),
                  "P_mean_I": (PmI * scale[0]).astype(np.float32), "P_mean_D": (PmD * scale[1]).astype(np.float32)}
    res = {"settings": {k: getattr(p, k) for k in ("loss_model", "junction_model", "ton_rule", "ton_min", "Np",
                                                   "op_resolution", "init", "slow_gate", "quasi_static_tau",
                                                   "kappa_IGBT", "kappa_Diode", "N_parallel", "profile_dt")},
           "mission_seconds": float(mission_seconds), "n_operating_bins": int(nb),
           "thermal_time_constants_s": net.time_constants.tolist(), "Rth_K_per_W": net.Rth.tolist()}

    # thermal quantities per device (the same for every lifetime model)
    dev = {}
    for d in (0, 1):
        up_s = up[idx, d] * scale[d]; dn_s = dn[idx, d] * scale[d]; Tm = Y[d]
        if p.init == "periodic":
            k0 = int(np.argmax(Tm))
            sec = np.r_[np.arange(k0, N), np.arange(0, k0), k0]
            series, s_up, s_dn, has = Tm[sec], up_s[sec], dn_s[sec], np.ones(sec.size, bool)
        else:
            sec = np.r_[0, np.arange(N)]                       # point 0 = state at t = 0
            series = np.r_[y0[d], Tm]
            has = np.ones(N + 1, bool); has[0] = p.init != "ambient"
            s_up = np.where(has, up_s[sec], 0.0); s_dn = np.where(has, dn_s[sec], 0.0)
        Tmax = (Tm + up_s).astype(np.float32)
        limit = p.max_IGBT_temperature if d == 0 else p.max_Diode_temperature
        over = float(np.mean(Tmax > limit))
        if over > 0:
            msg = (f"{names[d]}: junction temperature above {limit - 273.15:.0f} degC in {100 * over:.2f} % of the "
                   f"time (max {Tmax.max() - 273.15:.1f} degC). The device would not survive; lifetime is not meaningful.")
            if p.temperature_limit_action == "raise":
                raise ValueError(msg)
            warnings.warn(msg)
        dev[d] = dict(Tm=Tm, sec=sec, series=series, s_up=s_up, s_dn=s_dn, has=has, Tmax=Tmax, over=over,
                      dTj50=(up_s - dn_s).astype(np.float32))
        del up_s, dn_s

    models = build_lifetime_models(p.lifetime_model, p)
    res["lifetime_models"] = {}
    for mi, model in enumerate(models):
        key = model.name if model.name not in res["lifetime_models"] else f"{model.name}_{mi}"
        primary = mi == 0
        out = {"description": model.describe()}
        mc = {}
        for d in (0, 1):
            D_ = dev[d]
            Nf = model.bind(names[d])
            fast = CF.fast_cycle_damage(idx, D_["Tm"], scale[d], cycles[d], f0 * dt, Nf, None, p.ton_rule,
                                        p.ton_min, edges)

            def main_dmg(k, d=d, D_=D_, Nf=Nf):
                b = idx[D_["sec"][k]]
                return np.where(D_["has"][k], CF.damage_per_cycle(
                    D_["s_up"][k] - D_["s_dn"][k], D_["series"][k] + 0.5 * (D_["s_up"][k] + D_["s_dn"][k]),
                    main_ton[d][b], Nf, None, p.ton_rule, p.ton_min), 0.0)

            slow = CF.slow_cycle_damage(D_["series"], D_["s_up"], D_["s_dn"], main_dmg, Nf, None, p.ton_rule,
                                        0.5 / f0, p.slow_gate, edges)
            D_total = fast["total"] + slow["total"]
            cells = CF.merge_cells([fast["cells"], slow["cells_merged"], slow["cells_absorbed"]])
            cell_damage = float(np.sum(cells["count"] * CF.damage_per_cycle(cells["dT"], cells["Tjm"], cells["ton"],
                                                                            Nf, None, p.ton_rule, p.ton_min)))
            calib = (D_total / cell_damage) if cell_damage > 0 else 1.0
            mc[names[d]] = CF.monte_carlo_lifetime(cells, mission_seconds, Nf, None, p.ton_rule, p.ton_min, p.MC_sigma,
                                                   p.MC_samples if primary else p.MC_samples_other_models,
                                                   seed=p.MC_seed + d, calibration=calib)
            life = CF.lifetime_years(D_total, mission_seconds)
            out[names[d]] = {
                "lifetime_years": life,
                "lifetime_years_capped": min(life, p.IGBT_max_life if d == 0 else p.Diode_max_life),
                "damage_total": D_total, "damage_50Hz": fast["total"], "damage_mission_cycles": slow["total"],
                "share_50Hz": fast["total"] / D_total if D_total > 0 else np.nan,
                "n_mission_cycles": slow["n_cycles"],
                "Tj_max_C": float(D_["Tmax"].max() - 273.15), "Tj_mean_C": float(D_["Tm"].mean() - 273.15),
                "dTj_50Hz_max_K": float(D_["dTj50"].max()), "time_above_limit": D_["over"],
                "rainflow_matrix": {"dT_edges_K": edges[0], "Tjm_edges_C": edges[1],
                                    "counts_50Hz": fast["matrix_counts"], "damage_50Hz": fast["matrix_damage"],
                                    "counts_mission": slow["matrix_counts"], "damage_mission": slow["matrix_damage"]},
            }
            if primary:
                sfx = "I" if d == 0 else "D"
                per_second[f"Tj_max_{sfx}"] = D_["Tmax"]
                per_second[f"dTj_50Hz_{sfx}"] = D_["dTj50"]
                per_second[f"damage_50Hz_{sfx}"] = fast["per_second"]
                hourly[f"Tj_mean_{sfx}_C"] = np.add.reduceat(D_["Tm"], hours) / np.diff(np.r_[hours, N]) - 273.15
                hourly[f"Tj_max_{sfx}_C"] = np.maximum.reduceat(D_["Tmax"], hours) - 273.15
                hourly[f"damage_50Hz_{sfx}"] = np.add.reduceat(fast["per_second"].astype(np.float64), hours)
                h_slow = np.zeros(hours.size)
                if slow["damage"].size:
                    np.add.at(h_slow, D_["sec"][slow["end_index"]] // steps_per_hour, slow["damage"])
                hourly[f"damage_mission_{sfx}"] = h_slow
        mc["Switch"] = np.minimum(mc["IGBT"], mc["Diode"])
        rel = {k: CF.weibull_summary(v) for k, v in mc.items()}
        sw = rel["Switch"]
        rel["Inverter"] = CF.series_system_weibull(sw["shape"], sw["scale"], p.n_devices) if np.isfinite(sw["shape"]) else {}
        out["switch_lifetime_years"] = min(out["IGBT"]["lifetime_years"], out["Diode"]["lifetime_years"])
        out["reliability"] = rel
        res["lifetime_models"][key] = out
        if primary:
            res["lifetime_model"] = key
            res["IGBT"], res["Diode"] = out["IGBT"], out["Diode"]
            res["switch_lifetime_years"] = out["switch_lifetime_years"]
            res["reliability"] = rel
            res["monte_carlo_years"] = mc
    hourly["P_kW"] = np.add.reduceat(p.P, hours) / np.diff(np.r_[hours, N]) / 1e3
    hourly["T_amb_C"] = np.add.reduceat(p.T_env, hours) / np.diff(np.r_[hours, N]) - 273.15
    timing["4_5_damage_reliability"] = time.time() - t

    res["hourly"] = hourly
    res["per_second"] = per_second
    res["params"] = p
    res["network"] = net
    res["bins"] = bins

    # ---------------------------------------------------------------------------------- #
    # 6. Save and plot
    # ---------------------------------------------------------------------------------- #
    t = time.time()
    if save or plot:
        sim_dir, data_dir, fig_dir = CF.create_result_folders(p.results_dir)
        res["folders"] = {"simulation": str(sim_dir), "data": str(data_dir), "figures": str(fig_dir)}
    if save:
        summary = {k: v for k, v in res.items() if k in ("settings", "mission_seconds", "n_operating_bins",
                                                          "thermal_time_constants_s", "Rth_K_per_W",
                                                          "switch_lifetime_years", "reliability")}
        summary["lifetime_model"] = res["lifetime_model"]
        summary["lifetime_models"] = {m: {"description": o["description"],
                                          "IGBT": {k: v for k, v in o["IGBT"].items() if k != "rainflow_matrix"},
                                          "Diode": {k: v for k, v in o["Diode"].items() if k != "rainflow_matrix"},
                                          "switch_lifetime_years": o["switch_lifetime_years"],
                                          "reliability": o["reliability"]} for m, o in res["lifetime_models"].items()}
        for n_ in names:
            summary[n_] = {k: v for k, v in res[n_].items() if k != "rainflow_matrix"}
            np.savez_compressed(data_dir / f"rainflow_matrix_{n_}.npz", **res[n_]["rainflow_matrix"])
        pd.DataFrame(hourly).to_parquet(data_dir / "hourly.parquet", index=False)
        pd.DataFrame(mc).to_parquet(data_dir / "monte_carlo_lifetime_years.parquet", index=False)
        rows = []
        for m, o in res["lifetime_models"].items():
            for n_ in names:
                rows.append({"lifetime_model": m, "device": n_, "lifetime_years": o[n_]["lifetime_years"],
                             "B10_years": o["reliability"][n_]["B10"], "share_50Hz": o[n_]["share_50Hz"],
                             "damage_50Hz": o[n_]["damage_50Hz"], "damage_mission_cycles": o[n_]["damage_mission_cycles"]})
        pd.DataFrame(rows).to_csv(data_dir / "lifetime_models.csv", index=False)
        if p.save_per_second:
            pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in per_second.items()}).to_parquet(
                data_dir / "per_second.parquet", index=False, compression="zstd")
    timing["6_save"] = time.time() - t
    if plot:
        t = time.time()
        import Plotting_results as PR
        PR.plot_all(res, fig_dir)
        timing["7_plots"] = time.time() - t
    timing["total"] = time.time() - t_start
    res["timing_s"] = timing
    if save:
        summary["timing_s"] = timing
        (data_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=lambda o: o if not hasattr(o, "tolist") else o.tolist()))

    if verbose:
        r = res["reliability"]
        print(f"PEARL v3 [{res['lifetime_model']}]: {N} steps of {dt:g} s ({mission_seconds / 86400:.1f} days), {nb} operating-point bins, "
              f"{timing['total']:.1f} s")
        for n_ in names:
            R = res[n_]
            print(f"  {n_:5s} lifetime {R['lifetime_years']:10.3g} y | 50 Hz share {R['share_50Hz']:6.1%} | "
                  f"Tj max {R['Tj_max_C']:6.1f} C | B10 {r[n_]['B10']:.3g} y")
        if r["Inverter"]:
            print(f"  Switch B10 {r['Switch']['B10']:.3g} y | Inverter ({p.n_devices} devices) B10 {r['Inverter']['B10']:.3g} y")
        if len(res["lifetime_models"]) > 1:
            print("  lifetime model comparison (Miner lifetime | B10):")
            for m, o in res["lifetime_models"].items():
                print(f"    {m:22s} IGBT {o['IGBT']['lifetime_years']:9.3g} y | {o['reliability']['IGBT']['B10']:9.3g} y   "
                      f"Diode {o['Diode']['lifetime_years']:9.3g} y | {o['reliability']['Diode']['B10']:9.3g} y")
    return res
