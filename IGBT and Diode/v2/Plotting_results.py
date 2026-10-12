"""
Plotting_results.py  (PEARL v3)

plot_all(res, fig_dir) writes:
  01_mission_profile.png      P, Q, ambient (min/mean/max per block)
  02_temperatures.png         Tj mean / max over the whole mission + one day at 1-s resolution
  03_one_period.png           losses and junction temperature over one 20 ms period (hottest point)
  04_thermal_impedance.png    |Zth(jf)|: why the heat sink only sees the mean
  05_rainflow_matrices.png    cycle counts and damage vs (dTj, Tjm) for IGBT and diode
  06_damage.png               damage by cycle amplitude and accumulated damage over the mission
  07_reliability.png          Monte Carlo lifetime histograms + Weibull unreliability curves
  08_lifetime_models.png      lifetime and B10 per lifetime model (only if several are selected)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import weibull_min

from Calculation_functions import Calculation_functions_class as CF
from Thermal_model import ripple_from_losses
from Electrical_model import compute_bin_losses

C_I, C_D, C_S, C_A = "tab:blue", "tab:red", "tab:green", "0.4"


def _blocks(x, n_points=3000):
    x = np.asarray(x, dtype=np.float64)
    b = max(1, x.size // n_points)
    m = (x.size // b) * b
    v = x[:m].reshape(-1, b)
    return np.arange(v.shape[0]) * b, v.min(1), v.mean(1), v.max(1)


def _save(fig, fig_dir, name):
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/{name}", dpi=130)
    plt.close(fig)


def plot_mission_profile(res, fig_dir):
    p = res["params"]; dt = p.profile_dt
    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for a, x, lab, c in ((ax[0], p.P / 1e3, "P [kW]", C_I), (ax[1], p.Q / 1e3, "Q [kvar]", C_D),
                         (ax[2], p.T_env - 273.15, "ambient [degC]", C_A)):
        i, lo, me, hi = _blocks(x)
        h = i * dt / 3600
        a.fill_between(h, lo, hi, color=c, alpha=0.25, lw=0); a.plot(h, me, color=c, lw=0.8)
        a.set_ylabel(lab); a.grid(alpha=0.3)
    ax[0].set_title("Mission profile (band = min..max, line = mean per plotted block)")
    ax[2].set_xlabel("time [h]")
    _save(fig, fig_dir, "01_mission_profile.png")


def plot_temperatures(res, fig_dir):
    p = res["params"]; ps = res["per_second"]; dt = p.profile_dt
    fig, ax = plt.subplots(2, 1, figsize=(11, 8))
    for key, lab, c in (("I", "IGBT", C_I), ("D", "diode", C_D)):
        i, _, me, _ = _blocks(ps[f"Tj_mean_{key}"]); _, _, _, hi = _blocks(ps[f"Tj_max_{key}"])
        ax[0].plot(i * dt / 3600, me - 273.15, color=c, lw=0.8, label=f"{lab} Tj mean")
        ax[0].plot(i * dt / 3600, hi - 273.15, color=c, lw=0.5, alpha=0.5, label=f"{lab} Tj max")
    ax[0].axhline(p.max_IGBT_temperature - 273.15, color="k", ls=":", lw=1, label="limit")
    ax[0].set_xlabel("time [h]"); ax[0].set_ylabel("[degC]"); ax[0].legend(fontsize=8, ncol=5); ax[0].grid(alpha=0.3)
    ax[0].set_title("Junction temperature over the whole mission")
    spd = int(round(86400 / dt))
    n_days = max(1, ps["Tj_mean_I"].size // spd)
    if p.plot_day is not None:
        day = int(p.plot_day)
    else:
        dmg = res["hourly"]["damage_50Hz_I"] + res["hourly"]["damage_mission_I"]
        day = int(np.argmax(np.add.reduceat(dmg, np.arange(0, dmg.size, 24)))) if dmg.size else 0
        day = min(day, n_days - 1)
    sl = slice(day * spd, min((day + 1) * spd, ps["Tj_mean_I"].size))
    h = np.arange(sl.start, sl.stop) * dt / 3600 - day * 24
    ax[1].fill_between(h, ps["Tj_mean_I"][sl] - 273.15, ps["Tj_max_I"][sl] - 273.15, color=C_I, alpha=0.25,
                       lw=0, label="IGBT 50 Hz ripple (up to Tj max)")
    ax[1].plot(h, ps["Tj_mean_I"][sl] - 273.15, color=C_I, lw=0.7, label="IGBT Tj mean")
    ax[1].plot(h, ps["Tj_mean_D"][sl] - 273.15, color=C_D, lw=0.7, label="diode Tj mean")
    ax[1].plot(h, ps["T_sink"][sl] - 273.15, color=C_S, lw=0.7, label="heat sink")
    ax[1].plot(h, p.T_env[sl] - 273.15, color=C_A, lw=0.7, label="ambient")
    ax[1].set_xlabel("hour of the day"); ax[1].set_ylabel("[degC]"); ax[1].legend(fontsize=8, ncol=5)
    ax[1].set_title(f"Day {day + 1} at full resolution"); ax[1].grid(alpha=0.3)
    _save(fig, fig_dir, "02_temperatures.png")


def plot_one_period(res, fig_dir):
    p, net, bins = res["params"], res["network"], res["bins"]
    k = int(np.argmax(res["per_second"]["Tj_max_I"]))
    b = int(bins["index"][k])
    PI, PD = compute_bin_losses(p, bins, b, b + 1)
    RI, RD = ripple_from_losses(net, PI, PD, p.f)
    t = np.arange(p.Np) / p.Np / p.f * 1e3
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(t, PI[0], color=C_I, label="IGBT"); ax[0].plot(t, PD[0], color=C_D, label="diode")
    ax[0].set_xlabel("time in one period [ms]"); ax[0].set_ylabel("loss [W]"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title(f"Losses at the hottest second (Is={bins['Is'][b]:.1f} A, phi={np.degrees(bins['phi'][b]):.0f} deg)")
    ps = res["per_second"]
    ax[1].plot(t, ps["Tj_mean_I"][k] + RI[0] - 273.15, color=C_I, label=f"IGBT, dTj={np.ptp(RI):.1f} K")
    ax[1].plot(t, ps["Tj_mean_D"][k] + RD[0] - 273.15, color=C_D, label=f"diode, dTj={np.ptp(RD):.1f} K")
    ax[1].set_xlabel("time in one period [ms]"); ax[1].set_ylabel("Tj [degC]"); ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[1].set_title("Junction temperature = Rth*P_mean + Zth(jw) ripple")
    _save(fig, fig_dir, "03_one_period.png")


def plot_thermal_impedance(res, fig_dir):
    net = res["network"]; p = res["params"]
    f = np.logspace(-4, 3.3, 400)
    H = net.H(2 * np.pi * f)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(f, np.abs(H[0, 0]), color=C_I, label="Tj IGBT <- P IGBT")
    ax.loglog(f, np.abs(H[1, 1]), color=C_D, label="Tj diode <- P diode")
    ax.loglog(f, np.abs(H[2, 0]), color=C_S, label="T case <- P (paste + heat sink)")
    ax.axvline(p.f, color="k", ls=":"); ax.text(p.f * 1.1, ax.get_ylim()[1] * 0.5, f"{p.f:g} Hz")
    ax.axvspan(f[0], 0.5 / p.profile_dt, color="0.92")
    ax.text(f[0] * 1.5, 1e-3, "frequencies present in\nthe mission profile", fontsize=8)
    ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("|Zth| [K/W]"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_title("Thermal impedance: Rth at f = 0, much smaller at 50 Hz")
    _save(fig, fig_dir, "04_thermal_impedance.png")


def plot_rainflow_matrices(res, fig_dir):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for r, dev in enumerate(("IGBT", "Diode")):
        M = res[dev]["rainflow_matrix"]
        ex, ey = M["dT_edges_K"], M["Tjm_edges_C"]
        for c, (key, lab) in enumerate((("counts", "cycles"), ("damage", "damage"))):
            Z = M[f"{key}_50Hz"] + M[f"{key}_mission"]
            Z = np.where(Z > 0, Z, np.nan)
            a = ax[r, c]
            if np.all(np.isnan(Z)):
                a.set_title(f"{dev}: no {lab}"); continue
            im = a.pcolormesh(ey, ex, Z, norm=LogNorm(vmin=np.nanmax(Z) * 1e-6, vmax=np.nanmax(Z)), shading="flat")
            nz = np.argwhere(np.isfinite(Z))
            a.set_ylim(0, ex[min(nz[:, 0].max() + 3, ex.size - 1)])
            a.set_xlim(ey[max(nz[:, 1].min() - 3, 0)], ey[min(nz[:, 1].max() + 3, ey.size - 1)])
            fig.colorbar(im, ax=a, label=lab)
            a.set_xlabel("cycle mid temperature Tjm [degC]"); a.set_ylabel("cycle swing dTj [K]")
            a.set_title(f"{dev}: {lab} in the mission")
    _save(fig, fig_dir, "05_rainflow_matrices.png")


def plot_damage(res, fig_dir):
    p = res["params"]; hr = res["hourly"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for dev, sfx, col in (("IGBT", "I", C_I), ("Diode", "D", C_D)):
        M = res[dev]["rainflow_matrix"]; e = M["dT_edges_K"]
        g = max(1, int(round(2.0 / (e[1] - e[0]))))                  # group into ~2 K bins
        n = (e.size - 1) // g
        c = 0.5 * (e[0:n * g:g] + e[g:n * g + 1:g])
        tot = res[dev]["damage_total"]
        if tot > 0:
            f_ = M["damage_50Hz"].sum(1)[:n * g].reshape(n, g).sum(1) / tot
            m_ = M["damage_mission"].sum(1)[:n * g].reshape(n, g).sum(1) / tot
            ax[0].step(c, f_, where="mid", color=col, label=f"{dev} 50 Hz cycles")
            ax[0].step(c, m_, where="mid", color=col, ls="--", label=f"{dev} mission cycles")
            cum = np.cumsum(hr[f"damage_50Hz_{sfx}"] + hr[f"damage_mission_{sfx}"])
            ax[1].plot(hr["t_start_h"] / 24, cum, color=col, label=f"{dev}: {res[dev]['lifetime_years']:.3g} years")
    ax[0].set_xlabel("cycle swing dTj [K]"); ax[0].set_ylabel("share of the device's damage")
    ax[0].set_xlim(0, None); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3); ax[0].set_title("Which cycles do the damage")
    ax[1].set_xlabel("time [days]"); ax[1].set_ylabel("accumulated damage (1 = end of life)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both"); ax[1].set_title("Damage over the mission")
    _save(fig, fig_dir, "06_damage.png")


def plot_reliability(res, fig_dir):
    mc, rel = res["monte_carlo_years"], res["reliability"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    cols = {"IGBT": C_I, "Diode": C_D, "Switch": "k", "Inverter": "tab:purple"}
    finite = [v[np.isfinite(v)] for v in mc.values() if np.isfinite(v).any()]
    tmax = max(np.percentile(np.concatenate(finite), 99), 1e-6) if finite else 1.0
    for k in ("IGBT", "Diode", "Switch"):
        x = mc[k][np.isfinite(mc[k])]
        if x.size and np.isfinite(rel[k]["shape"]) and np.percentile(x, 50) < 5 * tmax:
            ax[0].hist(x, bins=60, density=True, alpha=0.3, color=cols[k], range=(0, tmax))
            tt = np.linspace(1e-9, tmax, 400)
            ax[0].plot(tt, weibull_min.pdf(tt, rel[k]["shape"], scale=rel[k]["scale"]), color=cols[k], label=k)
    ax[0].set_xlabel("lifetime [years]"); ax[0].set_ylabel("probability density"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title("Monte Carlo lifetime + Weibull fit")
    tt = np.linspace(1e-9, tmax, 600)
    for k in ("IGBT", "Diode", "Switch", "Inverter"):
        w = rel.get(k) or {}
        if w and np.isfinite(w.get("shape", np.nan)):
            ax[1].plot(tt, 100 * weibull_min.cdf(tt, w["shape"], scale=w["scale"]), color=cols[k],
                       label=f"{k}: B10 = {w['B10']:.3g} y")
    ax[1].axhline(10, color="k", ls=":", lw=1)
    ax[1].set_xlabel("time [years]"); ax[1].set_ylabel("unreliability F(t) [%]"); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3); ax[1].set_title("Probability of failure before time t")
    _save(fig, fig_dir, "07_reliability.png")


def plot_lifetime_models(res, fig_dir):
    """Only when several lifetime models were selected: Miner lifetime and B10 per model."""
    lm = res.get("lifetime_models", {})
    if len(lm) < 2:
        return
    names = list(lm)
    y = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(12, 0.6 * len(names) + 2.2), sharey=True)
    for a, dev, col in ((ax[0], "IGBT", C_I), (ax[1], "Diode", C_D)):
        life = [lm[n][dev]["lifetime_years"] for n in names]
        b10 = [lm[n]["reliability"][dev]["B10"] for n in names]
        a.barh(y, life, color=col, alpha=0.5, label="Miner lifetime")
        a.plot(b10, y, "k|", ms=18, mew=2, label="B10 (Monte Carlo)")
        for yi, v in zip(y, life):
            a.text(v, yi, f" {v:.3g} y", va="center", fontsize=8)
        a.set_xscale("log"); a.set_xlabel("years"); a.set_title(f"{dev}: lifetime by model"); a.grid(alpha=0.3, which="both")
    ax[0].set_yticks(y); ax[0].set_yticklabels(names); ax[0].invert_yaxis(); ax[1].legend(fontsize=8)
    _save(fig, fig_dir, "08_lifetime_models.png")


def plot_all(res, fig_dir):
    for fn in (plot_mission_profile, plot_temperatures, plot_one_period, plot_thermal_impedance,
               plot_rainflow_matrices, plot_damage, plot_reliability, plot_lifetime_models):
        try:
            fn(res, fig_dir)
        except Exception as exc:          # a plot must never stop a simulation
            print(f"[Plotting_results] {fn.__name__} skipped: {exc}")
