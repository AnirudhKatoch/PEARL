"""Study 11 - uncertainty: Monte Carlo spread, Weibull fit, B10 of device, switch and inverter."""
from _common import *
from scipy.stats import weibull_min


def main():
    d = outdir("11_monte_carlo_reliability")
    P, Q, T = synthetic_pv_profile(30)
    rows, rel = [], {}
    for name, sig in (("low (+-2 % at 3 sigma)", {"dT": 0.02 / 3, "Tjm_K": 1.0, "A": 0.02 / 3, "alpha": 0.02 / 3}),
                      ("default (+-5 % at 3 sigma)", None),
                      ("high (+-15 % at 3 sigma)", {"dT": 0.15 / 3, "Tjm_K": 5.0, "A": 0.15 / 3, "alpha": 0.15 / 3})):
        kw = {} if sig is None else {"MC_sigma": sig}
        r = run(P, Q, T, MC_samples=5000, **kw)
        rel[name] = r["reliability"]
        for lvl in ("IGBT", "Diode", "Switch", "Inverter"):
            w = r["reliability"][lvl]
            rows.append({"uncertainty": name, "level": lvl, "B1_y": w.get("B1"), "B10_y": w.get("B10"), "B50_y": w.get("B50"),
                         "weibull_shape": w.get("shape")})
        if sig is None:
            mc = r["monte_carlo_years"]; miner = (r["IGBT"]["lifetime_years"], r["Diode"]["lifetime_years"])
    df = table(rows, d, "reliability.csv")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, col in (("IGBT", C_I), ("Diode", C_D), ("Switch", "k")):
        x = mc[k][np.isfinite(mc[k])]
        ax[0].hist(x, bins=80, density=True, alpha=0.35, color=col, label=k, range=(0, np.percentile(mc["Diode"], 99)))
    ax[0].axvline(miner[0], color=C_I, ls="--", label="IGBT Miner lifetime"); ax[0].axvline(miner[1], color=C_D, ls="--", label="diode Miner lifetime")
    ax[0].set_xlabel("lifetime [years]"); ax[0].set_ylabel("density"); ax[0].legend(fontsize=7); ax[0].set_title("Monte Carlo spread (default uncertainty)")
    tt = np.linspace(0.01, 40, 400)
    for (name, R), ls in zip(rel.items(), (":", "-", "--")):
        for lvl, col in (("Switch", "k"), ("Inverter", "tab:purple")):
            w = R[lvl]
            if w and np.isfinite(w.get("shape", np.nan)):
                ax[1].plot(tt, 100 * weibull_min.cdf(tt, w["shape"], scale=w["scale"]), ls, color=col, label=f"{lvl}, {name}")
    ax[1].axhline(10, color="0.5", lw=1); ax[1].set_xlabel("time [years]"); ax[1].set_ylabel("failed [%]")
    ax[1].legend(fontsize=6); ax[1].grid(alpha=0.3); ax[1].set_title("Unreliability: one switch vs whole inverter (12 devices)")
    save_fig(fig, d, "reliability.png")
    summary(d, "11 - Monte Carlo and reliability",
            "Each sample shifts the cycle swings, the mean temperatures and the lifetime-model coefficient and slope a little, "
            "then recounts the damage of every stored cycle. B10 = time when 10 % have failed. The inverter fails when any of "
            "its 12 devices fails, so its B10 is shorter than one switch's.", [("B-lives", df)])


if __name__ == "__main__":
    main()
