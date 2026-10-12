"""Study 10 - what moves the lifetime most: losses vs temperature, heat sink, devices in parallel, Rth, DC voltage."""
from _common import *


def main():
    d = outdir("10_sensitivity")
    P, Q, T = synthetic_pv_profile(30)
    base = run(P, Q, T, MC_samples=300)
    L0 = (base["IGBT"]["lifetime_years"], base["Diode"]["lifetime_years"])
    cases = [("losses +0.1 %/K (kappa 0.001)", dict(kappa_IGBT=0.001, kappa_Diode=0.001)),
             ("losses +0.2 %/K (kappa 0.002)", dict(kappa_IGBT=0.002, kappa_Diode=0.002)),
             ("losses +0.3 %/K (kappa 0.003)", dict(kappa_IGBT=0.003, kappa_Diode=0.003)),
             ("heat sink 1.14 K/W (2 m/s)", dict(r_sink=1.14)), ("heat sink 0.72 K/W (10U, 4 m/s)", dict(r_sink=0.72, sink_mass=0.018)),
             ("3 devices in parallel", dict(N_parallel=3)),
             ("Zth_jc +10 %", dict(r_I=1.1 * base["params"].r_I, r_D=1.1 * base["params"].r_D)),
             ("Zth_jc -10 %", dict(r_I=0.9 * base["params"].r_I, r_D=0.9 * base["params"].r_D)),
             ("DC link 580 V", dict(V_dc_value=580.0)), ("DC link 560 V (too low for 230 V AC)", dict(V_dc_value=560.0)), ("switching 5 kHz", dict(f_sw=5e3)), ("switching 20 kHz", dict(f_sw=20e3))]
    rows = [{"case": "baseline (2 devices, 2 K/W, 10 kHz, 600 V)", "IGBT_life_y": L0[0], "Diode_life_y": L0[1], "IGBT_factor": 1.0,
             "Diode_factor": 1.0, "IGBT_Tj_max_C": base["IGBT"]["Tj_max_C"], "IGBT_dTj50_max_K": base["IGBT"]["dTj_50Hz_max_K"]}]
    for name, kw in cases:
        try:
            r = run(P, Q, T, MC_samples=300, **kw)
            rows.append({"case": name, "IGBT_life_y": r["IGBT"]["lifetime_years"], "Diode_life_y": r["Diode"]["lifetime_years"],
                         "IGBT_factor": r["IGBT"]["lifetime_years"] / L0[0], "Diode_factor": r["Diode"]["lifetime_years"] / L0[1],
                         "IGBT_Tj_max_C": r["IGBT"]["Tj_max_C"], "IGBT_dTj50_max_K": r["IGBT"]["dTj_50Hz_max_K"]})
        except ValueError as e:
            rows.append({"case": f"{name} (not possible: {str(e)[:60]})"})
    df = table(rows, d, "sensitivity.csv")
    g = df.dropna(subset=["IGBT_factor"]).iloc[1:]
    fig, ax = plt.subplots(figsize=(10, 5)); y = np.arange(len(g))
    ax.barh(y - 0.2, g.IGBT_factor, 0.4, color=C_I, label="IGBT"); ax.barh(y + 0.2, g.Diode_factor, 0.4, color=C_D, label="diode")
    ax.axvline(1, color="k"); ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(g.case, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("lifetime relative to baseline (log)"); ax.legend(); ax.grid(alpha=0.3, which="both")
    ax.set_title("Sensitivity of lifetime to inputs and design choices (30 days)")
    save_fig(fig, d, "sensitivity.png")
    summary(d, "10 - Sensitivity", "Each row changes one input from the baseline. Temperature-dependent losses (kappa) and the "
            "heat sink move the lifetime the most; small changes in Zth_jc also matter because the lifetime model is steep.",
            [("Lifetime factors", df)])


if __name__ == "__main__":
    main()
