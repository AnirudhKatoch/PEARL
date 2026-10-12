"""Study 09 - which thermal capacitances matter at 1-s resolution (and why the pure-Rth formula fails for the mean)."""
from _common import *


def main():
    d = outdir("09_timescales_capacitance")
    P, Q, T = synthetic_pv_profile(7)
    variants = (("exact (all capacitances)", 0.0), ("chip quasi-static (tau < 1 s)", 1.0),
                ("only heat sink dynamic (tau < 10 s)", 10.0), ("pure Rth, no capacitance", np.inf))
    rows, curves = [], {}
    ref = None
    for name, qs in variants:
        r = run(P, Q, T, quasi_static_tau=qs, MC_samples=300)
        Tm = r["per_second"]["Tj_mean_I"]
        ref = Tm if ref is None else ref
        curves[name] = Tm
        rows.append({"variant": name, "IGBT_life_y": r["IGBT"]["lifetime_years"], "Diode_life_y": r["Diode"]["lifetime_years"],
                     "max_|Tj_mean - exact|_K": float(np.max(np.abs(Tm - ref))), "IGBT_mission_cycles": r["IGBT"]["n_mission_cycles"],
                     "IGBT_share_50Hz": r["IGBT"]["share_50Hz"]})
    df = table(rows, d, "capacitance_variants.csv")
    tau = r["network"].time_constants
    day = int(np.argmax([np.abs(np.diff(P[k * 86400:(k + 1) * 86400])).sum() for k in range(7)]))
    sl = slice(day * 86400 + 11 * 3600, day * 86400 + 13 * 3600); h = np.arange(sl.start, sl.stop) / 3600 - day * 24
    fig, ax = plt.subplots(figsize=(11, 4))
    for (name, _), col in zip(variants, ("k", "tab:orange", "tab:green", "tab:red")):
        ax.plot(h, curves[name][sl] - 273.15, color=col, lw=0.8 if col != "k" else 2, alpha=0.9, label=name)
    ax.set_xlabel("hour of the day"); ax.set_ylabel("IGBT Tj mean [degC]"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Two cloudy hours: without the heat-sink capacitance every cloud edge is an instant jump")
    save_fig(fig, d, "capacitance_variants_cloudy_hours.png")
    summary(d, "09 - Time scales and capacitances",
            f"Thermal time constants of the network [s]: {', '.join(f'{t:.2g}' for t in tau)}. At a 1-s step the chip modes "
            "(<= 0.1 s) are already settled, so dropping their capacitance changes nothing; the heat sink (about 34 s here) must "
            "keep its capacitance: without it the mean temperature jumps with every cloud edge (errors up to about 60 K here) and the "
            "predicted lifetime roughly halves.",
            [("Variants (7 days)", df)])


if __name__ == "__main__":
    main()
