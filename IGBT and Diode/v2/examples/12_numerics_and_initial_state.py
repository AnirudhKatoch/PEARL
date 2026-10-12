"""Study 12 - numerical settings converge; the start condition (cold start) can dominate short profiles."""
from _common import *


def main():
    d = outdir("12_numerics_and_initial_state")
    P, Q, T = synthetic_pv_profile(30)
    rows = []
    ref = run(P, Q, T, MC_samples=200)
    for name, kw in (("Np = 20", dict(Np=20)), ("Np = 40 (default)", dict()), ("Np = 100", dict(Np=100)),
                     ("no operating-point binning", dict(op_resolution=None)), ("coarse binning (0.1 A, 5 mrad)", dict(op_resolution=(0.1, 5e-3))),
                     ("slow_gate = 0", dict(slow_gate=0.0)), ("slow_gate = 0.5 K", dict(slow_gate=0.5)),
                     ("init = ambient (cold start)", dict(init="ambient")), ("init = periodic", dict(init="periodic"))):
        r = run(P, Q, T, MC_samples=200, **kw)
        rows.append({"setting": name, "IGBT_life_y": r["IGBT"]["lifetime_years"], "Diode_life_y": r["Diode"]["lifetime_years"],
                     "IGBT_vs_default_%": 100 * (r["IGBT"]["lifetime_years"] / ref["IGBT"]["lifetime_years"] - 1),
                     "Diode_vs_default_%": 100 * (r["Diode"]["lifetime_years"] / ref["Diode"]["lifetime_years"] - 1),
                     "operating_bins": r["n_operating_bins"], "mission_cycles_IGBT": r["IGBT"]["n_mission_cycles"], "time_s": r["timing_s"]["total"]})
    df = table(rows, d, "numerical_settings_30_days.csv")
    rows2 = []
    for pf in (1.0, 0.7, 0.4):
        P1, Q1, T1 = constant(S_RATED * pf, S_RATED * np.sqrt(1 - pf ** 2), 3600)
        for init in ("ambient", "steady", "periodic"):
            r = run(P1, Q1, T1, init=init, MC_samples=200)
            rows2.append({"pf": pf, "init": init, "IGBT_life_y": r["IGBT"]["lifetime_years"], "Diode_life_y": r["Diode"]["lifetime_years"],
                          "IGBT_damage_share_from_slow_cycles": r["IGBT"]["damage_mission_cycles"] / r["IGBT"]["damage_total"]})
    df2 = table(rows2, d, "initial_state_1h_constant.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    for init, g in df2.groupby("init"):
        ax.semilogy(g.pf, np.minimum(g.IGBT_life_y, g.Diode_life_y), "o-", label=f"init = {init}")
    ax.set_xlabel("power factor"); ax.set_ylabel("switch lifetime [years]"); ax.grid(alpha=0.3, which="both"); ax.legend()
    ax.set_title("1 h of constant operation repeated over life: a cold start each hour inflates damage")
    save_fig(fig, d, "initial_state.png")
    summary(d, "12 - Numerical settings and initial state",
            "The numerical settings (points per period, binning, slow gate) change the lifetime by a few percent at most. "
            "The start condition matters for short repeated profiles: 'ambient' counts a warm-up cycle every repetition.",
            [("Settings (30 days)", df), ("1 h constant operation, three start conditions", df2)])


if __name__ == "__main__":
    main()
