"""Study 08 - what you lose with coarse mission-profile data (same 30 days at 1 s ... 600 s)."""
from _common import *


def main():
    d = outdir("08_data_resolution")
    P, Q, T = synthetic_pv_profile(30)
    rows = []
    for dt in (1, 5, 10, 30, 60, 300, 600):
        n = P.size // dt
        avg = lambda x: x[:n * dt].reshape(n, dt).mean(1)
        r = run(avg(P), avg(Q), avg(T), profile_dt=float(dt), MC_samples=300)
        rows.append({"data_every_s": dt, "IGBT_life_y": r["IGBT"]["lifetime_years"], "Diode_life_y": r["Diode"]["lifetime_years"],
                     "IGBT_share_50Hz": r["IGBT"]["share_50Hz"], "IGBT_mission_cycles": r["IGBT"]["n_mission_cycles"],
                     "IGBT_Tj_max_C": r["IGBT"]["Tj_max_C"], "run_time_s": r["timing_s"]["total"]})
    df = table(rows, d, "resolution.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogx(df.data_every_s, df.IGBT_life_y, "o-", color=C_I, label="IGBT"); ax[0].semilogx(df.data_every_s, df.Diode_life_y, "o-", color=C_D, label="diode")
    ax[0].set_xlabel("mission-profile time step [s]"); ax[0].set_ylabel("lifetime [years]"); ax[0].legend(); ax[0].grid(alpha=0.3, which="both")
    ax[0].set_title("Coarse data hides cloud cycles -> lifetime too long")
    ax[1].semilogx(df.data_every_s, df.IGBT_mission_cycles, "o-", color="k"); ax[1].set_xlabel("time step [s]")
    ax[1].set_ylabel("mission-profile cycles found (IGBT)"); ax[1].grid(alpha=0.3, which="both")
    save_fig(fig, d, "resolution.png")
    summary(d, "08 - Mission-profile resolution",
            "The same 30 days, block-averaged before the simulation. Averaging removes short cloud dips, so fewer and smaller "
            "slow cycles are seen and the predicted lifetime grows. Use 1-s data when you have it.", [("Results", df)])


if __name__ == "__main__":
    main()
