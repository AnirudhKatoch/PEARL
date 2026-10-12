"""Study 13 - run time vs profile length, compared with PEARL v1 (BDF at 1 ms, measured 0.144 s per simulated second)."""
from _common import *


def main():
    d = outdir("13_speed_benchmark")
    rows = []
    for days in (1, 7, 30, 90):
        P, Q, T = synthetic_pv_profile(days)
        t0 = time.time(); r = run(P, Q, T, MC_samples=5000); t = time.time() - t0
        rows.append({"days": days, "seconds_of_profile": P.size, "PEARL_v3_s": t, "PEARL_v1_estimate_h": P.size * 0.144 / 3600,
                     "speed_up": P.size * 0.144 / t, "operating_bins": r["n_operating_bins"]})
        del P, Q, T, r
    df = table(rows, d, "speed.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(df.days, df.PEARL_v3_s, "o-", label="PEARL v3 (measured, incl. Monte Carlo)")
    ax.loglog(df.days, df.PEARL_v1_estimate_h * 3600, "s--", label="PEARL v1 (0.144 s per simulated second)")
    ax.set_xlabel("profile length [days]"); ax.set_ylabel("computing time [s]"); ax.grid(alpha=0.3, which="both"); ax.legend()
    ax.set_title("Computing time"); save_fig(fig, d, "speed.png")
    summary(d, "13 - Speed", "v1 time is extrapolated from a measured 10-s run of the old BDF solver (thermal step only, "
            "without the 1-ms rainflow). v3 time includes losses, thermal, damage and a 5000-sample Monte Carlo, no plots.",
            [("Timing", df)])


if __name__ == "__main__":
    main()
