"""Study 01 - the standard PEARL run: 90 days of 1-s PV mission profile, all figures and data."""
from _common import *
import shutil, json


def main():
    d = outdir("01_full_run_90_days")
    P, Q, T = synthetic_pv_profile(90)
    t0 = time.time()
    r = mother_function(P, Q, T, N_parallel=2, results_dir=str(d / "PEARL_output"), save=True, plot=True)
    rows = [{"device": n, "lifetime_years": r[n]["lifetime_years"], "B10_years": r["reliability"][n]["B10"],
             "share_of_damage_from_50Hz": r[n]["share_50Hz"], "Tj_max_C": r[n]["Tj_max_C"], "Tj_mean_C": r[n]["Tj_mean_C"],
             "max_50Hz_dTj_K": r[n]["dTj_50Hz_max_K"], "mission_cycles": r[n]["n_mission_cycles"]} for n in ("IGBT", "Diode")]
    df = table(rows, d, "lifetime_summary.csv")
    sys_rows = [{"level": k, "B1_years": v.get("B1"), "B10_years": v.get("B10"), "B50_years": v.get("B50"),
                 "weibull_shape": v.get("shape")} for k, v in r["reliability"].items()]
    df2 = table(sys_rows, d, "reliability_B10.csv")
    tm = pd.DataFrame([{"step": k, "seconds": v} for k, v in r["timing_s"].items()])
    tm.to_csv(d / "timing.csv", index=False)
    summary(d, "01 - Full run: 90 days at 1 s",
            "The normal PEARL v3 run on a synthetic 90-day PV profile (clouds, reactive support, daily ambient cycle), "
            "2 devices in parallel, the original 2 K/W heat sink. Figures are in PEARL_output/Simulation_1/Figures, "
            f"data in PEARL_output/Simulation_1/Data. Total time {time.time() - t0:.1f} s.",
            [("Lifetime per device", df), ("Reliability (Monte Carlo + Weibull); Inverter = 12 devices in series", df2),
             ("Where the time goes [s]", tm)])


if __name__ == "__main__":
    main()
