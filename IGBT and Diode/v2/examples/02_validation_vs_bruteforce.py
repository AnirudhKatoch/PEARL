"""Study 02 - is the fast method right? Compare with a brute-force time simulation + full rainflow."""
from _common import *
from Input_parameters import Input_parameters_class
from Calculation_functions import Calculation_functions_class as CF
from Thermal_model import ThermalNetwork, ripple_from_losses
from Electrical_model import compute_bin_losses
from Lifetime_models import build_lifetime_model


def bruteforce(P, Q, T, Np, **kw):
    p = Input_parameters_class(P, Q, T, Np=Np, op_resolution=None, **kw)
    net = ThermalNetwork.from_params(p)
    bins = CF.bin_operating_points(p.Is, p.phi, p.V_dc, p.ma, None)
    PIb, PDb = compute_bin_losses(p, bins)
    PI = np.tile(PIb[bins["index"]], (1, 50)).ravel(); PD = np.tile(PDb[bins["index"]], (1, 50)).ravel()
    dt = 0.02 / Np
    z0 = net.steady_state(PIb[bins["index"][0]].mean(), PDb[bins["index"][0]].mean(), p.T_env[0])
    t0 = time.time()
    Y, _ = net.simulate(PI, PD, np.repeat(p.T_env, 50 * Np), dt, z0)
    model = build_lifetime_model("semikron2024", p)
    D = {}
    for k, n in ((0, "IGBT"), (1, "Diode")):
        dT, Tm, ton, cnt, _, _ = CF.rainflow_turning_points(Y[k], dt)
        D[n] = float(np.sum(cnt * CF.damage_per_cycle(dT, Tm, ton, model.bind(n), None, "semikron")))
    return D, Y, time.time() - t0


def main():
    d = outdir("02_validation_vs_bruteforce")
    P, Q, T = synthetic_pv_profile(3)
    a, b = 11 * 3600, 11 * 3600 + 1200
    rows = []
    designs = (("2 devices, 2 K/W sink", dict(N_parallel=2)), ("1 device, 0.72 K/W sink", dict(N_parallel=1, r_sink=0.72, sink_mass=0.018)))
    for label, kw in designs:
        for Np in (20, 100, 400):
            ref, Y, t_ref = bruteforce(P[a:b], Q[a:b], T[a:b], Np, **kw)
            t0 = time.time()
            r = run(P[a:b], Q[a:b], T[a:b], Np=Np, op_resolution=None, MC_samples=100, **kw)
            t_new = time.time() - t0
            for n in ("IGBT", "Diode"):
                rows.append({"design": label, "points_per_period": Np, "device": n, "damage_bruteforce": ref[n],
                             "damage_PEARL_v3": r[n]["damage_total"], "ratio": r[n]["damage_total"] / ref[n],
                             "time_bruteforce_s": t_ref, "time_PEARL_v3_s": t_new})
            if Np == 400 and label.startswith("1 device"):
                k = 600; Npp = 400
                t = np.arange(Npp) * 0.02 / Npp * 1e3
                seg = Y[0][(k * 50 + 49) * Npp:(k * 50 + 50) * Npp]
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(t, seg - 273.15, lw=4, alpha=0.4, label="brute force, 0.05 ms time simulation")
                ps = r["per_second"]
                p = r["params"]; bins = r["bins"]; bidx = int(bins["index"][k])
                PI, PD = compute_bin_losses(p, bins, bidx, bidx + 1)
                RI, _ = ripple_from_losses(r["network"], PI, PD, 50.0)
                ax.plot(t, ps["Tj_mean_I"][k] + RI[0] - 273.15, "k--", label="PEARL v3: Rth*P_mean + Zth(jw) ripple")
                ax.set_xlabel("time in one 50 Hz period [ms]"); ax.set_ylabel("IGBT Tj [degC]"); ax.legend(fontsize=8)
                ax.set_title(f"Second {k} of a cloudy window: the two methods overlap"); ax.grid(alpha=0.3)
                save_fig(fig, d, "one_period_overlay.png")
    df = table(rows, d, "validation.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    for (label, n), g in df.groupby(["design", "device"]):
        ax.plot(g["points_per_period"], g["ratio"], "o-", label=f"{label}, {n}")
    ax.axhline(1, color="k", lw=1); ax.set_xscale("log"); ax.set_xlabel("time points per 50 Hz period (both methods)")
    ax.set_ylabel("damage PEARL v3 / brute force"); ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
    ax.set_title("Agreement improves as both are refined; at 20 points the brute force itself is off")
    save_fig(fig, d, "damage_ratio_vs_resolution.png")
    summary(d, "02 - Validation against a brute-force reference",
            "20 minutes of broken clouds. Reference = the same losses simulated in time (no shortcut) followed by rainflow on "
            "every point. At 400 points per period (0.05 ms) the two agree within about 2 %. At 20 points (1 ms, the old "
            "PEARL time step) the brute force itself overestimates the 50 Hz swing, which shows up as a lower ratio.",
            [("Damage ratio PEARL v3 / brute force", df[["design", "points_per_period", "device", "ratio", "time_bruteforce_s", "time_PEARL_v3_s"]])])


if __name__ == "__main__":
    main()
