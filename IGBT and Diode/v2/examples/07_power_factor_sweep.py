"""Study 07 - constant rated apparent power, power factor from 0 to 1, delivering and charging (four quadrants)."""
from _common import *


def main():
    d = outdir("07_power_factor_sweep")
    rows = []
    for direction, sP in (("P > 0 (inverter)", 1), ("P < 0 (charging)", -1)):
        for pf in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0):
            for sQ, qlab in ((1, "Q delivered"), (-1, "Q absorbed")):
                P, Q, T = constant(sP * S_RATED * pf, sQ * S_RATED * np.sqrt(1 - pf ** 2), 3600)
                r = run(P, Q, T, init="periodic", MC_samples=200)
                ps = r["per_second"]
                rows.append({"direction": direction, "Q": qlab, "pf": pf, "IGBT_loss_W": float(ps["P_mean_I"][-1]),
                             "Diode_loss_W": float(ps["P_mean_D"][-1]), "IGBT_dTj50_K": r["IGBT"]["dTj_50Hz_max_K"],
                             "Diode_dTj50_K": r["Diode"]["dTj_50Hz_max_K"], "IGBT_Tj_max_C": r["IGBT"]["Tj_max_C"],
                             "Diode_Tj_max_C": r["Diode"]["Tj_max_C"], "IGBT_life_y": r["IGBT"]["lifetime_years"],
                             "Diode_life_y": r["Diode"]["lifetime_years"], "switch_life_y": r["switch_lifetime_years"]})
    df = table(rows, d, "pf_sweep.csv")
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for (direction, qlab), g in df.groupby(["direction", "Q"]):
        ls = "-" if qlab == "Q delivered" else "--"
        lab = f"{direction}, {qlab}"
        ax[0, 0].plot(g.pf, g.IGBT_loss_W, ls, color=C_I); ax[0, 0].plot(g.pf, g.Diode_loss_W, ls, color=C_D)
        ax[0, 1].plot(g.pf, g.IGBT_dTj50_K, ls, color=C_I); ax[0, 1].plot(g.pf, g.Diode_dTj50_K, ls, color=C_D)
        ax[1, 0].plot(g.pf, g.IGBT_Tj_max_C, ls, color=C_I); ax[1, 0].plot(g.pf, g.Diode_Tj_max_C, ls, color=C_D)
        ax[1, 1].semilogy(g.pf, g.switch_life_y, ls, marker="o", ms=3, label=lab)
    for a, t in zip(ax.ravel(), ("mean loss [W] (blue IGBT, red diode)", "50 Hz swing dTj [K]", "Tj max [degC]", "switch lifetime [years]")):
        a.set_xlabel("|power factor|"); a.set_title(t); a.grid(alpha=0.3, which="both")
    ax[1, 1].legend(fontsize=7)
    save_fig(fig, d, "pf_sweep.png")
    summary(d, "07 - Power-factor sweep (rated S, 2 devices, continuous operation, no cold start)",
            "Delivering power loads the IGBT; charging (P < 0) and low power factor load the diode. Solid = Q delivered, "
            "dashed = Q absorbed (nearly identical). Continuous operation at full rated S is a stress test, not a mission.",
            [("Delivering, Q delivered", df[(df.direction == "P > 0 (inverter)") & (df.Q == "Q delivered")][
                ["pf", "IGBT_loss_W", "Diode_loss_W", "IGBT_dTj50_K", "Diode_dTj50_K", "IGBT_life_y", "Diode_life_y"]]),
             ("Charging, Q delivered", df[(df.direction == "P < 0 (charging)") & (df.Q == "Q delivered")][
                ["pf", "IGBT_loss_W", "Diode_loss_W", "IGBT_dTj50_K", "Diode_dTj50_K", "IGBT_life_y", "Diode_life_y"]])])


if __name__ == "__main__":
    main()
