"""Study 05 - how each methodology change moves the lifetime (same 30-day profile)."""
from _common import *


def main():
    d = outdir("05_v1_to_v3_steps")
    P, Q, T = synthetic_pv_profile(30)
    steps = [("1 v1 physics: legacy losses, ladder, raw t_on, cold start, diode k=0.5",
              dict(loss_model="legacy", junction_model="ladder", ton_rule="raw", init="ambient", k_thickness={"IGBT": 1, "Diode": 0.5})),
             ("2 + physically consistent losses", dict(junction_model="ladder", ton_rule="raw", init="ambient", k_thickness={"IGBT": 1, "Diode": 0.5})),
             ("3 + datasheet Foster Zth_jc", dict(ton_rule="raw", init="ambient", k_thickness={"IGBT": 1, "Diode": 0.5})),
             ("4 + t_on >= 40 ms (AN 21-001)", dict(init="ambient", k_thickness={"IGBT": 1, "Diode": 0.5})),
             ("5 + steady-state start (v2)", dict(k_thickness={"IGBT": 1, "Diode": 0.5})),
             ("6 + freewheeling-diode factor 0.65 (v3 default)", dict())]
    rows = []
    for name, kw in steps:
        r = run(P, Q, T, **kw)
        rows.append({"step": name, "IGBT_life_years": r["IGBT"]["lifetime_years"], "Diode_life_years": r["Diode"]["lifetime_years"],
                     "IGBT_B10_years": r["reliability"]["IGBT"]["B10"], "IGBT_share_50Hz": r["IGBT"]["share_50Hz"],
                     "IGBT_dTj50_max_K": r["IGBT"]["dTj_50Hz_max_K"], "IGBT_Tj_max_C": r["IGBT"]["Tj_max_C"]})
    df = table(rows, d, "steps.csv")
    fig, ax = plt.subplots(figsize=(10, 4.5)); y = np.arange(len(df))
    ax.barh(y - 0.2, df.IGBT_life_years, 0.4, color=C_I, label="IGBT"); ax.barh(y + 0.2, df.Diode_life_years, 0.4, color=C_D, label="diode")
    ax.set_yticks(y); ax.set_yticklabels(df.step, fontsize=8); ax.invert_yaxis(); ax.set_xscale("log")
    ax.set_xlabel("lifetime [years], log scale"); ax.legend(); ax.grid(alpha=0.3, which="both")
    ax.set_title("Each correction, one at a time (30 days, 2 devices, 2 K/W sink)")
    save_fig(fig, d, "steps.png")
    summary(d, "05 - From PEARL v1 physics to PEARL v3",
            "Every row adds one correction to the row above. The big jumps are the loss model (v1 losses were about half), the "
            "Foster thermal model (larger 50 Hz swings) and the 40 ms t_on floor.", [("Lifetimes", df)])


if __name__ == "__main__":
    main()
