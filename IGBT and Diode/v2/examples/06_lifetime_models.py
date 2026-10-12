"""Study 06 - lifetime model choice: built-in models, user curves, fitted curves, side by side."""
from _common import *
from Lifetime_models import build_lifetime_model, USER_CURVES_EXAMPLE, fit_cma, Semikron2024


def main():
    d = outdir("06_lifetime_models")
    models = ["semikron2024", {"model": "semikron2024", "parameter_set": "no_baseplate"},
              {"model": "semikron2024", "parameter_set": "sintered"}, "lesit",
              {"model": "bayerer2008", "I_per_wire_A": 10.0, "V_blocking_V": 600, "D_wire_um": 300}, USER_CURVES_EXAMPLE]
    # Nf curves of every model
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    dT = np.linspace(5, 120, 200)
    for spec in models:
        m = build_lifetime_model(spec)
        for a, ton in ((ax[0], 0.04), (ax[1], 10.0)):
            a.loglog(dT, m.bind("IGBT")(dT, np.full(dT.size, 373.15), np.full(dT.size, ton)), label=m.name)
    for a, ton in ((ax[0], "0.04 s (50 Hz cycles)"), (ax[1], "10 s (cloud cycles)")):
        a.set_xlabel("dTj [K]"); a.set_ylabel("cycles to failure Nf"); a.set_title(f"IGBT, Tjm = 100 degC, t_on = {ton}")
        a.grid(alpha=0.3, which="both"); a.legend(fontsize=7)
    save_fig(fig, d, "Nf_curves_by_model.png")
    P, Q, T = synthetic_pv_profile(30)
    r = run(P, Q, T, lifetime_model=models, MC_samples=1000, MC_samples_other_models=1000)
    rows = [{"model": m, "device": n, "lifetime_years": o[n]["lifetime_years"], "B10_years": o["reliability"][n]["B10"],
             "share_50Hz": o[n]["share_50Hz"]} for m, o in r["lifetime_models"].items() for n in ("IGBT", "Diode")]
    df = table(rows, d, "lifetime_by_model.csv")
    piv = df.pivot(index="model", columns="device", values="lifetime_years").reindex(list(r["lifetime_models"]))
    fig, ax = plt.subplots(figsize=(9, 4)); y = np.arange(len(piv))
    ax.barh(y - 0.2, piv["IGBT"], 0.4, color=C_I, label="IGBT"); ax.barh(y + 0.2, piv["Diode"], 0.4, color=C_D, label="diode")
    ax.set_yticks(y); ax.set_yticklabels(piv.index, fontsize=8); ax.invert_yaxis(); ax.set_xscale("log")
    ax.set_xlabel("lifetime [years]"); ax.legend(); ax.grid(alpha=0.3, which="both"); ax.set_title("Same 30 days, different lifetime models")
    save_fig(fig, d, "lifetime_by_model.png")
    # self-consistency: Semikron formula entered as digitised curves
    sk = Semikron2024(k_thickness={"IGBT": 1, "Diode": 0.65})
    curves = {dev: {Tc: {"dTj": list(np.arange(5, 125, 5.0)),
                         "Nf": list(sk.bind(dev)(np.arange(5, 125, 5.0), Tc + 273.15 - np.arange(5, 125, 5.0) / 2, np.full(24, 2.0)))}
                    for Tc in (100, 125, 150)} for dev in ("IGBT", "Diode")}
    spec = {"model": "curves", "name": "semikron_as_curves", "temperature": "Tj_max", "curves": curves, "t_on_s": 2.0,
            "ton_correction": {"C": 1.0, "gamma": -0.75}, "t_on_min": 0.04}
    r2 = run(P, Q, T, lifetime_model=spec, MC_samples=200)
    rng = np.random.default_rng(0)
    pts = [(x, Tc, 3e14 * x ** -5.2 * np.exp(1e-19 / (1.380649e-23 * (Tc + 273.15))) * np.exp(rng.normal(0, 0.05)))
           for x in (40, 60, 80, 100) for Tc in (100, 125, 150)]
    fit = fit_cma(pts, temperature="Tj_max")
    checks = pd.DataFrame([
        {"check": "Semikron formula entered as 3 digitised curves: IGBT lifetime ratio", "value": r2["IGBT"]["lifetime_years"] / r["IGBT"]["lifetime_years"]},
        {"check": "same, diode", "value": r2["Diode"]["lifetime_years"] / r["Diode"]["lifetime_years"]},
        {"check": "fit_cma on 12 noisy points: n (true 5.2)", "value": fit["n"]},
        {"check": "fit_cma: Ea in 1e-19 J (true 1.0)", "value": fit["Ea_J"] / 1e-19}])
    checks.to_csv(d / "checks.csv", index=False)
    summary(d, "06 - Lifetime models",
            "The thermal part is computed once; every model then counts damage on the same cycles. The models agree on large, "
            "slow cycles but disagree strongly on small, short 50 Hz cycles (left plot), which is why the lifetimes differ by "
            "orders of magnitude. USER_CURVES_EXAMPLE uses invented numbers to show the format. Bayerer uses placeholder bond-wire data.",
            [("Lifetime by model (30 days)", df), ("Checks of the user-curve and fitting options", checks)])


if __name__ == "__main__":
    main()
