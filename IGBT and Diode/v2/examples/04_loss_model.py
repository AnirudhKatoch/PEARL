"""Study 04 - instantaneous loss model: v3 (physical) vs textbook averages vs v1 (legacy)."""
from _common import *
from Input_parameters import Input_parameters_class
from Calculation_functions import Calculation_functions_class as CF


def main():
    d = outdir("04_loss_model")
    p = Input_parameters_class(*constant(S_RATED, 0, 3), N_parallel=1)
    Is, Vdc, ma = float(p.Is[0]), 600.0, float(p.ma[0])
    rows = []
    for deg in range(-180, 181, 15):
        phi = np.radians(deg)
        PI, PD = CF.loss_waveforms_physical([Is], [phi], [Vdc], [ma], 4000, True, p.f_sw, p.t_on, p.t_off, p.Err_D, p.I_ref,
                                            p.V_ref, p.R_IGBT, p.V_0_IGBT, p.R_D, p.V_0_D)
        LI, LD = CF.loss_waveforms_legacy([Is], [phi], [Vdc], 4000, 1.0, p.f_sw, p.t_on, p.t_off, p.Err_D, p.I_ref, p.V_ref,
                                          p.R_IGBT, p.V_0_IGBT, p.R_D, p.V_0_D)
        Ipk, c, m1 = np.sqrt(2) * Is, np.cos(phi), min(ma, 1.0)
        TI = (p.V_0_IGBT * Ipk * (1 / (2 * np.pi) + m1 * c / 8) + p.R_IGBT * Ipk ** 2 * (1 / 8 + m1 * c / (3 * np.pi))
              + Ipk * Vdc * (p.t_on + p.t_off) * p.f_sw / (2 * np.pi))
        TD = (p.V_0_D * Ipk * (1 / (2 * np.pi) - m1 * c / 8) + p.R_D * Ipk ** 2 * (1 / 8 - m1 * c / (3 * np.pi))
              + p.f_sw * p.Err_D * Ipk * Vdc / (np.pi * p.I_ref * p.V_ref))
        rows.append({"phi_deg": deg, "IGBT_v3_W": PI.mean(), "IGBT_textbook_W": TI, "IGBT_v1_legacy_W": LI.mean(),
                     "Diode_v3_W": PD.mean(), "Diode_textbook_W": TD, "Diode_v1_legacy_W": LD.mean()})
    df = table(rows, d, "average_losses_vs_angle.csv")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, n, col in ((ax[0], "IGBT", C_I), (ax[1], "Diode", C_D)):
        a.plot(df.phi_deg, df[f"{n}_v3_W"], color=col, lw=2, label="v3 instantaneous model (SVPWM)")
        a.plot(df.phi_deg, df[f"{n}_textbook_W"], "k--", label="textbook average (sine PWM, m <= 1)")
        a.plot(df.phi_deg, df[f"{n}_v1_legacy_W"], color="0.5", label="v1 per-time-step formulas")
        a.set_xlabel("phi = voltage leads current [deg]  (|phi| > 90: P < 0, charging)"); a.set_ylabel("mean loss [W]")
        a.set_title(f"{n}, Is = {Is:.0f} A rms"); a.legend(fontsize=7); a.grid(alpha=0.3)
    save_fig(fig, d, "average_losses_vs_angle.png")
    th = 2 * np.pi * np.arange(400) / 400; t = th / (2 * np.pi) * 20
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, deg, lab in ((ax[0], 0, "pf = 1, P > 0 (inverter)"), (ax[1], 180, "pf = 1, P < 0 (charging / rectifier)")):
        PI, PD = CF.loss_waveforms_physical([Is], [np.radians(deg)], [Vdc], [ma], 400, True, p.f_sw, p.t_on, p.t_off, p.Err_D,
                                            p.I_ref, p.V_ref, p.R_IGBT, p.V_0_IGBT, p.R_D, p.V_0_D)
        a.plot(t, PI[0], color=C_I, label="IGBT"); a.plot(t, PD[0], color=C_D, label="diode")
        a.set_xlabel("time in one period [ms]"); a.set_ylabel("loss [W]"); a.set_title(lab); a.legend(); a.grid(alpha=0.3)
    save_fig(fig, d, "loss_waveforms_one_period.png")
    pick = df[df.phi_deg.isin([0, 90, 180])]
    summary(d, "04 - Loss model",
            "v3 computes losses at every instant from the real phase current and the duty cycle; its period average matches "
            "the textbook average formulas (exactly for sine PWM with m <= 1; this operating point needs m = 1.08, so v3 uses "
            "SVPWM and differs slightly from the sine formula). The v1 per-time-step formulas give about half the IGBT loss.",
            [("Mean losses at 0, 90 and 180 degrees [W]", pick)])


if __name__ == "__main__":
    main()
