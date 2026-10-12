"""Study 03 - why Tj = T_amb + P(t)*Rth gives absurd 50 Hz swings, and which shortcuts work."""
from _common import *
from Input_parameters import Input_parameters_class
from Calculation_functions import Calculation_functions_class as CF
from Thermal_model import ThermalNetwork, ripple_from_losses
from Electrical_model import compute_bin_losses


def rect(Pm, r, tau, T=0.02, D=0.5):
    tp = D * T
    return Pm / D * np.sum(r * (-np.expm1(-tp / tau)) * (-np.expm1(-(T - tp) / tau)) / (-np.expm1(-T / tau)))


def main():
    d = outdir("03_why_Rth_overshoots")
    p0 = Input_parameters_class(*constant(S_RATED / 2, 0, 3), N_parallel=1)
    net = ThermalNetwork.from_params(p0)
    f = np.logspace(-4, 3.3, 300); H = net.H(2 * np.pi * f)
    rows = [{"frequency_Hz": fz, "IGBT_Tj_per_W_IGBT": abs(net.H([2 * np.pi * fz])[0, 0, 0]),
             "Diode_Tj_per_W_Diode": abs(net.H([2 * np.pi * fz])[1, 1, 0]), "case_per_W_total": abs(net.H([2 * np.pi * fz])[2, 0, 0])}
            for fz in (0, 0.001, 0.01, 0.1, 1, 10, 50, 100, 150)]
    zt = table(rows, d, "thermal_impedance.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(f, abs(H[0, 0]), color=C_I, label="Tj IGBT <- P IGBT (chip + case + sink)")
    ax.loglog(f, abs(H[1, 1]), color=C_D, label="Tj diode <- P diode")
    ax.loglog(f, abs(H[2, 0]), color="tab:green", label="case <- P (paste + heat sink)")
    ax.axvline(50, color="k", ls=":"); ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("|Zth| [K/W]")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both"); ax.set_title("Rth is |Zth| at 0 Hz; at 50 Hz the heat sink is gone")
    save_fig(fig, d, "thermal_impedance_vs_frequency.png")

    res = []
    for lf in (1.0, 0.5, 0.2):
        for pf in (1.0, 0.7, 0.4, 0.0):
            p = Input_parameters_class(*constant(lf * S_RATED * pf, lf * S_RATED * np.sqrt(1 - pf ** 2), 3), N_parallel=1)
            bins = CF.bin_operating_points(p.Is[:1], p.phi[:1], p.V_dc[:1], p.ma[:1], None)
            p.Np = 4000; PI, PD = compute_bin_losses(p, bins)
            RI, RD = ripple_from_losses(net, PI, PD, 50.0)
            ref = (np.ptp(RI), np.ptp(RD))
            Np_lo = 4000
            w = 2 * np.pi * 50 * np.arange(Np_lo // 2 + 1)
            Hw = net.H(w); FI, FD = np.fft.rfft(PI, axis=1), np.fft.rfft(PD, axis=1)
            def harm(nmax, k):
                keep = (np.arange(w.size) >= 1) & (np.arange(w.size) <= nmax)
                R = np.fft.irfft(keep * (FI * Hw[k, 0] + FD * Hw[k, 1]), n=Np_lo, axis=1)
                return np.ptp(R)
            Rja_I = abs(net.H([0.0])[0, 0, 0]); Rja_D = abs(net.H([0.0])[1, 1, 0])
            for k, n, rr, tt in ((0, "IGBT", p.r_I, p.tau_I), (1, "Diode", p.r_D, p.tau_D)):
                Pw = PI if k == 0 else PD
                res.append({"load": lf, "pf": pf, "device": n, "true_dTj_K": ref[k],
                            "naive_P(t)*Rth_ja_%": 100 * (np.ptp(Pw) * (Rja_I if k == 0 else Rja_D) / ref[k] - 1),
                            "P(t)*Rth_jc_%": 100 * (np.ptp(Pw) * rr.sum() / ref[k] - 1),
                            "1st_harmonic_%": 100 * (harm(1, k) / ref[k] - 1),
                            "harmonics_1-5_%": 100 * (harm(5, k) / ref[k] - 1),
                            "rectangular_pulse_formula_%": 100 * (rect(Pw.mean(), rr, tt) / ref[k] - 1)})
    df = table(res, d, "ripple_method_errors.csv")
    g = df.groupby("device")[["naive_P(t)*Rth_ja_%", "P(t)*Rth_jc_%", "1st_harmonic_%", "harmonics_1-5_%", "rectangular_pulse_formula_%"]]
    lo, hi = g.min(), g.max()
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(lo.shape[1])
    for i, n in enumerate(lo.index):
        ax.bar(x + 0.2 * (2 * i - 1), hi.loc[n] - lo.loc[n], bottom=lo.loc[n], width=0.4, color=[C_I, C_D][i], alpha=0.7, label=n)
    ax.set_xticks(x); ax.set_xticklabels([c.replace("_%", "") for c in lo.columns], rotation=15, fontsize=8)
    ax.set_yscale("symlog", linthresh=10); ax.axhline(0, color="k"); ax.set_ylabel("error range vs exact [%]")
    ax.legend(); ax.grid(alpha=0.3); ax.set_title("Shortcuts for the 50 Hz swing: only Zth harmonics 1-5 are accurate")
    save_fig(fig, d, "ripple_method_error_ranges.png")
    summary(d, "03 - Why P(t)*Rth overshoots",
            "|Zth| falls with frequency. Rth (0 Hz) is right for the mean temperature; the 50 Hz ripple sees |Zth| at 50, 100, "
            "150 Hz..., where the heat sink contributes almost nothing. Using Rth for the ripple therefore overshoots by "
            "hundreds of percent. Columns '..._%' are errors relative to the exact periodic solution.",
            [("Thermal impedance [K/W]", zt), ("Error of each shortcut [%] (min..max over 12 operating points)",
              pd.concat([lo.add_suffix(" min"), hi.add_suffix(" max")], axis=1).reset_index())])


if __name__ == "__main__":
    main()
