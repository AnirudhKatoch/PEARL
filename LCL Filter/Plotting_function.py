import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, no tkinter windows

plt.rcParams.update({"font.size": 15, "font.family": "Times New Roman", "axes.labelsize": 15, "axes.titlesize": 15,"xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15})

class Plotting_functions_class:

    @staticmethod
    def plot_df_1_power_flow_RMS(df_1_power_flow_RMS, figures_dir,xlabel):
        """
        Plot the per-second power-flow RMS profile (df_1) into three figures
        saved in figures_dir. x- and y-limits are taken from the data
        (min/max of the plotted signals), with a small fractional y-margin.

            Fig_1 : S_RMS, P_RMS, Q_RMS   (single axes)        6.4 x 4.8
            Fig_2 : Vg_RMS / Ig_RMS       (2 stacked subplots) 6.4 x 7.2
            Fig_3 : pf                    (single axes)        6.4 x 4.8
        """

        y_margin = 0.05

        df = df_1_power_flow_RMS
        x = np.arange(len(df))
        x_range = (x[0], x[-1])

        def ylim_from(*cols):
            """min/max across the given columns, with fractional margin."""
            lo = min(np.min(df[c]) for c in cols)
            hi = max(np.max(df[c]) for c in cols)
            if lo == hi:  # flat signal: avoid zero-height axis
                pad = abs(lo) * y_margin or 1.0
            else:
                pad = (hi - lo) * y_margin
            return lo - pad, hi + pad

        # ---------- Fig_1 : S, P, Q ----------
        fig1, ax1 = plt.subplots(figsize=(6.4, 4.8))
        ax1.plot(x, df["S_RMS"], label="S_RMS [VA]")
        ax1.plot(x, df["P_RMS"], label="P_RMS [W]")
        ax1.plot(x, df["Q_RMS"], label="Q_RMS [Var]")
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel("Power")
        ax1.set_title("Apparent / Active / Reactive power")
        ax1.set_xlim(x_range)
        ax1.set_ylim(ylim_from("S_RMS", "P_RMS", "Q_RMS"))  # shared scale across the three
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        fig1.savefig(f"{figures_dir}/Fig_1_power_SPQ.png", dpi=300)
        plt.close(fig1)

        # ---------- Fig_2 : Vg_RMS / Ig_RMS (stacked, taller) ----------
        fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(6.4, 4.8*1.5), sharex=True)

        ax2a.plot(x, df["Vg_RMS"], color="tab:blue", label="Vg_RMS [V]")
        ax2a.set_ylabel("Voltage [V]")
        ax2a.set_title("Grid Voltage (RMS)")
        ax2a.set_ylim(ylim_from("Vg_RMS"))
        ax2a.legend()
        ax2a.grid(True, alpha=0.3)

        ax2b.plot(x, df["Ig_RMS"], color="tab:orange", label="Ig_RMS [A]")
        ax2b.set_xlabel(xlabel)
        ax2b.set_ylabel("Current [A]")
        ax2b.set_title("Grid Current (RMS)")
        ax2b.set_ylim(ylim_from("Ig_RMS"))
        ax2b.legend()
        ax2b.grid(True, alpha=0.3)

        ax2b.set_xlim(x_range)
        fig2.tight_layout()
        fig2.savefig(f"{figures_dir}/Fig_2_Vg_Ig.png", dpi=300)
        plt.close(fig2)

        # ---------- Fig_3 : pf ----------
        fig3, ax3 = plt.subplots(figsize=(6.4, 4.8))
        ax3.plot(x, df["pf"], color="tab:green", label="pf [-]")
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel("Power factor [-]")
        ax3.set_title("Power factor")
        ax3.set_xlim(x_range)
        ax3.set_ylim(ylim_from("pf"))
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        fig3.savefig(f"{figures_dir}/Fig_3_pf.png", dpi=300)
        plt.close(fig3)


        # ---------- Fig_3 : pf ----------
        fig3a, ax3a = plt.subplots(figsize=(6.4, 4.8))
        ax3a.plot(x, df["T_amb"]-273, color="blue", label="T_amb")
        ax3.set_xlabel(xlabel)
        ax3a.set_ylabel("Temperature [°C]")
        ax3a.set_title("Ambient Temperature")
        ax3a.set_xlim(x_range)
        #ax3a.set_ylim(min(df["T_amb"]),max(df["T_amb"]))
        ax3a.legend()
        ax3a.grid(True, alpha=0.3)
        fig3a.tight_layout()
        fig3a.savefig(f"{figures_dir}/Fig_3a_T_amb.png", dpi=300)
        plt.close(fig3a)

    @staticmethod
    def plot_df_2_power_flow_inst(df_2_power_flow_inst, figures_dir, resolution_per_cycle, f=50, t=None, y_margin=0.05, xlabel = "Time [s]"):

        """
        Plot the last fundamental cycle of the instantaneous waveforms (df_2)
        into figures saved in figures_dir. Only the final resolution_per_cycle
        samples are shown. x/y limits are taken from the plotted data.

            Fig_4  : pf_inst / phi        (2 stacked subplots)  6.4 x 7.2
            Fig_5  : Ig_ref / Vs_ref      (2 stacked subplots)  6.4 x 7.2
            Fig_6  : Vs                   (single axes)         6.4 x 4.8
            Fig_7  : V_L1 / I_L1          (2 stacked subplots)  6.4 x 7.2
            Fig_8  : V_C  / I_C           (2 stacked subplots)  6.4 x 7.2
            Fig_9  : V_L2 / I_L2          (2 stacked subplots)  6.4 x 7.2
            Fig_10 : THD_percent          (single bar)          6.4 x 4.8
        """



        df = df_2_power_flow_inst
        last = slice(-resolution_per_cycle, None)

        # x-axis: real time if provided, else sample index within the cycle
        if t is not None:
            x = np.asarray(t)[last]
        else:
            # build real-time axis: one fundamental cycle = resolution_per_cycle samples
            dt = (1.0 / f) / resolution_per_cycle
            x = np.arange(resolution_per_cycle) * dt
        x_range = (x[0], x[-1]+dt)
        x_tick = (x_range[1] - x_range[0]) / 4.0  # 0.02 s span → ticks every 0.005 s

        def ylim_from(*series):
            lo = min(np.min(s) for s in series)
            hi = max(np.max(s) for s in series)
            if lo == hi:
                pad = abs(lo) * y_margin or 1.0
            else:
                pad = (hi - lo) * y_margin
            return lo - pad, hi + pad

        def col(name):
            return df[name].to_numpy()[last]

        # ---- helper for a 2-row stacked figure ----
        def stacked(fname, top_name, top_color, top_title, top_ylabel, bot_name, bot_color, bot_title, bot_ylabel, xlabel):
            top = col(top_name)
            bot = col(bot_name)
            fig, (axa, axb) = plt.subplots(2, 1, figsize=(6.4, 4.8*1.5), sharex=True)

            axa.plot(x, top, color=top_color, label=top_name)
            axa.set_ylabel(top_ylabel)
            axa.set_title(top_title)  # own heading
            axa.set_ylim(ylim_from(top))
            axa.legend()
            axa.grid(True, alpha=0.3)

            axb.plot(x, bot, color=bot_color, label=bot_name)
            axb.set_xlabel(xlabel)
            axb.set_ylabel(bot_ylabel)
            axb.set_title(bot_title)  # own heading
            axb.set_ylim(ylim_from(bot))
            axb.legend()
            axb.grid(True, alpha=0.3)

            axb.set_xlim(x_range)
            axb.xaxis.set_major_locator(MultipleLocator(x_tick))  # in stacked
            fig.tight_layout()
            fig.savefig(f"{figures_dir}/{fname}.png", dpi=300)
            plt.close(fig)

            # ---- helper for a single-axes figure ----
        def single(fname, name, color, title, ylabel, xlabel):
            y = col(name)
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            ax.plot(x, y, color=color, label=name)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xlim(x_range)
            ax.xaxis.set_major_locator(MultipleLocator(x_tick))  # in single
            ax.set_ylim(ylim_from(y))
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(f"{figures_dir}/{fname}.png", dpi=300)
            plt.close(fig)

        # ---------- Fig_4 : pf_inst / phi ----------
        stacked("Fig_4_pf_phi",
                    "pf_inst", "tab:green", "Instantaneous power factor", "Power factor [-]",
                    "phi", "tab:purple", "Phase angle", "Phase angle [rad]",xlabel)

        # ---------- Fig_5 : Ig_ref / Vs_ref ----------
        stacked("Fig_5_Igref_Vg",
                    "Ig_ref", "tab:orange", "Reference grid current", "Current [A]",
                    "Vg", "tab:blue", "Grid voltage", "Voltage [V]",xlabel)

        # ---------- Fig_6 : Vs ----------
        single("Fig_6_Vs", "Vs", "tab:blue",
                   "Inverter switching output Vs", "Voltage [V]",xlabel)

        # ---------- Fig_7 : V_L1 / I_L1 ----------
        stacked("Fig_7_L1",
                    "V_L1", "tab:red", "Inverter side inductor voltage", "Voltage [V]",
                    "I_L1", "tab:orange", "Inverter side inductor current", "Current [A]",xlabel)

        # ---------- Fig_8 : V_C / I_C ----------
        stacked("Fig_8_C",
                    "V_C", "tab:red", "Capacitor voltage", "Voltage [V]",
                    "I_C", "tab:orange", "Capacitor current", "Current [A]",xlabel)

        # ---------- Fig_9 : V_L2 / I_L2 ----------
        stacked("Fig_9_L2",
                    "V_L2", "tab:red", "Grid side inductor voltage", "Voltage [V]",
                    "I_L2", "tab:orange", "Grid side inductor current", "Current [A]",xlabel)

        # ---------- Fig_11 : I_L2 vs Ig_ref ----------
        stacked("Fig_11_I_L2_vs_Ig_ref",
                    "I_L2", "tab:red", "Grid-side current I_L2", "Current [A]",
                    "Ig_ref", "tab:orange", "Reference current Ig_ref", "Current [A]",xlabel)


        # ---------- Fig_10 : THD_percent_I_L2 (single bar) ----------
        thd = df["THD_percent_I_L2"].dropna()
        thd_val = float(thd.iloc[-1]) if len(thd) else np.nan
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.bar(["I_L2"], [thd_val], color="tab:cyan", width=0.4)
        ax.set_ylabel("THD [%]")
        ax.set_title("Total Harmonic Distortion")
        ax.set_ylim(0, thd_val * (1 + y_margin) if thd_val > 0 else 1.0)
        ax.text(0, thd_val, f"{thd_val:.4f} %", ha="center", va="bottom")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{figures_dir}/Fig_10_THD_I_L2.png", dpi=300)
        plt.close(fig)

        '''
        

        # ---------- Fig_15 : V_L1 / I_C / V_L2 (switching-dominated waveforms) ----------
        V_L1 = col("V_L1")
        I_C = col("I_C")
        V_L2 = col("V_L2")
        fig15, (ax15a, ax15b, ax15c) = plt.subplots(3, 1, figsize=(6.4, 4.8 * 3 * 0.5), sharex=True)

        ax15a.plot(x, V_L1, color="tab:red", label="V_L1")
        ax15a.set_ylabel("Voltage [V]")
        ax15a.set_title("Inverter-side inductor voltage")
        ax15a.set_ylim(ylim_from(V_L1))
        #ax15a.legend()
        ax15a.grid(True, alpha=0.3)

        ax15b.plot(x, I_C, color="tab:orange", label="I_C")
        ax15b.set_ylabel("Current [A]")
        ax15b.set_title("Capacitor current")
        ax15b.set_ylim(ylim_from(I_C))
        #ax15b.legend()
        ax15b.grid(True, alpha=0.3)

        ax15c.plot(x, V_L2, color="tab:purple", label="V_L2")
        ax15c.set_xlabel(xlabel)
        ax15c.set_ylabel("Voltage [V]")
        ax15c.set_title("Grid-side inductor voltage")
        ax15c.set_ylim(ylim_from(V_L2))
        #ax15c.legend()
        ax15c.grid(True, alpha=0.3)

        ax15c.set_xlim(x_range)
        ax15c.xaxis.set_major_locator(MultipleLocator(x_tick))
        fig15.tight_layout()
        fig15.savefig(f"{figures_dir}/Fig_15_switching_waveforms.pdf", dpi=300)
        plt.close(fig15)
        '''

    @staticmethod
    def plot_df_components(df_3_C, df_4_L1, df_5_L2, figures_dir, xlabel, y_margin=0.05):
        """
        Plot per-second component results (df_3_C, df_4_L1, df_5_L2) and the
        aggregated lifetimes into figures saved in figures_dir.
        """

        n = len(df_3_C)
        x = np.arange(n)
        x_range = (x[0], x[-1])

        def ylim_from(*arrs):
            lo = min(np.nanmin(a) for a in arrs)
            hi = max(np.nanmax(a) for a in arrs)
            if lo == hi:
                pad = abs(lo) * y_margin or 1.0
            else:
                pad = (hi - lo) * y_margin
            return lo - pad, hi + pad

        # ---------- Fig_12 : V_C_RMS / I_C_RMS ----------
        V_C_RMS = df_3_C["V_C_RMS"].to_numpy()
        I_C_RMS = df_3_C["I_C_RMS"].to_numpy()
        fig12, (ax12a, ax12b) = plt.subplots(2, 1, figsize=(6.4, 4.8 * 1.5), sharex=True)
        ax12a.plot(x, V_C_RMS, color="tab:red", label="V_C_RMS")
        ax12a.set_ylabel("Voltage [V]")
        ax12a.set_title("Capacitor C - RMS voltage")
        ax12a.set_ylim(ylim_from(V_C_RMS))
        ax12a.legend()
        ax12a.grid(True, alpha=0.3)
        ax12b.plot(x, I_C_RMS, color="tab:orange", label="I_C_RMS")
        ax12b.set_xlabel(xlabel)
        ax12b.set_ylabel("Current [A]")
        ax12b.set_title("Capacitor C - RMS current")
        ax12b.set_ylim(ylim_from(I_C_RMS))
        ax12b.legend()
        ax12b.grid(True, alpha=0.3)
        ax12b.set_xlim(x_range)
        fig12.tight_layout()
        fig12.savefig(f"{figures_dir}/Fig_12_C_VI.png", dpi=300)
        plt.close(fig12)

        # ---------- Fig_13 : V_L1_RMS / I_L1_RMS ----------
        V_L1_RMS = df_4_L1["V_L1_RMS"].to_numpy()
        I_L1_RMS = df_4_L1["I_L1_RMS"].to_numpy()
        fig13, (ax13a, ax13b) = plt.subplots(2, 1, figsize=(6.4, 4.8 * 1.5), sharex=True)
        ax13a.plot(x, V_L1_RMS, color="tab:red", label="V_L1_RMS")
        ax13a.set_ylabel("Voltage [V]")
        ax13a.set_title("Inductor L1 - RMS voltage")
        ax13a.set_ylim(ylim_from(V_L1_RMS))
        ax13a.legend()
        ax13a.grid(True, alpha=0.3)
        ax13b.plot(x, I_L1_RMS, color="tab:orange", label="I_L1_RMS")
        ax13b.set_xlabel(xlabel)
        ax13b.set_ylabel("Current [A]")
        ax13b.set_title("Inductor L1 - RMS current")
        ax13b.set_ylim(ylim_from(I_L1_RMS))
        ax13b.legend()
        ax13b.grid(True, alpha=0.3)
        ax13b.set_xlim(x_range)
        fig13.tight_layout()
        fig13.savefig(f"{figures_dir}/Fig_13_L1_VI.png", dpi=300)
        plt.close(fig13)

        # ---------- Fig_14 : V_L2_RMS / I_L2_RMS ----------
        V_L2_RMS = df_5_L2["V_L2_RMS"].to_numpy()
        I_L2_RMS = df_5_L2["I_L2_RMS"].to_numpy()
        fig14, (ax14a, ax14b) = plt.subplots(2, 1, figsize=(6.4, 4.8 * 1.5), sharex=True)
        ax14a.plot(x, V_L2_RMS, color="tab:red", label="V_L2_RMS")
        ax14a.set_ylabel("Voltage [V]")
        ax14a.set_title("Inductor L2 - RMS voltage")
        ax14a.set_ylim(ylim_from(V_L2_RMS))
        ax14a.legend()
        ax14a.grid(True, alpha=0.3)
        ax14b.plot(x, I_L2_RMS, color="tab:orange", label="I_L2_RMS")
        ax14b.set_xlabel(xlabel)
        ax14b.set_ylabel("Current [A]")
        ax14b.set_title("Inductor L2 - RMS current")
        ax14b.set_ylim(ylim_from(I_L2_RMS))
        ax14b.legend()
        ax14b.grid(True, alpha=0.3)
        ax14b.set_xlim(x_range)
        fig14.tight_layout()
        fig14.savefig(f"{figures_dir}/Fig_14_L2_VI.png", dpi=300)
        plt.close(fig14)

        # ---------- Fig_15 : P_total C / L1 / L2 ----------
        P_total_C = df_3_C["P_total_C"].to_numpy()
        P_total_L1 = df_4_L1["P_total_L1"].to_numpy()
        P_total_L2 = df_5_L2["P_total_L2"].to_numpy()
        fig15, (ax15a, ax15b, ax15c) = plt.subplots(3, 1, figsize=(6.4, 4.8 * 3), sharex=True)
        ax15a.plot(x, P_total_C, color="tab:green", label="P_total_C")
        ax15a.set_ylabel("Power loss [W]")
        ax15a.set_title("Total power loss - Capacitor C")
        ax15a.set_ylim(ylim_from(P_total_C))
        ax15a.legend()
        ax15a.grid(True, alpha=0.3)
        ax15b.plot(x, P_total_L1, color="tab:blue", label="P_total_L1")
        ax15b.set_ylabel("Power loss [W]")
        ax15b.set_title("Total power loss - Inductor L1")
        ax15b.set_ylim(ylim_from(P_total_L1))
        ax15b.legend()
        ax15b.grid(True, alpha=0.3)
        ax15c.plot(x, P_total_L2, color="tab:purple", label="P_total_L2")
        ax15c.set_xlabel(xlabel)
        ax15c.set_ylabel("Power loss [W]")
        ax15c.set_title("Total power loss - Inductor L2")
        ax15c.set_ylim(ylim_from(P_total_L2))
        ax15c.legend()
        ax15c.grid(True, alpha=0.3)
        ax15c.set_xlim(x_range)
        fig15.tight_layout()
        fig15.savefig(f"{figures_dir}/Fig_15_P_total.png", dpi=300)
        plt.close(fig15)

        # ---------- Fig_16 : core / winding losses (L1, L2) ----------
        P_c_L1 = df_4_L1["P_c_L1"].to_numpy()
        P_c_L2 = df_5_L2["P_c_L2"].to_numpy()
        P_w_L1 = df_4_L1["P_w_L1"].to_numpy()
        P_w_L2 = df_5_L2["P_w_L2"].to_numpy()
        fig16, (ax16a, ax16b) = plt.subplots(2, 1, figsize=(6.4, 4.8 * 1.5), sharex=True)
        ax16a.plot(x, P_c_L1, color="tab:blue", label="P_c_L1")
        ax16a.plot(x, P_c_L2, color="tab:purple", label="P_c_L2")
        ax16a.set_ylabel("Core loss [W]")
        ax16a.set_title("Core losses (L1, L2)")
        ax16a.set_ylim(ylim_from(P_c_L1, P_c_L2))
        ax16a.legend()
        ax16a.grid(True, alpha=0.3)
        ax16b.plot(x, P_w_L1, color="tab:blue", label="P_w_L1")
        ax16b.plot(x, P_w_L2, color="tab:purple", label="P_w_L2")
        ax16b.set_xlabel(xlabel)
        ax16b.set_ylabel("Winding loss [W]")
        ax16b.set_title("Winding losses (L1, L2)")
        ax16b.set_ylim(ylim_from(P_w_L1, P_w_L2))
        ax16b.legend()
        ax16b.grid(True, alpha=0.3)
        ax16b.set_xlim(x_range)
        fig16.tight_layout()
        fig16.savefig(f"{figures_dir}/Fig_16_core_winding.png", dpi=300)
        plt.close(fig16)

        # ---------- Fig_17 : temperatures C / L1 / L2 ----------
        T_C = df_3_C["T_C"].to_numpy() - 273.15
        T_L1 = df_4_L1["T_inductor_L1"].to_numpy() - 273.15
        T_L2 = df_5_L2["T_inductor_L2"].to_numpy() - 273.15
        fig17, (ax17a, ax17b, ax17c) = plt.subplots(3, 1, figsize=(6.4, 4.8 * 3), sharex=True)
        ax17a.plot(x, T_C, color="tab:green", label="T_C")
        ax17a.set_ylabel("Temperature [°C]")
        ax17a.set_title("Capacitor C temperature")
        ax17a.set_ylim(ylim_from(T_C))
        ax17a.legend()
        ax17a.grid(True, alpha=0.3)
        ax17b.plot(x, T_L1, color="tab:blue", label="T_L1")
        ax17b.set_ylabel("Temperature [°C]")
        ax17b.set_title("Inductor L1 temperature")
        ax17b.set_ylim(ylim_from(T_L1))
        ax17b.legend()
        ax17b.grid(True, alpha=0.3)
        ax17c.plot(x, T_L2, color="tab:purple", label="T_L2")
        ax17c.set_xlabel(xlabel)
        ax17c.set_ylabel("Temperature [°C]")
        ax17c.set_title("Inductor L2 temperature")
        ax17c.set_ylim(ylim_from(T_L2))
        ax17c.legend()
        ax17c.grid(True, alpha=0.3)
        ax17c.set_xlim(x_range)
        fig17.tight_layout()
        fig17.savefig(f"{figures_dir}/Fig_17_T.png", dpi=300)
        plt.close(fig17)

        # ---------- Fig_18 : lifetime bar ----------
        Lifetime_C = df_3_C["Lifetime_C"].dropna().iloc[-1]
        Lifetime_L1 = df_4_L1["Lifetime_L1"].dropna().iloc[-1]
        Lifetime_L2 = df_5_L2["Lifetime_L2"].dropna().iloc[-1]
        life = [Lifetime_C, Lifetime_L1, Lifetime_L2]
        fig18, ax18 = plt.subplots(figsize=(6.4, 4.8))
        bars = ax18.bar(["C", "L1", "L2"], life, color=["tab:green", "tab:blue", "tab:purple"], width=0.5)
        ax18.set_ylabel("Lifetime [years]")
        ax18.set_title("Predicted lifetime per component")
        ax18.set_ylim(0, max(life) * (1 + y_margin))
        for b, v in zip(bars, life):
            ax18.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom")
        ax18.grid(True, axis="y", alpha=0.3)
        fig18.tight_layout()
        fig18.savefig(f"{figures_dir}/Fig_18_lifetime.png", dpi=300)
        plt.close(fig18)

        # ---------- Fig_19 : lifetime consumed bar ----------
        Consumed_C = df_3_C["Lifetime_consumed_C"].dropna().iloc[-1]
        Consumed_L1 = df_4_L1["Lifetime_consumed_L1"].dropna().iloc[-1]
        Consumed_L2 = df_5_L2["Lifetime_consumed_L2"].dropna().iloc[-1]
        consumed = [Consumed_C, Consumed_L1, Consumed_L2]
        fig19, ax19 = plt.subplots(figsize=(6.4, 4.8))
        bars = ax19.bar(["C", "L1", "L2"], consumed, color=["tab:green", "tab:blue", "tab:purple"], width=0.5)
        ax19.set_ylabel("Lifetime consumed [%]")
        ax19.set_title("Lifetime consumed per component")
        ax19.set_ylim(0, max(consumed) * (1 + y_margin))
        for b, v in zip(bars, consumed):
            ax19.text(b.get_x() + b.get_width() / 2, v,f"{v:.2f}", ha="center", va="bottom")
        ax19.grid(True, axis="y", alpha=0.3)
        fig19.tight_layout()
        fig19.savefig(f"{figures_dir}/Fig_19_lifetime_consumed.png", dpi=300)
        plt.close(fig19)

        # ---------- Fig_20 : average temperature bar ----------
        Temp_C = (df_3_C["T_C"].dropna() - 273).mean()
        Temp_L1 = (df_4_L1["T_inductor_L1"].dropna() - 273).mean()
        Temp_L2 = (df_5_L2["T_inductor_L2"].dropna() - 273).mean()
        temps = [Temp_C, Temp_L1, Temp_L2]
        fig20, ax20 = plt.subplots(figsize=(6.4, 4.8))
        bars = ax20.bar(["C", "L1", "L2"], temps, color=["tab:green", "tab:blue", "tab:purple"], width=0.5)
        ax20.set_ylabel("Average temperature [°C]")
        ax20.set_title("Average temperature per component")
        ax20.set_ylim(0, max(temps) * (1 + y_margin))
        for b, v in zip(bars, temps):
            ax20.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom")
        ax20.grid(True, axis="y", alpha=0.3)
        fig20.tight_layout()
        fig20.savefig(f"{figures_dir}/Fig_20_average_temperature.png", dpi=300)
        plt.close(fig20)

    @staticmethod
    def plot_lifetime_monte_carlo(Lifetime_C_MC, Lifetime_L1_MC, Lifetime_L2_MC, Lifetime_LCL_MC,figures_dir,
                                  B10_C=None, B10_L1=None, B10_L2=None, B10_LCL=None,plot_type="histogram", bins=50):

        """
        Plot Monte Carlo lifetime distributions as four separate figures
        (Capacitor, L1, L2, LCL filter), each saved individually to figures_dir.

        Parameters
        ----------
        Lifetime_C_MC, Lifetime_L1_MC, Lifetime_L2_MC, Lifetime_LCL_MC : np.ndarray
            Monte Carlo lifetime samples [years].
        figures_dir : str
            Directory to save the figures into (created if missing).
        plot_type : str
            "histogram" -> bar histogram
            "line"      -> smooth density curve (line)
        bins : int
            Histogram bin count (also used to build the line curve).
        """

        panels = [
            ("Capacitor C", Lifetime_C_MC, "Fig_21_lifetime_MC_C", "red"),
            ("Inverter-side inductor L1", Lifetime_L1_MC, "Fig_22_lifetime_MC_L1", "red"),
            ("Grid-side inductor L2", Lifetime_L2_MC, "Fig_23_lifetime_MC_L2", "red"),
            ("LCL filter", Lifetime_LCL_MC, "Fig_24_lifetime_MC_LCL", "red"),
        ]

        for title, data, fname, color in panels:
            data = np.asarray(data, dtype=float)
            data = data[np.isfinite(data)]  # drop NaN/inf

            fig, ax = plt.subplots(figsize=(6.4, 4.8))

            if plot_type == "histogram":
                ax.hist(data, bins=bins, color=color, edgecolor="black", linewidth=0.8)
                ax.set_ylabel("Count")

            elif plot_type == "line":
                # smooth density curve via Gaussian KDE

                kde = gaussian_kde(data)
                x = np.linspace(data.min(), data.max())
                ax.plot(x, kde(x), color=color)
                # ax.fill_between(x, kde(x), color=color, alpha=0.15)
                ax.set_ylabel("Probability density")

            else:
                raise ValueError("plot_type must be 'histogram' or 'line'.")

            ax.set_title(title)
            ax.set_xlabel("Lifetime [years]")
            #ax.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(f"{figures_dir}/{fname}.png", dpi=300)
            plt.close(fig)

            # ── B10 bar chart ─────────────────────────────────────────────
        B10_values = [B10_C, B10_L1, B10_L2, B10_LCL]
        if all(v is not None for v in B10_values):
            labels = ["Capacitor C", "Inductor L1", "Inductor L2", "LCL filter"]
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            bars = ax.bar(labels, B10_values, color=colors, edgecolor="black", linewidth=0.8)

            # annotate each bar with its value
            for bar, val in zip(bars, B10_values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.1f}", ha="center", va="bottom")

            ax.set_ylabel(r"$B_{10}$ lifetime [years]")
            ax.set_ylim(0, max(B10_values) * 1.15)  # headroom for labels

            fig.tight_layout()
            fig.savefig(f"{figures_dir}/Fig_25_B10_bar.png", dpi=300)
            plt.close(fig)

    '''

    @staticmethod
    def plot_Ig_ref_vs_I_L2(df_2_power_flow_inst, figures_dir, resolution_per_cycle, y_margin, f=50, t=None,xlabel="Time [s]"):
        """
        Compare the reference grid current (Ig_ref) against the delivered
        grid-side current (I_L2) over the last fundamental cycle, overlaid
        on a single axes. Ig_ref is drawn as a continuous line and I_L2 as
        sparse hollow markers on top. Saves Fig_I_ref_vs_IL_2.png in figures_dir.
        """

        df = df_2_power_flow_inst
        last = slice(-resolution_per_cycle, None)

        # x-axis: real time if provided, else build one fundamental cycle
        if t is not None:
            x = np.asarray(t)[last]
            dt = x[1] - x[0]
        else:
            dt = (1.0 / f) / resolution_per_cycle
            x = np.arange(resolution_per_cycle) * dt
        x_range = (x[0], x[-1] + dt)
        x_tick = (x_range[1] - x_range[0]) / 4.0

        def col(name):
            return df[name].to_numpy()[last]

        def ylim_from(*series):
            lo = min(np.min(s) for s in series)
            hi = max(np.max(s) for s in series)
            if lo == hi:
                pad = abs(lo) * y_margin or 1.0
            else:
                pad = (hi - lo) * y_margin
            return lo - pad, hi + pad

        Ig_ref = col("Ig_ref")
        I_L2 = col("I_L2")
        n = len(x)
        step = max(1, n // 50)  # ~50 markers across the cycle

        # ---------- Overlay: Ig_ref line + I_L2 sparse hollow markers ----------
        fig, ax = plt.subplots(figsize=(6.4, 4.8 * 0.525))

        ax.plot(x[::step], I_L2[::step], color="red",
                linestyle="none", marker="o", markersize=5,
                markerfacecolor="none", markeredgewidth=1.6,
                label="Actual current")
        ax.plot(x, Ig_ref, color="blue", linewidth=1.6, label="Reference current")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Current [A]")
        #ax.set_title("Reference vs delivered grid current")
        ax.set_xlim(x_range)
        ax.xaxis.set_major_locator(MultipleLocator(x_tick))
        ax.set_ylim(-1300,1300)
        ax.legend()
        #ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{figures_dir}/Fig_I_ref_vs_IL_2.pdf", dpi=300)
        plt.close(fig)

    @staticmethod
    def plot_six_waveforms(df_2_power_flow_inst, figures_dir, resolution_per_cycle, f=50, t=None, y_margin=0.05,xlabel="Time [s]"):
        """
        Plot the six LCL branch waveforms over the last fundamental cycle in a
        single 6-row, 1-column figure:

            Row 1 : V_L1   Row 2 : I_L1
            Row 3 : V_C    Row 4 : I_C
            Row 5 : V_L2   Row 6 : I_L2

        Saves Fig_six_waveforms.png in figures_dir.
        """

        df = df_2_power_flow_inst
        last = slice(-resolution_per_cycle, None)


        df = df.groupby(np.arange(len(df)) // 3).transform('mean')

        # x-axis: real time if provided, else build one fundamental cycle
        if t is not None:
            x = np.asarray(t)[last]
            dt = x[1] - x[0]
        else:
            dt = (1.0 / f) / resolution_per_cycle
            x = np.arange(resolution_per_cycle) * dt
        x_range = (x[0], x[-1] + dt)
        x_tick = (x_range[1] - x_range[0]) / 4.0

        def col(name):
            return df[name].to_numpy()[last]

        def ylim_from(s):
            lo, hi = np.min(s), np.max(s)
            if lo == hi:
                pad = abs(lo) * y_margin or 1.0
            else:
                pad = (hi - lo) * y_margin
            return lo - pad, hi + pad

        # name, color, ylabel, title
        rows = [
            ("V_L1", "blue", "Voltage [V]", r"Inverter-side inductor voltage ($V_{L1}$)"),
            ("I_L1", "red", "Current [A]", r"Inverter-side inductor current ($I_{L1}$)"),
            ("V_C", "blue", "Voltage [V]", r"Capacitor voltage ($V_{C}$)"),
            ("I_C", "red", "Current [A]", r"Capacitor current ($I_{C}$)"),
            ("V_L2", "blue", "Voltage [V]", r"Grid-side inductor voltage ($V_{L2}$)"),
            ("I_L2", "red", "Current [A]", r"Grid-side inductor current ($I_{L2}$)"),
        ]

        fig, axes = plt.subplots(6, 1, figsize=(6.4, 4.8 * 6 * 0.33), sharex=True)

        for ax, (name, color, ylabel, title) in zip(axes, rows):
            y = col(name)
            ax.plot(x, y, color=color, label=name)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_ylim(ylim_from(y))

            #ax.legend(loc="upper right")
            #ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel(xlabel)
        axes[-1].set_xlim(x_range)
        axes[-1].xaxis.set_major_locator(MultipleLocator(x_tick))

        fig.tight_layout()
        #fig.subplots_adjust(hspace=1.0, top=1.0, bottom=0.0, left=0.0, right=1.0)
        fig.savefig(f"{figures_dir}/Python_benchmarking_visualization.pdf", dpi=300)
        plt.close(fig)
    '''
