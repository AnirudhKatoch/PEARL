import matplotlib.pyplot as plt
import numpy as np

def Plotting_lifetime(df_IGBT, df_Diode, Nf_igbt_eq, lifetime_years_igbt, Nf_diode_eq, lifetime_years_diode, Figures_dir):

    # -------------------------------------------------
    # Figure 6: IGBT and Diode Lifetime
    # -------------------------------------------------

    fig6, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 6), sharex=False)

    ax1.plot(df_Diode["Nf_diode"])
    ax1.set_yscale("log")  # <-- LOG SCALE
    ax1.set_ylabel("Nf Diode")
    ax1.set_title("Diode – Number of cycles to failure")
    ax1.grid(True)

    ax2.plot(df_IGBT["Nf_igbt"])
    ax2.set_yscale("log")  # <-- LOG SCALE
    ax2.set_xlabel("Cycle index")
    ax2.set_ylabel("Nf IGBT")
    ax2.set_title("IGBT – Number of cycles to failure")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(Figures_dir / "6_IGBT_and_Diode_number_of_cycles_separate.png")
    plt.close(fig6)

    # -------------------------------------------------
    # Figure 7: DeltaT Comparison
    # -------------------------------------------------

    fig7, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 6), sharex=False)

    ax1.plot(df_Diode["deltaT_diode"])
    ax1.set_ylabel("ΔT Diode [C]")
    ax1.set_title("Diode – Temperature Swing (ΔT)")
    ax1.grid(True)

    ax2.plot(df_IGBT["deltaT_igbt"])
    ax2.set_xlabel("Cycle index")
    ax2.set_ylabel("ΔT IGBT [C]")
    ax2.set_title("IGBT – Temperature Swing (ΔT)")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(Figures_dir / "7_DeltaT_IGBT_and_Diode.png")
    plt.close(fig7)

    # -------------------------------------------------
    # Figure 8: Tmean Comparison
    # -------------------------------------------------

    fig8, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 6), sharex=False)

    ax1.plot(df_Diode["Tmean_diode"] - 273.15)
    ax1.set_ylabel("Tmean Diode [C]")
    ax1.set_title("Diode – Mean Temperature (Tmean)")
    ax1.grid(True)


    ax2.plot(df_IGBT["Tmean_igbt"] - 273.15)
    ax2.set_xlabel("Cycle index")
    ax2.set_ylabel("Tmean IGBT [C]")
    ax2.set_title("IGBT – Mean Temperature (Tmean)")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(Figures_dir / "8_Tmean_IGBT_and_Diode.png")
    plt.close(fig8)

    # -------------------------------------------------
    # Figure 9: Thermal Cycle Period
    # -------------------------------------------------

    fig9, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 6), sharex=False)

    ax1.plot(df_Diode["thermal_cycle_period_diode"])
    ax1.set_ylabel("Period Diode [s]")
    ax1.set_title("Diode – Thermal Cycle Period")
    ax1.set_ylim(0, 0.2)
    ax1.grid(True)

    ax2.plot(df_IGBT["thermal_cycle_period_igbt"])
    ax2.set_xlabel("Cycle index")
    ax2.set_ylabel("Period IGBT [s]")
    ax2.set_title("IGBT – Thermal Cycle Period")
    ax2.set_ylim(0, 0.2)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(Figures_dir / "9_ThermalCyclePeriod_IGBT_and_Diode.png")
    plt.close(fig9)

    # -------------------------------------------------
    # Figure 10: Cycle Count
    # -------------------------------------------------

    fig10, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 6), sharex=False)

    ax1.plot(df_Diode["count_diode"])
    ax1.set_ylabel("Count Diode")
    ax1.set_title("Diode – Cycle Count")
    ax1.grid(True)

    ax2.plot(df_IGBT["count_igbt"])
    ax2.set_xlabel("Cycle index")
    ax2.set_ylabel("Count IGBT")
    ax2.set_title("IGBT – Cycle Count")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(Figures_dir / "10_CycleCount_IGBT_and_Diode.png")
    plt.close(fig10)

    # -------------------------------------------------
    # Figure 11: Equivalent Nf Comparison (Twin Y-Axis)
    # -------------------------------------------------

    fig11, ax_left = plt.subplots(figsize=(6.4, 4.8))

    # Bar positions
    x_left = 0  # IGBT
    x_right = 1  # Diode

    # Plot IGBT bar on left axis
    ax_left.bar(x_left, Nf_igbt_eq, width=0.5, color="C0")
    ax_left.set_ylabel("IGBT Equivalent Nf")

    # Create right axis
    ax_right = ax_left.twinx()

    # Plot Diode bar on right axis
    ax_right.bar(x_right, Nf_diode_eq, width=0.5, color="C1")
    ax_right.set_ylabel("Diode Equivalent Nf")

    # Set title
    ax_left.set_title("IGBT vs Diode – Equivalent Number of Cycles to Failure")

    # Set x-ticks and labels
    ax_left.set_xticks([x_left, x_right])
    ax_left.set_xticklabels(["IGBT", "Diode"])

    # Grid on the left axis only
    ax_left.grid(axis="y")

    plt.tight_layout()
    plt.savefig(Figures_dir / "11_Equivalent_Nf_TwinAxis.png")
    plt.close(fig11)

    # -------------------------------------------------
    # Figure 12: Lifetime (Years) with Twin Y-Axis
    # -------------------------------------------------

    fig12, ax_left = plt.subplots(figsize=(6.4, 4.8))

    # Bar positions
    x_left = 0  # IGBT
    x_right = 1  # Diode

    # Plot IGBT bar on left axis
    ax_left.bar(x_left, lifetime_years_igbt, width=0.5, color="C0", label="IGBT")

    # Create right axis
    ax_right = ax_left.twinx()

    # Plot Diode bar on right axis
    ax_right.bar(x_right, lifetime_years_diode, width=0.5, color="C1", label="Diode")

    # Set axis labels
    ax_left.set_ylabel("IGBT Lifetime [years]")
    ax_right.set_ylabel("Diode Lifetime [years]")

    # Set title
    ax_left.set_title("IGBT vs Diode – Lifetime in Years ")

    # Set x-ticks and labels
    ax_left.set_xticks([x_left, x_right])
    ax_left.set_xticklabels(["IGBT", "Diode"])

    # Grid only for left axis
    ax_left.grid(axis="y")

    plt.tight_layout()
    plt.savefig(Figures_dir / "12_Lifetime_Years_TwinAxis.png")
    plt.close(fig12)


def Plotting_electrical(S,P,Q,Vs,Is,V_dc,pf,phi,T_env, Figures_dir):

    time_axis = np.arange(len(Is))

    # -------------------------------------------------
    # Figure 1: Apparent, Active, Reactive Power (S, P, Q)
    # -------------------------------------------------

    fig1, ax1 = plt.subplots(3, 1, figsize=(6.4, 6), sharex=True)

    ax1[0].plot(time_axis,S)
    ax1[0].set_ylabel("S [VA]")
    ax1[0].set_title("Apparent Power (S)")
    ax1[0].grid(True)
    ax1[0].set_xlim(min(time_axis),max(time_axis))

    ax1[1].plot(time_axis,P)
    ax1[1].set_ylabel("P [W]")
    ax1[1].set_title("Active Power (P)")
    ax1[1].grid(True)
    ax1[1].set_xlim(min(time_axis), max(time_axis))

    ax1[2].plot(time_axis,Q)
    ax1[2].set_ylabel("Q [VAR]")
    ax1[2].set_xlabel("Time [s]")
    ax1[2].set_title("Reactive Power (Q)")
    ax1[2].grid(True)
    ax1[2].set_xlim(min(time_axis), max(time_axis))

    plt.tight_layout()
    plt.savefig(Figures_dir / "1_S_P_Q.png")
    plt.close(fig1)

    # -------------------------------------------------
    # Figure 2: Voltage and Current (Vs, Is)
    # -------------------------------------------------

    fig2, ax2 = plt.subplots(2, 1, figsize=(6.4, 6), sharex=True)

    ax2[0].plot(time_axis,Vs)
    ax2[0].set_ylabel("Vs [V]")
    ax2[0].set_title("Supply Voltage (Vs)")
    ax2[0].grid(True)
    ax2[0].set_xlim(min(time_axis), max(time_axis))

    ax2[1].plot(time_axis,Is)
    ax2[1].set_ylabel("Is [A]")
    ax2[1].set_xlabel("Time [s]")
    ax2[1].set_title("Supply Current (Is)")
    ax2[1].grid(True)
    ax2[1].set_xlim(min(time_axis), max(time_axis))

    plt.tight_layout()
    plt.savefig(Figures_dir / "2_Vs_Is.png")
    plt.close(fig2)

    # -------------------------------------------------
    # Figure 3: Power Factor and Phase Angle (pf, phi)
    # -------------------------------------------------

    fig3, ax3 = plt.subplots(2, 1, figsize=(6.4, 6), sharex=True)

    ax3[0].plot(time_axis,pf)
    ax3[0].set_ylabel("pf [-]")
    ax3[0].set_title("Power Factor")
    ax3[0].grid(True)
    ax3[0].set_xlim(min(time_axis), max(time_axis))

    ax3[1].plot(time_axis,phi)
    ax3[1].set_ylabel("phi [rad]")
    ax3[1].set_xlabel("Time [s]")
    ax3[1].set_title("Phase Angle (phi)")
    ax3[1].grid(True)
    ax3[1].set_xlim(min(time_axis), max(time_axis))

    plt.tight_layout()
    plt.savefig(Figures_dir / "3_pf_phi.png")
    plt.close(fig3)

    # -------------------------------------------------
    # Figure 4: DC Link Voltage (V_dc)
    # -------------------------------------------------

    fig4, ax4 = plt.subplots(figsize=(6.4, 4.8))

    ax4.plot(time_axis,V_dc)
    ax4.set_ylabel("V_dc [V]")
    ax4.set_xlabel("Time [s]")
    ax4.set_title("DC Link Voltage (V_dc)")
    ax4.grid(True)
    ax4.set_xlim(min(time_axis), max(time_axis))

    plt.tight_layout()
    plt.savefig(Figures_dir / "4_Vdc.png")
    plt.close(fig4)

    # -------------------------------------------------
    # Figure 5: Ambient Temperature (T_env)
    # -------------------------------------------------

    fig5, ax5 = plt.subplots(figsize=(6.4, 4.8))

    ax5.plot(time_axis,T_env-273.15)
    ax5.set_ylabel("T_env [C]")
    ax5.set_xlabel("Time [s]")
    ax5.set_title("Ambient Temperature (T_env)")
    ax5.grid(True)
    ax5.set_xlim(min(time_axis), max(time_axis))

    plt.tight_layout()
    plt.savefig(Figures_dir / "5_T_env.png")
    plt.close(fig5)


def Plotting_electrical_loss(df_electrical_loss, Figures_dir):

    # -------------------------------------------------
    # Figure 13: IGBT & Diode Total Power Losses (P_I, P_D)
    # -------------------------------------------------

    fig13, ax = plt.subplots(figsize=(6.4, 4.8))

    print("P_I",np.sqrt(np.mean(df_electrical_loss["P_I"] ** 2)))
    print("P_D", np.sqrt(np.mean(df_electrical_loss["P_D"] ** 2)))

    ax.plot(df_electrical_loss["time"], df_electrical_loss["P_I"], label="P_I (IGBT)")
    ax.plot(df_electrical_loss["time"], df_electrical_loss["P_D"], label="P_D (Diode)")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Power Loss [W]")
    ax.set_title("Total Power Losses – IGBT & Diode")
    ax.legend(loc="best")
    ax.grid(True)
    ax.set_xlim(min(df_electrical_loss["time"]),max(df_electrical_loss["time"]))

    plt.tight_layout()
    plt.savefig(Figures_dir / "13_PowerLoss_P_I_P_D.png")
    plt.close(fig13)

    # -------------------------------------------------
    # Figure 14: IGBT & Diode Currents (is_I, is_D)
    # -------------------------------------------------

    fig14, ax = plt.subplots(figsize=(6.4, 4.8))

    print("is_I",np.sqrt(np.mean(df_electrical_loss["is_I"] ** 2)))
    print("is_D", np.sqrt(np.mean(df_electrical_loss["is_D"] ** 2)))

    ax.plot(df_electrical_loss["time"], df_electrical_loss["is_I"], label="is_I (IGBT)")
    ax.plot(df_electrical_loss["time"], df_electrical_loss["is_D"], label="is_D (Diode)")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Current [A]")
    ax.set_title("Device Currents – IGBT & Diode")
    ax.legend(loc="best")
    ax.grid(True)
    ax.set_xlim(min(df_electrical_loss["time"]), max(df_electrical_loss["time"]))

    plt.tight_layout()
    plt.savefig(Figures_dir / "14_Currents_is_I_is_D.png")
    plt.close(fig14)

    # -------------------------------------------------
    # Figure 15: IGBT & Diode Switching Losses (P_sw_I, P_sw_D)
    # -------------------------------------------------

    fig15, ax = plt.subplots(figsize=(6.4, 4.8))

    ax.plot(df_electrical_loss["time"], df_electrical_loss["P_sw_I"], label="P_sw_I (IGBT)")
    ax.plot(df_electrical_loss["time"], df_electrical_loss["P_sw_D"], label="P_sw_D (Diode)")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Switching Loss [W]")
    ax.set_title("Switching Losses – IGBT & Diode")
    ax.legend(loc="best")
    ax.grid(True)
    ax.set_xlim(min(df_electrical_loss["time"]), max(df_electrical_loss["time"]))

    plt.tight_layout()
    plt.savefig(Figures_dir / "15_SwitchingLosses_P_sw_I_P_sw_D.png")
    plt.close(fig15)

    # -------------------------------------------------
    # Figure 16: IGBT & Diode Conduction Losses (P_con_I, P_con_D)
    # -------------------------------------------------

    fig16, ax = plt.subplots(figsize=(6.4, 4.8))

    ax.plot(df_electrical_loss["time"], df_electrical_loss["P_con_I"], label="P_con_I (IGBT)")
    ax.plot(df_electrical_loss["time"], df_electrical_loss["P_con_D"], label="P_con_D (Diode)")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Conduction Loss [W]")
    ax.set_title("Conduction Losses – IGBT & Diode")
    ax.legend(loc="best")
    ax.grid(True)
    ax.set_xlim(min(df_electrical_loss["time"]), max(df_electrical_loss["time"]))

    plt.tight_layout()
    plt.savefig(Figures_dir / "16_ConductionLosses_P_con_I_P_con_D.png")
    plt.close(fig16)


def Plotting_thermal(df_thermal, Figures_dir):

    # -------------------------------------------------
    # Figure 17: Junction Temperatures (Tj_igbt, Tj_diode)
    # -------------------------------------------------


    print("Tj_igbt",np.sqrt(np.mean((df_thermal["Tj_igbt"]-273.15) ** 2)))
    print("Tj_diode", np.sqrt(np.mean((df_thermal["Tj_diode"]-273.15) ** 2)))

    fig17, ax = plt.subplots(figsize=(6.4, 4.8))

    ax.plot(df_thermal["time"], df_thermal["Tj_igbt"] - 273.15, label="Tj_IGBT [°C]")
    ax.plot(df_thermal["time"], df_thermal["Tj_diode"] - 273.15, label="Tj_Diode [°C]")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("Junction Temperatures – IGBT & Diode")
    ax.legend(loc="best")
    ax.grid(True)
    ax.set_xlim(min(df_thermal["time"]), max(df_thermal["time"]))

    plt.tight_layout()
    plt.savefig(Figures_dir / "17_JunctionTemperatures_IGBT_Diode.png")
    plt.close(fig17)

    # -------------------------------------------------
    # Figure 18: Case and Heat Sink Temperatures (T_case, T_sink)
    # -------------------------------------------------

    print("T_case",np.sqrt(np.mean((df_thermal["T_case"]-273.15) ** 2)))
    print("T_sink", np.sqrt(np.mean((df_thermal["T_sink"]-273.15) ** 2)))

    fig18, ax = plt.subplots(figsize=(6.4, 4.8))

    ax.plot(df_thermal["time"], df_thermal["T_case"] - 273.15, label="T_case [°C]")
    ax.plot(df_thermal["time"], df_thermal["T_sink"] - 273.15, label="T_sink [°C]")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("Case & Heat Sink Temperatures")
    ax.legend(loc="best")
    ax.grid(True)
    ax.set_xlim(min(df_thermal["time"]), max(df_thermal["time"]))

    plt.tight_layout()
    plt.savefig(Figures_dir / "18_Tcase_Tsink.png")
    plt.close(fig18)
