import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 15, "font.family": "Times New Roman", "axes.labelsize": 15, "axes.titlesize": 15,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15})



def Plotting(df,Figures_dir):

    time_axis = np.arange(len(df["P"]))

    # -------------------------------------------------
    # Figure 1: Apparent, Active, Reactive Power (S, P, Q)
    # -------------------------------------------------

    fig1, ax1 = plt.subplots(3, 1, figsize=(6.4, 6), sharex=True)

    ax1[0].plot(time_axis, df["S"])
    ax1[0].set_ylabel("S [VA]")
    ax1[0].set_title("Apparent Power")
    ax1[0].grid(True)
    ax1[0].set_xlim(min(time_axis), max(time_axis))

    ax1[1].plot(time_axis, df["P"])
    ax1[1].set_ylabel("P [W]")
    ax1[1].set_title("Active Power")
    ax1[1].grid(True)
    ax1[1].set_xlim(min(time_axis), max(time_axis))

    ax1[2].plot(time_axis, df["Q"])
    ax1[2].set_ylabel("Q [VAR]")
    ax1[2].set_xlabel("Time [s]")
    ax1[2].set_title("Reactive Power")
    ax1[2].grid(True)
    ax1[2].set_xlim(min(time_axis), max(time_axis))

    plt.tight_layout()
    plt.savefig(Figures_dir / "1_power.png")
    plt.close(fig1)

    # -------------------------------------------------
    # Figure 2: Voltage and Current (Vs, Is)
    # -------------------------------------------------

    fig2, ax2 = plt.subplots(2, 1, figsize=(6.4, 6), sharex=True)

    ax2[0].plot(time_axis, df["Vs"])
    ax2[0].set_ylabel("Voltage [V]")
    ax2[0].set_title("Inverter AC side RMS voltage")
    ax2[0].grid(True)
    ax2[0].set_xlim(min(time_axis), max(time_axis))

    ax2[1].plot(time_axis, df["Is"])
    ax2[1].set_ylabel("Current [A]")
    ax2[1].set_xlabel("Time [s]")
    ax2[1].set_title("Inverter AC side RMS current")
    ax2[1].grid(True)
    ax2[1].set_xlim(min(time_axis), max(time_axis))

    plt.tight_layout()
    plt.savefig(Figures_dir / "2_inverter_voltage_and_current.png")
    plt.close(fig2)

    # -------------------------------------------------
    # Figure 3: Power Factor and Phase Angle (pf, phi)
    # -------------------------------------------------

    fig3, ax3 = plt.subplots(2, 1, figsize=(6.4, 6), sharex=True)

    ax3[0].plot(time_axis, df["pf"])
    ax3[0].set_ylabel("pf [-]")
    ax3[0].set_title("Power Factor")
    ax3[0].grid(True)
    ax3[0].set_xlim(min(time_axis), max(time_axis))

    ax3[1].plot(time_axis, df["phi"])
    ax3[1].set_ylabel("phi [rad]")
    ax3[1].set_xlabel("Time [s]")
    ax3[1].set_title("Phase Angle")
    ax3[1].grid(True)
    ax3[1].set_xlim(min(time_axis), max(time_axis))

    plt.tight_layout()
    plt.savefig(Figures_dir / "3_power_factor_and_phase_angle.png")
    plt.close(fig3)


    # -------------------------------------------------
    # Figure 4: DC Link Voltage (V_dc)
    # -------------------------------------------------

    fig4, ax4 = plt.subplots(figsize=(6.4, 4.8))

    ax4.plot(time_axis,df["V_dc"])
    ax4.set_ylabel("Voltage [V]")
    ax4.set_xlabel("Time [s]")
    ax4.set_title("DC Link Voltage")
    ax4.grid(True)
    ax4.set_xlim(min(time_axis), max(time_axis))

    plt.tight_layout()
    plt.savefig(Figures_dir / "4_DC_side_voltage.png")
    plt.close(fig4)

    # -------------------------------------------------
    # Figure 5: Ambient Temperature (T_env)
    # -------------------------------------------------

    fig5, ax5 = plt.subplots(figsize=(6.4, 4.8))

    ax5.plot(time_axis,df["T_env"]-273.15)
    ax5.set_ylabel("Temperature [°C]")
    ax5.set_xlabel("Time [s]")
    ax5.set_title("Ambient Temperature")
    ax5.grid(True)
    ax5.set_xlim(min(time_axis), max(time_axis))

    plt.tight_layout()
    plt.savefig(Figures_dir / "5_Ambient_temperature.png")
    plt.close(fig5)

    # -------------------------------------------------
    # Figure 6: Capacitors in series and parallel
    # -------------------------------------------------

    fig6, ax_left = plt.subplots(figsize=(6.4, 4.8))

    # Bar positions
    x_left = 0  # Series
    x_right = 1  # Parallel

    # Plot IGBT bar on left axis
    ax_left.bar(x_left, df["N_series"][0], width=0.5, color="C0", label="Series")

    # Create right axis
    ax_right = ax_left.twinx()

    # Plot Diode bar on right axis
    ax_right.bar(x_right, df["N_parallel"][0], width=0.5, color="C1", label="Parallel")

    # Set axis labels
    ax_left.set_ylabel("Series")
    ax_right.set_ylabel("Parallel")

    # Set title
    ax_left.set_title("Number of capacitors in series and parallel")

    # Set x-ticks and labels
    ax_left.set_xticks([x_left, x_right])
    ax_left.set_xticklabels(["Series", "Parallel"])

    # Grid only for left axis
    ax_left.grid(axis="y")

    plt.tight_layout()
    plt.savefig(Figures_dir / "6_Number_of_capacitors_in_series_and_parallel.png")
    plt.close(fig6)

    # -------------------------------------------------
    # Figure 7:  DC-link capacitor RMS ripple current
    # -------------------------------------------------

    fig7, ax7 = plt.subplots(figsize=(6.4, 4.8))

    ax7.plot(time_axis,df["Idcl"])
    ax7.set_ylabel("RMS Current [A]")
    ax7.set_xlabel("Time [s]")
    ax7.set_title("DC-link capacitor RMS ripple current")
    ax7.grid(True)
    ax7.set_xlim(min(time_axis), max(time_axis))

    plt.tight_layout()
    plt.savefig(Figures_dir / "7_DC_link_capacitor_RMS_ripple_current.png")
    plt.close(fig7)

    # -------------------------------------------------
    # Figure 8: Voltage and Current per capacitor
    # -------------------------------------------------

    fig8, ax8 = plt.subplots(2, 1, figsize=(6.4, 6), sharex=True)

    ax8[0].plot(time_axis, df["V_per_cap"])
    ax8[0].set_ylabel("Voltage [V]")
    ax8[0].set_title("Voltage per capacitor")
    ax8[0].grid(True)
    ax8[0].set_xlim(min(time_axis), max(time_axis))

    ax8[1].plot(time_axis, df["I_per_cap"])
    ax8[1].set_ylabel("Current [A]")
    ax8[1].set_xlabel("Time [s]")
    ax8[1].set_title("Current per capacitor")
    ax8[1].grid(True)
    ax8[1].set_xlim(min(time_axis), max(time_axis))

    plt.tight_layout()
    plt.savefig(Figures_dir / "8_Voltage_and_Current_per_capacitor.png")
    plt.close(fig8)

    # -------------------------------------------------
    # Figure 9: Ambient Temperature (T_env)
    # -------------------------------------------------

    fig9, ax9 = plt.subplots(figsize=(6.4, 4.8))

    ax9.plot(time_axis, df["T_core"] - 273.15)
    ax9.set_ylabel("Temperature [°C]")
    ax9.set_xlabel("Time [s]")
    ax9.set_title("Capacitor core temperature")
    ax9.grid(True)
    ax9.set_xlim(min(time_axis), max(time_axis))

    plt.tight_layout()
    plt.savefig(Figures_dir / "9_Capacitor_core_temperature.png")
    plt.close(fig9)

    # -------------------------------------------------
    # Figure 10: Lifetime of capacitor
    # -------------------------------------------------

    fig10, ax = plt.subplots(figsize=(6.4, 4.8))

    # Simple single bar
    ax.bar(0, df["L_tot"][0], width=0.5, color="C2", label="Capacitor")

    # Labels and title
    ax.set_ylabel("Capacitor lifetime [years]")
    ax.set_title("Capacitor lifetime")

    # X-axis label
    ax.set_xticks([0])
    ax.set_xticklabels(["Capacitor"])

    # Grid for readability
    ax.grid(axis="y")

    plt.tight_layout()
    plt.savefig(Figures_dir / "10_Lifetime_capacitor.png")
    plt.close(fig10)




