import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.lines as mlines
import matplotlib.font_manager as fm


# register the TUM font from the ttf file
tum_font_path = "fonts/TUMNeueHelvetica-Regular.ttf"   # adjust path to your ttf
fm.fontManager.addfont(tum_font_path)
tum_name = fm.FontProperties(fname=tum_font_path).get_name()

#plt.rcParams.update({"font.size": 17.5, "font.family": "Times New Roman", "axes.labelsize": 17.5, "axes.titlesize": 17.5, "xtick.labelsize": 17.5, "ytick.labelsize": 17.5, "legend.fontsize": 17.5})

plt.rcParams.update({"font.size": 17.5, "font.family": tum_name, "axes.labelsize": 17.5, "axes.titlesize": 17.5, "xtick.labelsize": 17.5, "ytick.labelsize": 17.5, "legend.fontsize": 17.5})


def plotting_electrical_loss(df_pf1, df_pf0):

    fig, ax = plt.subplots(figsize=(6.4, 4.8*0.75))

    t_pf1 = df_pf1["time"] - df_pf1["time"].iloc[0]
    t_pf0 = df_pf0["time"] - df_pf0["time"].iloc[0]

    # ---- Plot curves ----
    ax.plot(t_pf1, df_pf1["P_sw_I"], linestyle="-", color="b")
    ax.plot(t_pf1, df_pf1["P_sw_D"], linestyle="-", color="r")

    ax.plot(t_pf0, df_pf0["P_sw_I"], linestyle=":", color="b")
    ax.plot(t_pf0, df_pf0["P_sw_D"], linestyle=":", color="r")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Power losses [W]")
    ax.grid(True)

    # -------- Legend 1: Device (color meaning) --------
    igbt_line = mlines.Line2D([], [], color='b', linestyle='-', label='IGBT')
    diode_line = mlines.Line2D([], [], color='r', linestyle='-', label='Diode')

    legend1 = ax.legend(handles=[igbt_line, diode_line],
                        loc='upper left',
                        title="")
    ax.add_artist(legend1)

    # -------- Legend 2: Power Factor (linestyle meaning) --------
    pf1_line = mlines.Line2D([], [], color='black', linestyle='-', label='pf = 1')
    pf0_line = mlines.Line2D([], [], color='black', linestyle=':', label='pf = 0')

    ax.legend(handles=[pf1_line, pf0_line],
              loc='upper right',
              title="")

    ax.set_xlim(min(t_pf1),max(t_pf1))

    ax.set_ylim(0,65)
    ax.set_xticks(np.arange(0, 0.0401, 0.005))

    plt.tight_layout()
    plt.savefig("Paper_figures/IGBT and diode power losses.png")
    plt.close(fig)

#df_pf1 = pd.read_parquet("Simulation_results/Simulation_1_0.0002/df_electrical_loss/df_1.parquet",engine="pyarrow")
#df_pf0 = pd.read_parquet("Simulation_results/Simulation_2_0.0002/df_electrical_loss/df_1.parquet",engine="pyarrow")
#plotting_electrical_loss(df_pf1[:201], df_pf0[:201])

def plotting_instantaneous_current(df_pf1, df_pf0):

    fig, ax = plt.subplots(figsize=(6.4, 4.8*0.75))

    t_pf1 = df_pf1["time"] - df_pf1["time"].iloc[0]
    t_pf0 = df_pf0["time"] - df_pf0["time"].iloc[0]

    # ---- Plot curves ----
    ax.plot(t_pf1, df_pf1["is_I"], linestyle="-", color="b")
    ax.plot(t_pf1, df_pf1["is_D"], linestyle="-", color="r")

    ax.plot(t_pf0, df_pf0["is_I"], linestyle=":", color="b")
    ax.plot(t_pf0, df_pf0["is_D"], linestyle=":", color="r")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Current [A]")
    ax.grid(True)

    # -------- Legend 1: Device (color meaning) --------
    igbt_line = mlines.Line2D([], [], color='b', linestyle='-', label='IGBT')
    diode_line = mlines.Line2D([], [], color='r', linestyle='-', label='Diode')

    legend1 = ax.legend(handles=[igbt_line, diode_line],
                        loc='upper left',
                        title="")
    ax.add_artist(legend1)

    # -------- Legend 2: Power Factor (linestyle meaning) --------
    pf1_line = mlines.Line2D([], [], color='black', linestyle='-', label='pf = 1')
    pf0_line = mlines.Line2D([], [], color='black', linestyle=':', label='pf = 0')

    ax.legend(handles=[pf1_line, pf0_line],
              loc='upper right',
              title="")

    ax.set_xlim(min(t_pf1),max(t_pf1))

    ax.set_ylim(0,155)
    ax.set_xticks(np.arange(0, 0.0401, 0.005))

    plt.tight_layout()
    plt.savefig("Paper_figures/IGBT and diode instantaneous current.png")
    plt.close(fig)

#df_pf1 = pd.read_parquet("Simulation_results/Simulation_1_0.0002/df_electrical_loss/df_1.parquet",engine="pyarrow")
#df_pf0 = pd.read_parquet("Simulation_results/Simulation_2_0.0002/df_electrical_loss/df_1.parquet",engine="pyarrow")
#plotting_instantaneous_current(df_pf1[:201], df_pf0[:201])

def plotting_thermal(df_pf1, df_pf0):
    fig, ax = plt.subplots(figsize=(6.4, 4.8*0.75))

    # Time vectors (start at 0)
    t_pf1 = df_pf1["time"] - df_pf1["time"].iloc[0]
    t_pf0 = df_pf0["time"] - df_pf0["time"].iloc[0]

    # ---- Plot curves ----
    ax.plot(t_pf1, df_pf1["Tj_igbt"] - 273.15, linestyle="-", color="b")
    ax.plot(t_pf1, df_pf1["Tj_diode"] - 273.15, linestyle="-", color="r")
    ax.plot(t_pf0, df_pf0["Tj_igbt"] - 273.15, linestyle=":", color="b")
    ax.plot(t_pf0, df_pf0["Tj_diode"] - 273.15, linestyle=":", color="r")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.grid(True)

    # -------- Legend 1: Device (color meaning) --------
    igbt_line = mlines.Line2D([], [], color="b", linestyle="-", label="IGBT")
    diode_line = mlines.Line2D([], [], color="r", linestyle="-", label="Diode")
    legend1 = ax.legend(handles=[igbt_line, diode_line], loc="upper left", title="")
    ax.add_artist(legend1)

    # -------- Legend 2: Power Factor (linestyle meaning) --------
    pf1_line = mlines.Line2D([], [], color="black", linestyle="-", label="pf = 1")
    pf0_line = mlines.Line2D([], [], color="black", linestyle=":", label="pf = 0")
    ax.legend(handles=[pf1_line, pf0_line], loc="upper right", title="")

    ax.set_xlim(min(t_pf1), max(t_pf1))
    ax.set_ylim(25, 62.5)
    ax.set_xticks(np.arange(0, 0.0401, 0.005))

    plt.tight_layout()
    plt.savefig("Paper_figures/Short-time junction temperature of IGBT and diode.png")
    plt.close(fig)

#df_pf1 = pd.read_parquet("Simulation_results/Simulation_1_0.0002/df_thermal/df_1.parquet",engine="pyarrow")
#df_pf0 = pd.read_parquet("Simulation_results/Simulation_2_0.0002/df_thermal/df_1.parquet",engine="pyarrow")
#plotting_thermal(df_pf1[:201], df_pf0[:201])

def plotting_thermal_simple(df_pf1, df_pf0):
    fig, ax = plt.subplots(figsize=(6.4, 4.8*0.5))

    # Time vectors (start at 0)
    t_pf1 = df_pf1["time"] - df_pf1["time"].iloc[0]
    t_pf0 = df_pf0["time"] - df_pf0["time"].iloc[0]

    # ---- Plot curves ----
    ax.plot(t_pf1, df_pf1["Tj_igbt"] - 273.15, linestyle="-", color="tab:blue",linewidth=2.5*1.25)
    ax.plot(t_pf1, df_pf1["Tj_diode"] - 273.15, linestyle="-", color="tab:red",linewidth=2.5*1.25)
    #ax.plot(t_pf0, df_pf0["Tj_igbt"] - 273.15, linestyle=":", color="b")
    #ax.plot(t_pf0, df_pf0["Tj_diode"] - 273.15, linestyle=":", color="r")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.grid(True)

    # -------- Legend 1: Device (color meaning) --------
    igbt_line = mlines.Line2D([], [], color="tab:blue", linestyle="-", label="IGBT")
    diode_line = mlines.Line2D([], [], color="tab:red", linestyle="-", label="Diode")
    legend1 = ax.legend(handles=[igbt_line, diode_line], loc="lower right", title="")
    ax.add_artist(legend1)

    # -------- Legend 2: Power Factor (linestyle meaning) --------
    #pf1_line = mlines.Line2D([], [], color="black", linestyle="-", label="pf = 1")
    #pf0_line = mlines.Line2D([], [], color="black", linestyle=":", label="pf = 0")
    #ax.legend(handles=[pf1_line, pf0_line], loc="upper right", title="")

    ax.set_xlim(min(t_pf1), max(t_pf1))
    #ax.set_ylim(25, 57.5)
    #ax.set_xticks(np.arange(0, 0.0401, 0.005))

    plt.tight_layout()
    plt.savefig("Paper_figures/Long-duration junction temperature of IGBT and diode.pdf")
    plt.close(fig)

df_pf1 = pd.read_parquet("Simulation_results/Simulation_1_0.0002/df_thermal/df_1.parquet",engine="pyarrow")
df_pf0 = pd.read_parquet("Simulation_results/Simulation_2_0.0002/df_thermal/df_1.parquet",engine="pyarrow")
#plotting_thermal_simple(df_pf1, df_pf0)

def plotting_electro_thermal_chain_pf1_pf0(df_pf1_loss, df_pf1_thermal, df_pf0_loss, df_pf0_thermal, outpath="Paper_figures/Electro_thermal_chain_pf1_pf0.pdf", t_end=0.020):

    fig, axes = plt.subplots(3, 1, figsize=(6.4, 4.8 * 3 * 0.6675), sharex=False)
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 4.8 * 3 * 0.6675), sharex=False)

    # Time vectors (start at 0)
    t1 = df_pf1_loss["time"] - df_pf1_loss["time"].iloc[0]
    t0 = df_pf0_loss["time"] - df_pf0_loss["time"].iloc[0]

    # ============ 1) Instantaneous Current ============
    axes[0].plot(t1, df_pf1_loss["is_I"], linestyle="-", color="blue",linewidth=2.5*1.25)
    axes[0].plot(t1, df_pf1_loss["is_D"], linestyle="-", color="red",linewidth=2.5*1.25)
    axes[0].plot(t0, df_pf0_loss["is_I"], linestyle="--", color="blue",linewidth=2.5*1.25)
    axes[0].plot(t0, df_pf0_loss["is_D"], linestyle="--", color="red",linewidth=2.5*1.25)
    axes[0].set_ylabel("Current [A]")
    axes[0].set_xlabel("Time [s]")
    #axes[0].grid(True)
    #axes[0].set_title("(a) Instantaneous IGBT and diode currents")
    axes[0].set_title("Instantaneous IGBT and diode currents")
    axes[0].set_ylim(0, 140)  # adjust if you want tighter

    # ============ 2) Power Losses ============
    # If you want TOTAL losses, replace P_sw_* with (P_sw_* + P_con_*)
    axes[1].plot(t1, df_pf1_loss["P_sw_I"], linestyle="-", color="blue",linewidth=2.5*1.25)
    axes[1].plot(t1, df_pf1_loss["P_sw_D"], linestyle="-", color="red",linewidth=2.5*1.25)
    axes[1].plot(t0, df_pf0_loss["P_sw_I"], linestyle="--", color="blue",linewidth=2.5*1.25)
    axes[1].plot(t0, df_pf0_loss["P_sw_D"], linestyle="--", color="red",linewidth=2.5*1.25)
    axes[1].set_ylabel("Power loss [W]")
    axes[1].set_xlabel("Time [s]")
    #axes[1].grid(True)
    #axes[1].set_title("(b) Total IGBT and diode power losses")
    axes[1].set_title("Total IGBT and diode power losses")
    axes[1].set_ylim(0, 47.5)  # adjust if needed

    # ============ 3) Junction Temperature ============
    axes[2].plot(t1, df_pf1_thermal["Tj_igbt"] - 273.15, linestyle="-", color="blue",linewidth=2.5*1.25)
    axes[2].plot(t1, df_pf1_thermal["Tj_diode"] - 273.15, linestyle="-", color="red",linewidth=2.5*1.25)
    axes[2].plot(t0, df_pf0_thermal["Tj_igbt"] - 273.15, linestyle="--", color="blue",linewidth=2.5*1.25)
    axes[2].plot(t0, df_pf0_thermal["Tj_diode"] - 273.15, linestyle="--", color="red",linewidth=2.5*1.25)
    axes[2].set_ylabel("Temperature [°C]")
    axes[2].set_xlabel("Time [s]")
    #axes[2].set_title("(c) IGBT and diode junction temperature")
    axes[2].set_title("IGBT and diode junction temperature")
    #axes[2].grid(True)
    axes[2].set_ylim(25, 50)  # adjust if you want tighter

    # ============ Single Figure-Level Legend at Top ============

    # Device legend (color)
    igbt_line = mlines.Line2D([], [], color="blue", linestyle="-", label="IGBT",linewidth=2.5*1.25)
    diode_line = mlines.Line2D([], [], color="red", linestyle="-", label="Diode",linewidth=2.5*1.25)

    # Power factor legend (linestyle)
    pf1_line = mlines.Line2D([], [], color="black", linestyle="-", label="pf = 1",linewidth=2.5*1.25)
    pf0_line = mlines.Line2D([], [], color="black", linestyle="--", label="pf = 0",linewidth=2.5*1.25)

    # Combine all
    handles = [igbt_line, diode_line, pf1_line, pf0_line]

    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=True, columnspacing=1.25, handlelength=1.5, bbox_to_anchor=(0.5, 1.0))

    plt.tight_layout(rect=[0, 0, 1.0, 0.95])

    # ============ Axis formatting ============
    # Use common x-limits based on requested window
    axes[2].set_xlim(0, t_end)
    # ticks every 5 ms
    axes[2].set_xticks(np.arange(0, t_end + 1e-12, 0.005))

    plt.savefig(outpath)
    plt.close(fig)

df_pf1_loss = pd.read_parquet("Simulation_results/Simulation_1_0.0002/df_electrical_loss/df_1.parquet",engine="pyarrow")
df_pf1_thermal = pd.read_parquet("Simulation_results/Simulation_1_0.0002/df_thermal/df_1.parquet",engine="pyarrow")
df_pf0_loss = pd.read_parquet("Simulation_results/Simulation_2_0.0002/df_electrical_loss/df_1.parquet",engine="pyarrow")
df_pf0_thermal = pd.read_parquet("Simulation_results/Simulation_2_0.0002/df_thermal/df_1.parquet",engine="pyarrow")

N = 101  # 0.201 s at dt=0.001 (if time is per ms); adjust as needed
plotting_electro_thermal_chain_pf1_pf0(df_pf1_loss[:N], df_pf1_thermal[:N],df_pf0_loss[:N], df_pf0_thermal[:N],outpath="Paper_figures/Electro_thermal_chain_pf1_pf0.pdf",t_end=0.020)  # set 0.040 to show exactly 2 cycles at 50 Hz)