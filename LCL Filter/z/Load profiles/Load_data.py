import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size":16, "font.family": "Times New Roman", "axes.labelsize":16, "axes.titlesize":16,"xtick.labelsize":16, "ytick.labelsize":16, "legend.fontsize": 15})


files = {
    "2_fulltime":        "Load/synPRO_el_2_fulltime_employees.dat",
    "persons_over65":    "Load/synPRO_el_2_persons_over65.dat",
    "family":            "Load/synPRO_el_family.dat",
    "single_under30":    "Load/synPRO_el_single_person_under30.dat",
}

# --- read each file, index by timestamp, keep only P_el ---
series = {}
for name, path in files.items():
    df = pd.read_csv(path, comment="#", sep=";")
    df.index = pd.to_datetime(df["unixtimestamp"], unit="s", utc=True)
    series[name] = df["P_el"]

# --- combine into one dataframe (aligned on the shared timestamp index) ---
P_all = pd.DataFrame(series)

P_total = P_all.sum(axis=1)        # 15-min resolution, 35040 values

P_daily_mean = P_total.resample("1D").mean()   # average power per day [W]
P_daily_mean = P_daily_mean[1:len(P_daily_mean)]

df = pd.DataFrame()
df["Power"] = P_daily_mean
#df = df.reset_index()
#del df['unixtimestamp']

multiplier = (50000/max(df["Power"]))
multiplier = 30

df["Power"] = df["Power"] * multiplier
df["Power"] = df["Power"].clip(upper=50000)
df["Power"] = df["Power"].round().astype(int)

df.to_parquet("E:/PEARL/LCL Filter/z/Load profiles/Load_profiles.parquet")

import matplotlib.pyplot as plt


def plot_load_profile(df, figures_dir=None, fname="Fig_load_profile"):
    """
    Plot the daily load (power) profile over the mission-profile period.

    df : DataFrame with a 'Power' column [W], indexed by timestamp
         (365 daily samples).
    """

    fig, ax = plt.subplots(figsize=(6.4, 4.8*0.5))

    # x-axis as day number (0..364) rather than raw timestamps
    days = range(len(df))
    ax.plot(days, df["Power"].to_numpy()/1000, color="red", linewidth=1.2)

    ax.set_xlabel("Time [Days]")
    ax.set_ylabel("Power [kW]")
    ax.set_xlim(0, len(df) - 1)
    #ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if figures_dir:
        fig.savefig(f"{figures_dir}/{fname}.pdf", dpi=300)
        plt.close(fig)
    else:
        plt.show()

    return fig

plot_load_profile(df, figures_dir="Figures")

