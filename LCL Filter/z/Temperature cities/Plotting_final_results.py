import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from scipy.stats import weibull_min

plt.rcParams.update({"font.size":16, "font.family": "Times New Roman", "axes.labelsize":16, "axes.titlesize":16,"xtick.labelsize":16, "ytick.labelsize":16, "legend.fontsize": 15})

# directory holding the .nc files
base_dir = r"E:\PEARL\LCL Filter\z\Temperature cities\FinaL_Results"

Bahir_Dar     = pd.read_parquet(f"{base_dir}/Bahir_Dar/Dataframes/df_6_MC.parquet")
Bandung       = pd.read_parquet(f"{base_dir}/Bandung/Dataframes/df_6_MC.parquet")
Juja          = pd.read_parquet(f"{base_dir}/Juja/Dataframes/df_6_MC.parquet")
Kampala       = pd.read_parquet(f"{base_dir}/Kampala/Dataframes/df_6_MC.parquet")
Koudougou     = pd.read_parquet(f"{base_dir}/Koudougou/Dataframes/df_6_MC.parquet")
Kumasi        = pd.read_parquet(f"{base_dir}/Kumasi/Dataframes/df_6_MC.parquet")
Lima          = pd.read_parquet(f"{base_dir}/Lima/Dataframes/df_6_MC.parquet")
Mumbai        = pd.read_parquet(f"{base_dir}/Mumbai/Dataframes/df_6_MC.parquet")
Munich        = pd.read_parquet(f"{base_dir}/Munich/Dataframes/df_6_MC.parquet")
Stellenbosch  = pd.read_parquet(f"{base_dir}/Stellenbosch/Dataframes/df_6_MC.parquet")
Windhoek      = pd.read_parquet(f"{base_dir}/Windhoek/Dataframes/df_6_MC.parquet")

Lifetime_LCL_MC = {"Bahir_Dar"   : Bahir_Dar["Lifetime_LCL_MC"].to_numpy(),
                   "Bandung"     : Bandung["Lifetime_LCL_MC"].to_numpy(),
                   "Juja"        : Juja["Lifetime_LCL_MC"].to_numpy(),
                   "Kampala"     : Kampala["Lifetime_LCL_MC"].to_numpy(),
                   "Koudougou"   : Koudougou["Lifetime_LCL_MC"].to_numpy(),
                   "Kumasi"      : Kumasi["Lifetime_LCL_MC"].to_numpy(),
                   "Lima"        : Lima["Lifetime_LCL_MC"].to_numpy(),
                   "Mumbai"      : Mumbai["Lifetime_LCL_MC"].to_numpy(),
                   "Munich"      : Munich["Lifetime_LCL_MC"].to_numpy(),
                   "Stellenbosch": Stellenbosch["Lifetime_LCL_MC"].to_numpy(),
                   "Windhoek"    : Windhoek["Lifetime_LCL_MC"].to_numpy(),}



def plot_lifetime_MC_cities_weibull(Lifetime_LCL_MC, figures_dir=None,
                                    fname="Fig_lifetime_MC_cities",
                                    n_points=500):
    fig, ax = plt.subplots(figsize=(6.4, 4.8 ))

    all_data = np.concatenate([
        np.asarray(v, dtype=float)[np.isfinite(v)]
        for v in Lifetime_LCL_MC.values()])

    x = np.linspace(0.0, all_data.max(), n_points)

    # rank cities by median lifetime: shortest -> longest
    central = {city: np.median(np.asarray(v, dtype=float)[np.isfinite(v)])
               for city, v in Lifetime_LCL_MC.items()}
    cities_sorted = sorted(central, key=central.get)
    n = len(cities_sorted)

    cmap = plt.get_cmap("RdBu")
    color_for = {city: cmap(rank / (n - 1))
                 for rank, city in enumerate(cities_sorted)}

    for city in reversed(cities_sorted):
        data = np.asarray(Lifetime_LCL_MC[city], dtype=float)
        data = data[np.isfinite(data)]
        if data.size < 2 or np.allclose(data, data[0]):
            continue

        # fit 3-parameter Weibull (location floats)
        beta, loc, eta = weibull_min.fit(data)
        y = weibull_min.pdf(x, beta, loc=loc, scale=eta)

        label = city.replace("_", " ")
        ax.plot(x, y, linewidth=1.8, color=color_for[city], label=label)

    ax.set_xlabel("Lifetime [years]")
    ax.set_ylabel("Probability density [1/years]")
    ax.set_xlim(0.0, 60)
    ax.set_ylim(0, 0.0725)

    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.44, 1.0),
              borderaxespad=0.25, handlelength=1.25,
              columnspacing=1.0, labelspacing=0.25)
    fig.tight_layout()

    if figures_dir:
        fig.savefig(f"{figures_dir}/{fname}.png", dpi=300)
        plt.close(fig)
    else:
        plt.show()
    return fig

def plot_lifetime_MC_cities_weibull_B_10(Lifetime_LCL_MC, figures_dir=None, fname="Fig_lifetime_MC_cities", n_points=500):

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    all_data = np.concatenate([
        np.asarray(v, dtype=float)[np.isfinite(v)]
        for v in Lifetime_LCL_MC.values()])

    x = np.linspace(0.0, 33.50, n_points)  # fixed to xlim range

    # rank cities by median lifetime: shortest -> longest
    central = {city: np.median(np.asarray(v, dtype=float)[np.isfinite(v)])
               for city, v in Lifetime_LCL_MC.items()}
    cities_sorted = sorted(central, key=central.get)
    n = len(cities_sorted)

    # Use a perceptually distinct colormap for many cities
    cmap = plt.get_cmap("RdBu")
    color_for = {city: cmap(rank / (n - 1))
                 for rank, city in enumerate(cities_sorted)}

    B10_values = {}

    for city in reversed(cities_sorted):
        data = np.asarray(Lifetime_LCL_MC[city], dtype=float)
        data = data[np.isfinite(data)]
        if data.size < 2 or np.allclose(data, data[0]):
            continue

        beta, loc, eta = weibull_min.fit(data)
        F = weibull_min.cdf(x, beta, loc=loc, scale=eta)
        B10_values[city] = weibull_min.ppf(0.10, beta, loc=loc, scale=eta)

        label = city.replace("_", " ")
        ax.plot(x, F, linewidth=2.25, color=color_for[city], label=label)

    # B10 reference line — label anchored to xlim, not a hardcoded x
    ax.axhline(0.10, linestyle="--", color="k", linewidth=1.0)
    ax.text(ax.get_xlim()[1] * 0.98, 0.1015, "", ha="right", va="bottom", fontsize=15)

    ax.set_xlabel("Lifetime [years]")
    ax.set_ylabel("Unreliability [-]")
    ax.set_xlim(0.0, 10.5)
    ax.set_ylim(0, 0.11)

    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.44, 1.0), borderaxespad=0.25, handlelength=1.25, columnspacing=1.0, labelspacing=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if figures_dir:
        fig.savefig(f"{figures_dir}/{fname}_B10.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    print("B10 lifetimes [years]:")
    for city in cities_sorted:
        if city in B10_values:
            print(f"  {city.replace('_', ' '):<14} {B10_values[city]:.2f}")

    return fig

plot_lifetime_MC_cities_weibull_B_10(Lifetime_LCL_MC, figures_dir="Figures")