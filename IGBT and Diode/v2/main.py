"""
main.py  (PEARL v3) - run PEARL on a mission profile.

1. Your own data: set PROFILE_CSV to a CSV with one row per second and the columns
       P_W   (active power, W, + = delivered to the grid)
       Q_var (reactive power, var, + = delivered to the grid)
       T_env_C (ambient temperature, degC)
   An example file with this format is data/example_profile_1day.csv.
2. No data: leave PROFILE_CSV = None and a synthetic 90-day PV profile is used.

Settings: any attribute of Input_parameters_class can be passed to mother_function, see SETTINGS.
Results: Results/Simulation_N/{Data, Figures}.
"""
from pathlib import Path
import numpy as np
from mother_function import mother_function

PROFILE_CSV = None            # e.g. Path(__file__).parent / "data" / "example_profile_1day.csv"
SYNTHETIC_DAYS = 90

SETTINGS = dict(
    N_parallel=2,                         # devices in parallel per switch (1 overloads the 50 A part at 48.8 kVA)
    lifetime_model="semikron2024",        # or a list, e.g. ["semikron2024", "lesit", {...your curves...}]
    # r_sink=0.72, sink_mass=0.018,       # another heat sink (10U at 4 m/s)
    # kappa_IGBT=0.003, kappa_Diode=0.003,  # losses rising with temperature
    # profile_dt=60.0,                    # if your data has one row per minute
)


def load_profile_csv(path):
    import pandas as pd
    df = pd.read_csv(path)
    return df["P_W"].to_numpy(float), df["Q_var"].to_numpy(float), df["T_env_C"].to_numpy(float) + 273.15


def synthetic_pv_profile(days=90, seed=7):
    """Clear / broken-cloud / overcast days, reactive support, daily ambient cycle (stand-in for measured data)."""
    rng = np.random.default_rng(seed)
    S_rated = 48790.0
    n = days * 86400; t = np.arange(n); sod = t % 86400; day = t // 86400
    x = (sod - 5.5 * 3600) / (15 * 3600)
    clear = np.where((x > 0) & (x < 1), np.sin(np.pi * np.clip(x, 0, 1)) ** 1.3, 0.0)
    cloud = np.ones(n)
    for d in range(days):
        regime = rng.choice(["clear", "broken", "overcast"], p=[0.35, 0.45, 0.20]); s0 = d * 86400
        if regime == "overcast":
            cloud[s0:s0 + 86400] = rng.uniform(0.15, 0.35) + 0.05 * np.sin(np.arange(86400) / 900)
        elif regime == "broken":
            k = 0
            while k < 86400:
                k += int(rng.exponential(150)) + 5; L = int(rng.exponential(45)) + 5
                cloud[s0 + k:s0 + min(86400, k + L)] = rng.uniform(0.2, 0.5); k += L
    cloud = np.convolve(cloud, np.ones(5) / 5, mode="same")
    P = S_rated * 0.85 * clear * cloud
    Q = -0.33 * P + 0.05 * S_rated * np.sin(2 * np.pi * t / 5400) * (P > 0)
    T_env = 273.15 + 18 + 7 * np.sin(2 * np.pi * (sod / 86400 - 0.375)) + 4 * np.sin(2 * np.pi * day / 23)
    return P, Q, T_env


# kept for scripts written for PEARL v2
load_mission_profile = synthetic_pv_profile


if __name__ == "__main__":
    if PROFILE_CSV is not None:
        P, Q, T_env = load_profile_csv(PROFILE_CSV)
    else:
        P, Q, T_env = synthetic_pv_profile(SYNTHETIC_DAYS)
    results = mother_function(P, Q, T_env, **SETTINGS)
    print("results written to", results["folders"]["simulation"])
