import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({"font.size": 15, "font.family": "Times New Roman", "axes.labelsize": 15, "axes.titlesize": 15,"xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15})

def load_benchmark(path):
    """Read a benchmark_values.txt holding repr() of a dict."""
    with open(path) as fh:
        return ast.literal_eval(fh.read())


def _pct_err(py, pl):
    if py is None or pl is None or pl == 0:
        return np.nan
    return abs((py - pl)) / pl * 100.0


def load_waveform(path):
    """Load an IL2_waveform.csv, normalize columns to time / I_L2,
    and collapse duplicate timestamps (PLECS emits repeated event points)."""
    df = pd.read_csv(path)
    df = df.rename(columns={"Time / s": "time",
                            "IL_2:Measured current": "I_L2",
                            "I_L2": "I_L2"})
    # keep only the two columns we need, by position if names still differ
    if "time" not in df.columns or "I_L2" not in df.columns:
        df = df.iloc[:, :2]
        df.columns = ["time", "I_L2"]
    df = df.groupby("time", as_index=False)["I_L2"].mean()
    return df


def compute_nrmse(python_root, plecs_root, scenario, normalize_by="range"):
    """NRMSE [%] of Python I_L2 vs PLECS I_L2 (PLECS = reference)."""
    py_path = os.path.join(python_root, scenario, "IL2_waveform.csv")
    pl_path = os.path.join(plecs_root,  scenario, "IL2_waveform.csv")
    if not (os.path.exists(py_path) and os.path.exists(pl_path)):
        return np.nan

    df_py = load_waveform(py_path)
    df_pl = load_waveform(pl_path)

    # resample PLECS onto Python's time grid
    t = df_py["time"].to_numpy()
    ref = np.interp(t, df_pl["time"].to_numpy(), df_pl["I_L2"].to_numpy())  # PLECS
    mod = df_py["I_L2"].to_numpy()                                          # Python

    rmse = np.sqrt(np.mean((mod - ref) ** 2))

    if normalize_by == "range":
        denom = ref.max() - ref.min()
    elif normalize_by == "rms":
        denom = np.sqrt(np.mean(ref ** 2))
    else:
        raise ValueError("normalize_by must be 'range' or 'rms'")

    if denom == 0:
        return np.nan
    return rmse / denom * 100.0


def compare_python_vs_plecs(python_root, plecs_root, scenarios=None,nrmse_normalize_by="range"):
    """
    % error of Python vs PLECS (PLECS = ground truth).
    Both roots contain identically-named scenario sub-folders,
    each with a benchmark_values.txt and an IL2_waveform.csv.
    """
    error_metrics = ["V_L1_RMS", "I_L1_RMS", "I_C_RMS", "V_C_RMS",
                     "V_L2_RMS", "I_L2_RMS", "PF", "THD_percent_I_L2"]

    if scenarios is None:
        scenarios = ["100_1", "75_1", "50_1", "25_1",
                     "100_0.866025_lags", "100_0.5_lags",
                     "100_0.866025_leads", "100_0.5_leads"]

    python = {s: load_benchmark(os.path.join(python_root, s, "benchmark_values.txt"))
              for s in scenarios}
    plecs  = {s: load_benchmark(os.path.join(plecs_root,  s, "benchmark_values.txt"))
              for s in scenarios}

    # PLECS stores THD as a fraction; Python stores it as percent → align units
    for s in scenarios:
        if plecs[s].get("THD_percent_I_L2") is not None:
            plecs[s]["THD_percent_I_L2"] *= 100.0

    rows = {}
    for m in error_metrics:
        rows[m] = [_pct_err(python[s].get(m), plecs[s].get(m)) for s in scenarios]

    # --- NRMSE row from the waveforms ---
    rows["NRMSE_I_L2"] = [compute_nrmse(python_root, plecs_root, s,
                                        normalize_by=nrmse_normalize_by)
                          for s in scenarios]

    df = pd.DataFrame(rows, index=scenarios).T

    metric_labels = {
        "V_L1_RMS": "V_L1 RMS [% err]",
        "I_L1_RMS": "I_L1 RMS [% err]",
        "V_L2_RMS": "V_L2 RMS [% err]",
        "I_L2_RMS": "I_L2 RMS [% err]",
        "I_C_RMS":  "I_C RMS [% err]",
        "V_C_RMS":  "V_C RMS [% err]",
        "PF":       "pf [% err]",
        "THD_percent_I_L2": "THD [% err]",
        "NRMSE_I_L2": f"I_L2 [NRMSE]"}

    scenario_labels = {
        "100_1": "1.00 MVA, pf=1.0",
        "75_1":  "0.75 MVA, pf=1.0",
        "50_1":  "0.50 MVA, pf=1.0",
        "25_1":  "0.25 MVA, pf=1.0",
        "100_0.866025_lags":  "1.00 MVA, pf=0.866 lag",
        "100_0.5_lags":       "1.00 MVA, pf=0.5 lag",
        "100_0.866025_leads": "1.00 MVA, pf=0.866 lead",
        "100_0.5_leads":      "1.00 MVA, pf=0.5 lead"}

    df = df.rename(index=metric_labels, columns=scenario_labels)

    pd.set_option("display.float_format", lambda v: f"{v:.2f}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(df.to_string())
    return df

python_root = r"E:\PEARL\LCL Filter\Benchmarking\Results\Python"
plecs_root  = r"E:\PEARL\LCL Filter\Benchmarking\Results\PLECS"

df = compare_python_vs_plecs(python_root, plecs_root)
df.round(2).to_csv(r"E:\PEARL\LCL Filter\Benchmarking\Results\comparison_python_vs_plecs.csv", sep=";")




def plot_IL2_comparison(python_root, plecs_root, scenario, save_path=None, n_cycles=1, f=50, resolution_per_cycle=5000):
    """Overlay PLECS and Python I_L2 for one scenario to show their agreement."""
    py_path = os.path.join(python_root, scenario, "IL2_waveform.csv")
    pl_path = os.path.join(plecs_root,  scenario, "IL2_waveform.csv")

    df_py = load_waveform(py_path)
    df_pl = load_waveform(pl_path)

    t = df_py["time"].to_numpy()
    py = df_py["I_L2"].to_numpy()
    pl = np.interp(t, df_pl["time"].to_numpy(), df_pl["I_L2"].to_numpy())

    n = int(n_cycles * resolution_per_cycle)
    t, py, pl = t[-n:], py[-n:], pl[-n:]
    t = t - t[0]

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    # PLECS as a solid line
    ax.plot(t, pl, color="tab:blue", label="PLECS", linewidth=2.0)

    # Python as sparse markers on top — every Nth sample
    step = max(1, n // 50)   # ~40 markers across the cycle
    ax.plot(t[::step], py[::step], color="tab:orange", label="Python",
            linestyle="none", marker="o", markersize=5,
            markerfacecolor="none", markeredgewidth=1.4)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Current [A]")
    ax.set_title("Grid-side current.")
    ax.set_xlim(t[0], t[-1] + (t[1] - t[0]))
    ax.xaxis.set_major_locator(MultipleLocator(0.005))   # ticks at 0.000, 0.005, ...
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


plot_IL2_comparison(python_root, plecs_root, "100_1",save_path=r"E:\PEARL\LCL Filter\Benchmarking\Results\IL2_comparison.pdf")