"""Shared helpers for the example studies. Every study writes to examples/results/<study>/:
figures (.png), tables (.csv) and summary.md (what it shows, in plain words, with the numbers)."""
import sys, os, warnings, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from main import synthetic_pv_profile
from mother_function import mother_function

RESULTS = Path(__file__).resolve().parent / "results"
S_RATED = 48790.0
C_I, C_D = "tab:blue", "tab:red"


def outdir(name):
    d = RESULTS / name; d.mkdir(parents=True, exist_ok=True); return d


def save_fig(fig, d, fname):
    fig.tight_layout(); fig.savefig(d / fname, dpi=120); plt.close(fig)


def table(rows, d, fname):
    df = pd.DataFrame(rows); df.to_csv(d / fname, index=False); return df


def md_table(df, fmt="{:.4g}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


def summary(d, title, what, tables=(), notes=()):
    text = [f"# {title}", "", what, ""]
    for cap, df in tables:
        text += [f"**{cap}**", "", md_table(df), ""]
    text += list(notes)
    (d / "summary.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))


def run(P, Q, T, **kw):
    for k, v in dict(save=False, plot=False, verbose=False, N_parallel=2, MC_samples=1000).items():
        kw.setdefault(k, v)
    return mother_function(P, Q, T, **kw)


def constant(P, Q, seconds=3600, T_amb_C=25.0):
    n = seconds
    return np.full(n, float(P)), np.full(n, float(Q)), np.full(n, 273.15 + T_amb_C)
