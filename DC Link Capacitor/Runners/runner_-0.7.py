import os
import sys
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mother_function import mother_function

df = pd.read_parquet(f"{PROJECT_ROOT }/z/Apparent_power.parquet")
S_in = df["S_in"].to_numpy()

pf_in = -0.7
P_in = abs(S_in * pf_in)
Q_in = np.sqrt(S_in ** 2 - P_in ** 2)
if pf_in < 0:
    Q_in = Q_in * -1

T_env_in = np.full(len(S_in), 273.15 + 35)

mother_function(P_in, Q_in, T_env_in)
