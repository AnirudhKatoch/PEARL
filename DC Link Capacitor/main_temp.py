import os
import sys
import numpy as np
import pandas as pd
from mother_function import mother_function

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(f"{CURRENT_DIR }/z/Apparent_power.parquet")
S_in = df["S_in"].to_numpy()
#S_in = 1.0191e6
#S_in = np.full(3600, S_in)
pf_in = 0
P_in = abs(S_in * pf_in)
Q_in = np.sqrt(S_in ** 2 - P_in ** 2)
if pf_in < 0:
    Q_in = Q_in * -1



temp_values = np.arange(0, 150, .1)
print(len(temp_values))
for i, temp in enumerate(temp_values, start=1):
    print(f"Temp_{temp}: Simulation_{i}")
    T_env_in = np.full(len(S_in), 273.15 + temp)
    mother_function(P_in, Q_in, T_env_in)
    print("##################################")
