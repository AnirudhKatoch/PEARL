import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mother_function import mother_function
import numpy as np

S_in = 50000
pf_in = 1

P_in = abs(S_in*pf_in)
Q_in = np.sqrt(S_in**2 - P_in**2)

if pf_in<0:
    Q_in = Q_in*-1

Profile_size = 3600
P = np.full(Profile_size,P_in)
Q = np.full(Profile_size, Q_in)
T_env = np.full(Profile_size, 298.15)

r_sink_dic = {  1:2.20,
                2:1.25,
                3:0.90,
                4:0.72}

r_sink_input = r_sink_dic[4]
Weight_sink_input = 18/1000

mother_function(P=P,Q=Q,T_env=T_env,r_sink_input=r_sink_input,Weight_sink_input=Weight_sink_input)