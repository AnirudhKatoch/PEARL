from Input_parameters import Input_parameters_class
import numpy as np
from Calculation_functions import Calculation_functions_class
import matplotlib.pyplot as plt
from Electrical_model import compute_IGBT_and_Diode_power_losses
import time

start_time = time.time()

params = Input_parameters_class()     # create instance
locals().update(params.__dict__)      # inject variables



def compute_IGBT_and_Diode_power_losses(Is, phi, V_dc, pf, dt, M, omega, t_on, t_off, f_sw, I_ref, V_ref, Err_D, R_IGBT, V_0_IGBT, R_D, V_0_D):

    #t = np.arange(0.0, len(Is), dt, dtype=np.float64)

    #Is = np.repeat(Is, int(1 / dt))
    #phi = np.repeat(phi, int(1 / dt))
    #V_dc = np.repeat(V_dc, int(1 / dt))
    #pf = np.repeat(pf, int(1 / dt))

    t = np.arange(0.0, 1, dt, dtype=np.float64)

    m = Calculation_functions_class.Instantaneous_modulation(M=M, omega=omega, t=t, phi=phi)
    is_I, is_D = Calculation_functions_class.IGBT_and_diode_current(Is=Is, t=t, m=m, omega=omega)
    P_sw_I, P_sw_D = Calculation_functions_class.Switching_losses(V_dc=V_dc, is_I=is_I, t_on=t_on, t_off=t_off,f_sw=f_sw, is_D=is_D, I_ref=I_ref, V_ref=V_ref,Err_D=Err_D)
    P_con_I, P_con_D = Calculation_functions_class.Conduction_losses(is_I=is_I, R_IGBT=R_IGBT, V_0_IGBT=V_0_IGBT, M=M,pf=pf, is_D=is_D, R_D=R_D, V_0_D=V_0_D)
    P_I = np.ascontiguousarray(np.maximum(P_sw_I + P_con_I, 0.0), dtype=np.float64)
    P_D = np.ascontiguousarray(np.maximum(P_sw_D + P_con_D, 0.0), dtype=np.float64)

    return P_I, P_D, is_I, is_D, P_sw_I, P_sw_D, P_con_I, P_con_D

for i,(Is, phi, V_dc, pf) in enumerate(zip(Is, phi, V_dc, pf)):

    P_I, P_D, is_I, is_D, P_sw_I, P_sw_D, P_con_I, P_con_D = compute_IGBT_and_Diode_power_losses(Is,
                                                                                                 phi,
                                                                                                 V_dc,
                                                                                                 pf,
                                                                                                 dt,
                                                                                                 M,
                                                                                                 omega,
                                                                                                 t_on,
                                                                                                 t_off,
                                                                                                 f_sw,
                                                                                                 I_ref,
                                                                                                 V_ref,
                                                                                                 Err_D,
                                                                                                 R_IGBT,
                                                                                                 V_0_IGBT,
                                                                                                 R_D,
                                                                                                 V_0_D)



