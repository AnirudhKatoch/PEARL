import numpy as np
from Calculation_functions import Calculation_functions_class





    '''
    
    The following electrical inputs are required run the capacitor lifetime simulation:
    
    - V_per_cap  : Capacitor Voltage
    - I_per_cap  : Capacitor Current
    
    These values may be provided directly by the user, or they can be computed from a full inverter setup using mission
    profile data. Using the inverter setup is often preferred, since these values can be extracted can be extracted directly
    from realistic operating conditions (Mission profiles of Active and reactive power of the inverter).
    
    '''

    # ----------------------------------------#
    # Development of mission profile
    # ----------------------------------------#

    S_in = 50000
    pf_in = 1

    P_in = abs(S_in*pf_in)
    Q_in = np.sqrt(S_in**2 - P_in**2)
    if pf_in<0:
        Q_in = Q_in*-1
    Profile_size = 10

    # ----------------------------------------#
    # Inverter setup
    # ----------------------------------------#


    P = np.full(Profile_size, P_in)
    Q = np.full(Profile_size, Q_in)
    f_sw = 10 * 1000                          # [Hz] Inverter switching frequency
    Vs = np.full(Profile_size, 230)   # [V] Inverter phase RMS AC side voltage
    V_dc = np.full(Profile_size, 600) # [V] Inverter DC side voltage

    M = np.full(Profile_size, 1) # [-] Inverter modulation index # Modulation cannot be above 1 as model does not take into account. +

    inverter_phases = 3  # 1 or 3 (single-phase or three-phase)
    if inverter_phases not in (1, 3):
        raise ValueError("phases must be 1 or 3")

    modulation_scheme = "svm"  # options: "spwm" or "svm" , the type of modulation once can choose for three phase inverters."svm" is  Space Vector PWM (or Third-Harmonic Injection) and "spwm" is Sinusoidal PWM (reference = pure sine).
    if modulation_scheme not in ("spwm", "svm"):  # when inverter_phases == 1 this variable is invalid.
        raise ValueError("modulation_scheme must be 'spwm' or 'svm'")

    N_parallel = 4  # Number of Capacitors in parallel
    N_series = 1   # Number of Capacitors in series

    Vs, Is, phi, V_dc, pf, M, S = Calculation_functions_class.compute_power_flow(P=P, Q=Q, V_dc=V_dc, Vs=Vs, M=M,
                                                                                 modulation_scheme=modulation_scheme)

    # ----------------------------------------#
    # Ambient temperatures
    # ----------------------------------------#

    T_amb = np.full(Profile_size, 298.15)

    # ----------------------------------------#
    # Capacitor rated Voltage and current limits
    # ----------------------------------------#

    Max_voltage_datasheet_cap = 800   # [V]
    Max_current_datasheet_cap  = 34.5 # [A]

    # ----------------------------------------#
    # Capacitor dimensions
    # ----------------------------------------#

    Width = 60e-3    # [m]
    Height = 45e-3   # [m]
    Length = 57.5e-3 # [m]

    # ----------------------------------------#
    # Thermal resistance of capacitor
    # ----------------------------------------#

    Heat_coefficient = 192e-3  # [W/°C]
    Thermal_resistance = 1/Heat_coefficient # [°C/W] or [K/W]


    # ----------------------------------------#
    # Effective capacitor ESR at inverter switching frequency
    # ----------------------------------------#

    ESR_eff = 3.2e-3 # [Ohm]


    # ----------------------------------------#
    # Minimum insulation resistance of capacitor (from RC time constant)
    # ----------------------------------------#

    #Insulation resistance via a time constant
    Time_constant = 10000
    Capacitance_cap = 100e-6 # F
    #We know that Time_constant = Capacitance_cap * minimum_insulation_resistance
    minimum_insulation_resistance = Time_constant/Capacitance_cap # Value needs to calculate leakage current,
    # If available leakage current can be given directly as an input at different voltage levels



