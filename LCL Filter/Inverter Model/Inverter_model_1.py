import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -------------------------
# System parameters
# -------------------------

Vdc = 800              # DC bus voltage [V]
Vo = 400               # Output pulse amplitude (paper uses Vo)

f = 50                 # Fundamental frequency [Hz]
fsw = 10000            # Switching frequency [Hz]
T = 1 / f              # Fundamental period [s]
Tsw = 1 / fsw          # Switching period [s]

N = int(fsw / (2 * f)) # Number of pulses in half-cycle (paper definition)

M = 0.8                # Modulation index (controls pulse width)

# -------------------------
# Time vector for simulation
# -------------------------
samples_per_cycle = 50000
t = np.linspace(0, T, samples_per_cycle, endpoint=False)


########################################################################################################################
# Production of PMW voltage from the inverter
########################################################################################################################

def Sinusoidal_Pulse_Width_Modulation_One_Phase(t,M,f,Tsw,Vo,T):

    """
    Generate single-phase sinusoidal PWM (SPWM) voltage waveform.

    Parameters
    ----------
    t : array
    Time vector [s]
    M : float
    Modulation index [-]
    f : float
    Fundamental frequency [Hz]
    fsw : float
    Switching frequency [Hz]
    Vdc : float
    DC bus voltage [V]

    Returns
    -------
    vs : array
    PWM output voltage waveform [V]
    """

    def v_ref(t):
        return M * np.sin(2 * np.pi * f * t)

    def v_carrier(t):
        tau = (t % Tsw) / Tsw
        return 1.0 - np.abs(2 * tau - 1.0)

    def vs_half_spwm(t):
        return np.where(v_ref(t) >= v_carrier(t), Vo, 0.0)

    def vs_full_spwm(t):
        tt = t % T
        half_T = T / 2         # Half-cycle duration [s]
        return np.where(tt < half_T,vs_half_spwm(tt),-vs_half_spwm(tt - half_T))

    vs = vs_full_spwm(t)

    return vs

vs = Sinusoidal_Pulse_Width_Modulation_One_Phase(t=t,M=M,f=f,Tsw=Tsw,Vo=Vo,T=T)

def Sinusoidal_Pulse_Width_Modulation_Three_Phase(t, M, f, Tsw, Vdc):
    # Three references
    vref_a = M * np.sin(2 * np.pi * f * t)
    vref_b = M * np.sin(2 * np.pi * f * t - 2 * np.pi / 3)
    vref_c = M * np.sin(2 * np.pi * f * t + 2 * np.pi / 3)

    # Common carrier
    tau = (t % Tsw) / Tsw
    carrier = 1.0 - np.abs(2 * tau - 1.0)

    # Shift refs into [0, 1]
    ref_a = 0.5 * (vref_a + 1.0)
    ref_b = 0.5 * (vref_b + 1.0)
    ref_c = 0.5 * (vref_c + 1.0)

    # Pole voltages
    va0 = np.where(ref_a >= carrier, +Vdc/2, -Vdc/2)
    vb0 = np.where(ref_b >= carrier, +Vdc/2, -Vdc/2)
    vc0 = np.where(ref_c >= carrier, +Vdc/2, -Vdc/2)

    return va0, vb0, vc0

va0, vb0, vc0 = Sinusoidal_Pulse_Width_Modulation_Three_Phase(t=t, M=M, f=f, Tsw=Tsw, Vdc=Vdc)



########################################################################################################################
# Input for the LCL filter values
########################################################################################################################


########################################################################################################################
# Trial Case Study 1
########################################################################################################################

def Solving_filter_case_study_1(t, vs, R, L, C):

    def lrc_ode(t_now ,x ,t_grid ,vs_grid ,R ,L ,C ):
        iL, vC = x

        # interpolate vs at current solver time
        vs_now = np.interp(t_now, t_grid, vs_grid)

        diL_dt = (vs_now - vC) / L
        dvC_dt = (iL - (vC / R)) / C

        return [diL_dt, dvC_dt]

    # initial conditions
    x0 = [0.0, 0.0]   # [iL(0), vC(0)]

    sol = solve_ivp(fun=lambda t_now, x: lrc_ode(t_now, x, t, vs, R, L, C), t_span=(t[0], t[-1]), y0=x0, t_eval=t, method="RK45")

    iL = sol.y[0]
    vC = sol.y[1]

    # Capacitor
    vC = vC
    iC = iL - (vC / R)

    # Resistor
    vR = vC
    iR = vC / R

    # Inductor
    iL = iL
    vL = vs - vC

    return vC, iC, vR, iR, iL, vL

# Example parameters
R = 1.0*10
L = 100e-6
C = 50e-6

vC, iC, vR, iR, iL, vL = Solving_filter_case_study_1(t=t, vs=vs, R=R, L=L, C=C)


def Plotting_case_1():

    plt.figure(figsize=(6.4*2, 4.8))
    plt.plot(t, vL, label="v_L (inductor)", linewidth=1.2)
    plt.plot(t, vC, label="v_C (capacitor)", linewidth=1.2)
    plt.plot(t, vR, '--', label="v_R (resistor)", linewidth=1.2)
    plt.title("Voltages in L-RC Circuit")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Figures/Voltage_1.png")
    plt.close()


    plt.figure(figsize=(6.4*2, 4.8))
    plt.plot(t, iL, label="i_L (inductor)", linewidth=1.5)
    plt.plot(t, iC, label="i_C (capacitor)", linewidth=1.2)
    plt.plot(t, iR, label="i_R (resistor)", linewidth=1.2)
    plt.title("Currents in L-RC Circuit")
    plt.xlabel("Time [s]")
    plt.ylabel("Current [A]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Figures/Current_1.png")
    plt.close()

#Plotting_case_1()

########################################################################################################################
# Trial Case Study 2
########################################################################################################################

def Solving_filter_case_study_2(t, vs, R, L,L1, C):

    def lclr_ode(t_now, x, t_grid, vs_grid, R, L, L1, C):
        i, i1, vC = x

        vs_now = np.interp(t_now, t_grid, vs_grid)

        di_dt = (vs_now - vC) / L
        di1_dt = (vC - (R * i1)) / L1
        dvC_dt = (i - i1) / C

        return [di_dt, di1_dt, dvC_dt]

    x0 = [0.0, 0.0, 0.0]  # [i(0), i1(0), vC(0)]

    sol = solve_ivp(fun=lambda t_now, x: lclr_ode(t_now, x, t, vs, R, L, L1, C), t_span=(t[0], t[-1]), y0=x0, t_eval=t, method="RK45")

    i = sol.y[0]  # inverter-side inductor current
    i1 = sol.y[1]  # output/load current through R-L1 branch
    vC = sol.y[2]  # capacitor voltage

    # ----- Left Inductor L -----
    iL = i
    vL = vs - vC

    # ----- Capacitor C -----
    vC = vC
    iC = i - i1

    # ----- Right Inductor L1 -----
    iL1 = i1
    vL1 = vC - (R * i1)

    # ----- Resistor R -----
    iR = i1
    vR = R * i1

    # KCL at the node
    np.allclose(iL, iC + iR)

    # KVL in the right branch
    np.allclose(vC, vR + vL1)

    # KVL in the left branch
    np.allclose(vs, vL + vC)

    return iL ,vL ,vC ,iC ,iL1 ,vL1 ,iR ,vR

R =10
L = 100e-6*2
L1 = 300e-6*2
C = 50e-6*2

iL ,vL ,vC ,iC ,iL1 ,vL1 ,iR ,vR  = Solving_filter_case_study_2(t=t, vs=vs, R=R, L=L,L1=L1, C=C)

def Plotting_case_2():

    plt.figure(figsize=(6.4*2, 4.8))
    #plt.plot(t, vs, label="v_s (source)", linewidth=1.5)
    plt.plot(t, vL, label="v_L (left inductor)", linewidth=1.2)
    plt.plot(t, vC, label="v_C (capacitor)", linewidth=1.2)
    plt.plot(t, vL1, label="v_L1 (right inductor)", linewidth=1.2)
    plt.plot(t, vR, '--', label="v_R (resistor)", linewidth=1.2)
    plt.title("Voltages in L-C-LR Circuit")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Figures/Voltage_2.png")
    plt.close()

    plt.figure(figsize=(6.4*2, 4.8))
    plt.plot(t, iL, label="i_L (left inductor)", linewidth=1.5)
    plt.plot(t, iC, label="i_C (capacitor)", linewidth=1.2)
    plt.plot(t, iL1, label="i_L1 (right inductor)", linewidth=1.2)
    plt.plot(t, iR, '--', label="i_R (resistor)", linewidth=1.2)
    plt.title("Currents in L-C-LR Circuit")
    plt.xlabel("Time [s]")
    plt.ylabel("Current [A]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Figures/Current_2.png")
    plt.close()

#Plotting_case_2()

########################################################################################################################
# Finding the THD and the harmonics for Case Study
########################################################################################################################


########################################################################################################################
# New Case study with LCL filter connected to the grid
########################################################################################################################

def Solving_LCL_Filter_Grid_Connected(t, vs, vg, L1, L2, C):
    """
    Solve an LCL filter connected between an inverter voltage source and a known grid voltage.

    Naming convention
    -----------------
    V_s  : inverter voltage input
    V_g  : known grid voltage
    I_L1 : current through left inductor
    V_L1 : voltage across left inductor
    I_L2 : current through right inductor
    V_L2 : voltage across right inductor
    V_C  : capacitor voltage
    I_C  : capacitor current

    Differential equations
    ----------------------
    dI_L1/dt = (V_s - V_C) / L1
    dI_L2/dt = (V_C - V_g) / L2
    dV_C/dt  = (I_L1 - I_L2) / C
    """

    t = np.asarray(t)
    vs = np.asarray(vs)
    vg = np.asarray(vg)

    if t.ndim != 1 or vs.ndim != 1 or vg.ndim != 1:
        raise ValueError("t, vs, and vg must be 1D arrays.")

    if not (len(t) == len(vs) == len(vg)):
        raise ValueError("t, vs, and vg must have the same length.")

    if len(t) < 2:
        raise ValueError("t must contain at least two time points.")

    if L1 <= 0 or L2 <= 0 or C <= 0:
        raise ValueError("L1, L2, and C must be positive.")

    def lcl_ode(t_now, x, t_grid, vs_grid, vg_grid, L1, L2, C):
        I_L1, I_L2, V_C = x

        V_s_now = np.interp(t_now, t_grid, vs_grid)
        V_g_now = np.interp(t_now, t_grid, vg_grid)

        dI_L1_dt = (V_s_now - V_C) / L1
        dI_L2_dt = (V_C - V_g_now) / L2
        dV_C_dt  = (I_L1 - I_L2) / C

        return [dI_L1_dt, dI_L2_dt, dV_C_dt]

    # Initial conditions: [I_L1(0), I_L2(0), V_C(0)]
    x0 = [0.0, 0.0, 0.0]

    sol = solve_ivp(
        fun=lambda t_now, x: lcl_ode(t_now, x, t, vs, vg, L1, L2, C),
        t_span=(t[0], t[-1]),
        y0=x0,
        t_eval=t,
        method="RK45"
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    # State variables
    I_L1 = sol.y[0]
    I_L2 = sol.y[1]
    V_C  = sol.y[2]

    # Derived quantities
    V_L1 = vs - V_C
    V_L2 = V_C - vg
    I_C  = I_L1 - I_L2

    # Optional consistency checks
    kcl_ok = np.allclose(I_L1, I_C + I_L2)
    kvl_left_ok = np.allclose(vs, V_L1 + V_C)
    kvl_right_ok = np.allclose(V_C, V_L2 + vg)

    if not (kcl_ok and kvl_left_ok and kvl_right_ok):
        print("Warning: one or more KCL/KVL checks are not within tolerance.")

    return V_L1, I_L1, V_C, I_C, V_L2, I_L2


L1 = 100e-6
L2 = 100e-6
C  = 50e-6

V_g = 230 * np.sqrt(2) * np.sin(2 * np.pi * f * t)  # Voltage of the grid

V_L1, I_L1, V_C, I_C, V_L2, I_L2  = Solving_LCL_Filter_Grid_Connected(t=t,vs=vs,vg=V_g,L1=L1,L2=L2,C=C)


def Plotting_Grid_Connected_LCL_filter():

    plt.figure(figsize=(6.4*2, 4.8))
    #plt.plot(t, vs, label="v_s (source)", linewidth=1.5)
    plt.plot(t, V_L1, label="V_L1", linewidth=1.2)
    plt.plot(t, V_C, label="V_C", linewidth=1.2)
    plt.plot(t, V_L2, label="V_L2", linewidth=1.2)
    plt.title("Voltages in L-C-L filter")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Figures/Voltage_3.png")
    plt.close()

    plt.figure(figsize=(6.4*2, 4.8))
    plt.plot(t, I_L1, label="I_L1", linewidth=1.5)
    plt.plot(t, I_C, label="I_C", linewidth=1.2)
    plt.plot(t, I_L2, label="I_L2", linewidth=1.2)
    plt.title("Currents in L-C-L filter")
    plt.xlabel("Time [s]")
    plt.ylabel("Current [A]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Figures/Current_3.png")
    plt.close()

Plotting_Grid_Connected_LCL_filter()

def THD_and_harmonics(signal,t_ss):

    N = len(signal)
    dt = t_ss[1] - t_ss[0]
    fs = 1 / dt

    fft_vals = np.fft.fft(signal)
    fft_vals = np.abs(fft_vals) / N

    freqs = np.fft.fftfreq(N, d=dt)

    # keep only positive frequencies
    mask = freqs > 0
    freqs = freqs[mask]
    fft_vals = fft_vals[mask]

    f0 = 50   # or 60 depending on your setup

    fund_idx = np.argmin(np.abs(freqs - f0))
    I1 = fft_vals[fund_idx]

    harmonics = np.copy(fft_vals)
    harmonics[fund_idx] = 0

    THD = np.sqrt(np.sum(harmonics**2)) / I1

    print("THD (%):", THD * 100)

    num_harmonics = 20

    harmonic_numbers = np.arange(1, num_harmonics + 1)
    harmonic_amplitudes = []

    for n in harmonic_numbers:
        target_freq = n * f0
        idx = np.argmin(np.abs(freqs - target_freq))
        harmonic_amplitudes.append(fft_vals[idx])

    harmonic_amplitudes = np.array(harmonic_amplitudes)


    plt.figure(figsize=(10,5))
    plt.bar(harmonic_numbers, harmonic_amplitudes)

    plt.title("Harmonic Spectrum of Output Current i1(t)")
    plt.xlabel("Harmonic Number (n × f0)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.savefig("Figures/Harmonic_Spectrum.png")

THD_and_harmonics(signal=I_L2,t_ss=t)