import numpy as np
from All_the_functions import All_the_functions_class
from matplotlib import pyplot as plt


Functions = All_the_functions_class()

# -------------------------
# System parameters
# -------------------------

Vdc_rated = 2100                              # [V] Rated DC bus voltage, defines maximum available inverter voltage level
Vo_rated = 1050                               # [V] Rated PWM pulse amplitude, instantaneous switched level (±Vo), must be ≤ allowed by topology
inverter_phases = 3                           # [-] ["1" or "3"] Number of phases: 1 = single-phase inverter, 3 = three-phase inverter
M_rated = 1                                   # [-] Modulation index, controls PWM pulse widths and sets fundamental output voltage magnitude (0 ≤ M ≤ 1 in linear region)
single_phase_inverter_topology = "full"       # ["full" or "half"] Single-phase topology: "half" = ±Vdc/2 output, "full" = ±Vdc output (H-bridge)
waveform_voltage_definition = "pole_voltage"  # ["switched_output" or "pole_voltage"] Voltage meaning: "switched_output" = load voltage, "pole_voltage" = single leg voltage (±Vdc/2) # One phase inverter can have "switched_output" or "pole_voltage", Three phae inverter only has pole_voltage
modulation_scheme = "spwm"                    # ["spwm" or "svm"] # PWM strategy used to generate switching signals; "spwm" = Sinusoidal PWM , "svm" = Space Vector Modulation; NOTE: current system supports only "spwm" and does NOT support "svm"
f = 50                                        # [Hz] Fundamental frequency, desired AC output frequency of the inverter (e.g., grid frequency 50 Hz)
fsw = 10000                                   # [Hz] Switching frequency, frequency at which PWM switches turn ON/OFF (carrier frequency)
T = 1 / f                                     # [s] Fundamental period, time for one full AC cycle (e.g., 20 ms for 50 Hz)
Tsw = 1 / fsw                                 # [s] Switching period, time for one PWM switching cycle (e.g., 100 µs for 10 kHz)
omega = 2 * np.pi * f                         # [Hz] Angular frequency

Vo_rated, Vo_theoretical_max = Functions.validate_or_set_pulse_amplitude(Vdc_rated=Vdc_rated, inverter_phases=inverter_phases,
                                                                         single_phase_inverter_topology=single_phase_inverter_topology,
                                                                         waveform_voltage_definition=waveform_voltage_definition, Vo=Vo_rated)

Vs_RMS_max_theoretical = Functions.compute_theoretical_fundamental_rms_limit(Vdc_rated=Vdc_rated, M=M_rated, inverter_phases=inverter_phases,
                                                                             modulation_scheme=modulation_scheme, single_phase_inverter_topology=single_phase_inverter_topology)


# -------------------------
# Mission profiles
# -------------------------

Profile_size = 2                             # Profile size in seconds
Vdc_RMS = np.full(Profile_size, 800) # [V] Mission profile of DC bus voltage at a resolution of 1 sec
M = np.full(Profile_size, 1)         # [-] Mission profile of modulation index at a resolution of 1 sec
Vo = np.full(Profile_size, Vo_rated)         # [V] PWM pulse amplitude
Vg_RMS = np.full(Profile_size, 230)  # [V] Voltage of the
S_RMS = np.full(Profile_size, 1e6)   # [VA] Apparent Power, per sec resolution
pf = np.full(Profile_size, -0.77)            # [-] Power factor, per sec resolution, Negative is inductive and positive is capacitive

P_RMS = S_RMS * pf                               # [W] Active power
Q_RMS = S_RMS * np.sqrt(1 - pf**2) * np.sign(pf) # [Var]






resolution_per_cycle = 5000                  # Resolution per cycle
dt = T / resolution_per_cycle                # Number of steps per sec

samples_per_switching_period = Tsw/dt
if samples_per_switching_period < 15:
    raise ValueError(
        "Insufficient PWM simulation resolution.\n"
        f"Current samples per switching period = "
        f"{samples_per_switching_period:.2f}\n"
        "At least 10 samples per switching period are required "
        "to accurately resolve PWM switching events, carrier "
        "intersections, and inverter voltage transitions.\n")



t = np.arange(0, Profile_size, dt)                                          # Time step profile
samples_per_second = resolution_per_cycle * f                               # Samples per sec

Vg = np.sqrt(2) * np.repeat(Vg_RMS, samples_per_second) * np.sin(omega * t)  # Grid instantaneous voltage
Ig_RMS = S_RMS / Vg_RMS

# Power-factor angle
pf_inst = np.repeat(pf, samples_per_second)
phi = np.arccos(np.abs(pf_inst))
phase_shift = np.sign(pf_inst) * phi

Ig_ref = np.sqrt(2) * np.repeat(Ig_RMS, samples_per_second) * np.sin(omega * t + phase_shift) # Grid instantaneous Current

#Functions.plot_voltage_current(t=t,Vg=Vg,Ig=Ig_ref,t_end=0.02,voltage_label="Vg",current_label="Ig",Location="Figures/Voltage_and_Current")

L1  = 100e-6
L2  = 100e-6
C   = 50e-6
R1 = 0.05
R2 = 0.05

Vs_ref = Functions.Inverse_LCL_Filter_Grid_Connected_for_Vs(t=t, V_g=Vg, I_L2=Ig_ref, L1=L1, L2=L2, C=C, R1=R1, R2=R2)  # Reference for Inverter switching

#Functions.plot_voltage_signal(t=t, Vs=Vs_ref, t_end=0.02, voltage_label="Vs_ref", Location=r"Figures\1.png")

Vs_peak = np.max(np.abs(Vs_ref))
if Vs_peak > Vo_rated:
    raise ValueError(
        "Not feasible: required inverter voltage exceeds capability.\n"
        f"Required peak = {Vs_peak:.2f} V, Available = {Vo_rated:.2f} V")

Vs = Functions.Sinusoidal_Pulse_Width_Modulation_One_Phase(P_RMS=P_RMS, t=t, Vo=Vo, Vs_ref=Vs_ref, Tsw=Tsw)

#Functions.checking_V_s_to_Vs_ref(t=t,Tsw=Tsw,dt=dt,Vs=Vs,Vs_ref=Vs_ref,t_end=0.02, Location=r"Figures\checking_V_s_to_Vs_ref.png")


V_L1, I_L1, V_C, I_C, V_L2, I_L2 = Functions.LCL_Filter_Grid_Connected(t, Vs, Vg, L1, L2, C, R1, R2)


#Functions.checking_I_L2_to_Ig_ref(t=t,Ig_ref=Ig_ref, I_L2=I_L2, t_end=0.02, Location=r"Figures\checking_I_L2_to_Ig_ref.png")


def compute_THD_one_cycle(t, signal, f, Location, cycle_start=None ):

    t = np.asarray(t)
    signal = np.asarray(signal)

    T = 1 / f
    dt = t[1] - t[0]
    fs = 1 / dt

    if cycle_start is None:
        cycle_start = t[-1] - T

    cycle_end = cycle_start + T

    mask = (t >= cycle_start) & (t < cycle_end)

    y = signal[mask]
    # remove DC offset
    y = y - np.mean(y)
    N = len(y)
    # FFT
    Y = np.fft.rfft(y)
    # RMS spectrum
    mag_rms = np.abs(Y) * np.sqrt(2) / N
    mag_rms[0] = np.abs(Y[0]) / N
    freqs = np.fft.rfftfreq(N, d=dt)

    # fundamental bin
    fundamental_idx = np.argmin(np.abs(freqs - f))
    I1_rms = mag_rms[fundamental_idx]

    # harmonic bins: 2nd harmonic and above
    harmonic_rms_sq = 0.0

    max_harmonic = int(freqs[-1] // f)

    for h in range(2, max_harmonic + 1):
        idx = np.argmin(np.abs(freqs - h * f))
        harmonic_rms_sq += mag_rms[idx] ** 2

    THD = np.sqrt(harmonic_rms_sq) / I1_rms
    THD_percent = THD * 100

    max_plot_harmonic = 20

    harmonic_orders = np.arange(1, max_plot_harmonic + 1)

    harmonic_rms = []

    for h in harmonic_orders:
        idx = np.argmin(np.abs(freqs - h * f))
        harmonic_rms.append(mag_rms[idx])

    harmonic_rms = np.array(harmonic_rms)

    plt.figure(figsize=(6.4, 4.8))
    plt.bar(harmonic_orders, harmonic_rms)
    plt.xlabel("Harmonic Order")
    plt.ylabel("RMS Current [A]")
    plt.title(f"I_L2 Harmonics up to {max_plot_harmonic}th | THD = {THD_percent:.2f}%")
    plt.grid(True)
    plt.savefig(Location)

    return THD, THD_percent, I1_rms, freqs, mag_rms

THD, THD_percent, I1_rms, freqs, mag_rms = compute_THD_one_cycle(t=t,signal=I_L2,f=f, Location = r"Figures/I_L2_Harmonics.png")


print("Ig_RMS",Ig_RMS[0])
print(f"I_L2 fundamental RMS = {I1_rms:.3f} A")
print(f"I_L2 THD = {THD_percent:.3f} %")









