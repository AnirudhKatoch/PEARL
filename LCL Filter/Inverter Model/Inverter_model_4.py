import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Helpers
# ============================================================

def ensure_dir(path="Figures"):
    os.makedirs(path, exist_ok=True)

def park_transform(a, b, c, theta):
    d = (2/3) * (a * np.cos(theta) + b * np.cos(theta - 2*np.pi/3)+ c * np.cos(theta + 2*np.pi/3))
    q = -(2/3) * (a * np.sin(theta)+ b * np.sin(theta - 2*np.pi/3)+ c * np.sin(theta + 2*np.pi/3))
    return d, q

def inverse_park_transform(d, q, theta):
    a = d * np.cos(theta) - q * np.sin(theta)
    b = d * np.cos(theta - 2 * np.pi / 3) - q * np.sin(theta - 2 * np.pi / 3)
    c = d * np.cos(theta + 2 * np.pi / 3) - q * np.sin(theta + 2 * np.pi / 3)
    return a, b, c

def rms_per_second(signal, samples_per_second):
    signal = np.asarray(signal)
    n_seconds = len(signal) // samples_per_second
    trimmed = signal[:n_seconds * samples_per_second]
    reshaped = trimmed.reshape(n_seconds, samples_per_second)
    rms = np.sqrt(np.mean(reshaped**2, axis=1))
    return rms

def plot_three_phase_full(t, mask_value, a, b, c, title, ylabel, filename, labels=("a", "b", "c")):
    mask = t <= mask_value
    plt.figure(figsize=(12, 5))
    plt.plot(t[mask], a[mask], label=labels[0])
    plt.plot(t[mask], b[mask], label=labels[1])
    plt.plot(t[mask], c[mask], label=labels[2])
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def plot_two_signals_full(t, mask_value, x, y, title, ylabel, filename, labels=("x", "y")):
    mask = t <= mask_value
    plt.figure(figsize=(12, 5))
    plt.plot(t[mask], x[mask], label=labels[0])
    plt.plot(t[mask], y[mask], label=labels[1])
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def plot_one_signal_full(t, mask_value, x, title, ylabel, filename, label="signal"):
    mask = t <= mask_value
    plt.figure(figsize=(12, 5))
    plt.plot(t[mask], x[mask], label=label)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

# ============================================================
# Simulation parameters
# ============================================================

ensure_dir("Figures")

mission_profile_size = 2      # seconds
points_per_cycle = 100
f = 50                          # Hz
omega = 2 * np.pi * f
V_rms = 230                     # phase RMS voltage
V_peak = np.sqrt(2) * V_rms

dt = 1 / (f * points_per_cycle)
t = np.arange(0, mission_profile_size, dt)
N = len(t)
samples_per_second = int(1 / dt)

Pref_profile = [5000.0, 4000.0]   # W
Qref_profile = [1000.0, 2000.0]   # var

Pref = np.repeat(Pref_profile, samples_per_second)[:N]
Qref = np.repeat(Qref_profile, samples_per_second)[:N]

# ============================================================
# Grid voltage (slack source)
# ============================================================

# Optional actual grid-angle disturbance
theta_grid = np.zeros(N)

# Example phase jump:
#t_dist = 1.0
#delta_theta_dist = 0.1
#theta_grid[t >= t_dist] = delta_theta_dist

theta_g = omega * t + theta_grid

v_slack_a = V_peak * np.cos(theta_g)
v_slack_b = V_peak * np.cos(theta_g - 2*np.pi/3)
v_slack_c = V_peak * np.cos(theta_g + 2*np.pi/3)

# ============================================================
# Network / filter resistances
# ============================================================

R_g = 0.1
R_f = 0.05

'''
# ============================================================
# Inverter voltage (for now fixed, same as nominal source)
# ============================================================

e_inv_rms = 240
e_inv_peak = np.sqrt(2) * e_inv_rms

e_inv_a = e_inv_peak * np.cos(omega * t)
e_inv_b = e_inv_peak * np.cos(omega * t - 2*np.pi/3)
e_inv_c = e_inv_peak * np.cos(omega * t + 2*np.pi/3)


# ============================================================
# PCC voltages and grid currents in abc
# ============================================================

v_g_a = (e_inv_a * R_g + v_slack_a * R_f) / (R_g + R_f)
v_g_b = (e_inv_b * R_g + v_slack_b * R_f) / (R_g + R_f)
v_g_c = (e_inv_c * R_g + v_slack_c * R_f) / (R_g + R_f)

i_g_a = (e_inv_a - v_slack_a) / (R_g + R_f)
i_g_b = (e_inv_b - v_slack_b) / (R_g + R_f)
i_g_c = (e_inv_c - v_slack_c) / (R_g + R_f)
'''





# ============================================================
# INITIAL INVERTER / NETWORK / PCC STATES
# ============================================================

# Initial inverter voltage magnitude
e_inv_rms_init = 230.0
e_inv_peak_init = np.sqrt(2) * e_inv_rms_init

# Inverter voltage commands in dq frame
e_inv_d = np.zeros(N)
e_inv_q = np.zeros(N)

# Inverter voltage in abc frame
e_inv_a = np.zeros(N)
e_inv_b = np.zeros(N)
e_inv_c = np.zeros(N)

# PCC voltage in abc frame
v_g_a = np.zeros(N)
v_g_b = np.zeros(N)
v_g_c = np.zeros(N)

# Grid current in abc frame
i_g_a = np.zeros(N)
i_g_b = np.zeros(N)
i_g_c = np.zeros(N)

# Initial inverter abc voltage
e_inv_a[0] = e_inv_peak_init * np.cos(0.0)
e_inv_b[0] = e_inv_peak_init * np.cos(-2*np.pi/3)
e_inv_c[0] = e_inv_peak_init * np.cos( 2*np.pi/3)

# Initial PCC voltage using resistive network model
v_g_a[0] = (e_inv_a[0] * R_g + v_slack_a[0] * R_f) / (R_g + R_f)
v_g_b[0] = (e_inv_b[0] * R_g + v_slack_b[0] * R_f) / (R_g + R_f)
v_g_c[0] = (e_inv_c[0] * R_g + v_slack_c[0] * R_f) / (R_g + R_f)

# Initial grid current using resistive network model
i_g_a[0] = (e_inv_a[0] - v_slack_a[0]) / (R_g + R_f)
i_g_b[0] = (e_inv_b[0] - v_slack_b[0]) / (R_g + R_f)
i_g_c[0] = (e_inv_c[0] - v_slack_c[0]) / (R_g + R_f)


# ============================================================
# PLL PARAMETERS AND STATES
# ============================================================

Tf_PLL = 0.01
kp_PLL = 0.1
ki_PLL = 10.0

theta_PLL = np.zeros(N)
theta_hat = np.zeros(N)
omega_hat = np.zeros(N)

v_d_pll = np.zeros(N)
v_q_pll = np.zeros(N)

i_d_pll = np.zeros(N)
i_q_pll = np.zeros(N)

v_q_fil = np.zeros(N)
I_PLL = np.zeros(N)
delta_PLL = np.zeros(N)
delta_omega = np.zeros(N)

# Initial PLL conditions
theta_hat[0] = 0.0
omega_hat[0] = omega

# Initial dq measurements using initial PCC values and initial estimated angle
v_d_pll[0], v_q_pll[0] = park_transform(v_g_a[0], v_g_b[0], v_g_c[0], theta_hat[0])
i_d_pll[0], i_q_pll[0] = park_transform(i_g_a[0], i_g_b[0], i_g_c[0], theta_hat[0])

v_q_fil[0] = v_q_pll[0]


# ============================================================
# POWER CONTROLLER PARAMETERS AND STATES
# ============================================================

Tf_PQ = 0.02
kp_PQ = 0.01
ki_PQ = 1.0

Pg = np.zeros(N)
Qg = np.zeros(N)

Pg_fil = np.zeros(N)
Qg_fil = np.zeros(N)

I_dP = np.zeros(N)
I_qQ = np.zeros(N)

i_d_ref = np.zeros(N)
i_q_ref = np.zeros(N)

# Initial power calculation
Pg[0] = 1.5 * (v_d_pll[0] * i_d_pll[0] + v_q_pll[0] * i_q_pll[0])
Qg[0] = 1.5 * (v_d_pll[0] * i_q_pll[0] - v_q_pll[0] * i_d_pll[0])

Pg_fil[0] = Pg[0]
Qg_fil[0] = Qg[0]

I_dP[0] = 0.0
I_qQ[0] = 0.0

i_d_ref[0] = kp_PQ * (Pref[0] - Pg_fil[0]) + I_dP[0]
i_q_ref[0] = kp_PQ * (Qref[0] - Qg_fil[0]) + I_qQ[0]


# ============================================================
# CURRENT CONTROLLER PARAMETERS AND STATES
# ============================================================

Tf_ig = 0.01
kp_ig = 0.005
ki_ig = 0.5

i_d_fil = np.zeros(N)
i_q_fil = np.zeros(N)

V_d_cc = np.zeros(N)
V_q_cc = np.zeros(N)

eps_d = np.zeros(N)
eps_q = np.zeros(N)

# Simplified no-inductor case: controller output directly sets dq inverter voltage
e_inv_d[0] = 0.0
e_inv_q[0] = 0.0

i_d_fil[0] = i_d_pll[0]
i_q_fil[0] = i_q_pll[0]

V_d_cc[0] = 0.0
V_q_cc[0] = 0.0

eps_d[0] = 0.0
eps_q[0] = 0.0



'''

for k in range(1, N):

    # ========================================================
    # 1) NETWORK / PLANT UPDATE USING PREVIOUS INVERTER COMMAND
    # ========================================================
    # Use inverter voltage from previous step to compute PCC voltage
    # and grid current at the current step.

    v_g_a[k] = (e_inv_a[k-1] * R_g + v_slack_a[k] * R_f) / (R_g + R_f)
    v_g_b[k] = (e_inv_b[k-1] * R_g + v_slack_b[k] * R_f) / (R_g + R_f)
    v_g_c[k] = (e_inv_c[k-1] * R_g + v_slack_c[k] * R_f) / (R_g + R_f)

    i_g_a[k] = (e_inv_a[k-1] - v_slack_a[k]) / (R_g + R_f)
    i_g_b[k] = (e_inv_b[k-1] - v_slack_b[k]) / (R_g + R_f)
    i_g_c[k] = (e_inv_c[k-1] - v_slack_c[k]) / (R_g + R_f)

    # ========================================================
    # 2) MEASURED dq VALUES USING PREVIOUS PLL ANGLE
    # ========================================================
    # Transform measured PCC voltage and measured current into dq frame.

    v_d_pll[k], v_q_pll[k] = park_transform(v_g_a[k], v_g_b[k], v_g_c[k], theta_hat[k-1])
    i_d_pll[k], i_q_pll[k] = park_transform(i_g_a[k], i_g_b[k], i_g_c[k], theta_hat[k-1])

    # ========================================================
    # 3) PLL UPDATE
    # ========================================================
    # Filter q-axis PCC voltage
    v_q_fil[k] = v_q_fil[k-1] + (dt / Tf_PLL) * (v_q_pll[k] - v_q_fil[k-1])

    # PLL PI controller
    I_PLL[k] = I_PLL[k-1] + ki_PLL * v_q_fil[k] * dt
    delta_PLL[k] = kp_PLL * v_q_fil[k] + I_PLL[k]

    # Frequency estimate
    delta_omega[k] = omega * delta_PLL[k]
    omega_hat[k] = omega + delta_omega[k]

    # Angle estimate
    theta_PLL[k] = theta_PLL[k-1] + delta_omega[k] * dt
    theta_hat[k] = omega * t[k] + theta_PLL[k]

    # ========================================================
    # 4) POWER CONTROLLER (PC)
    # ========================================================
    # Instantaneous active and reactive power
    Pg[k] = 1.5 * (v_d_pll[k] * i_d_pll[k] + v_q_pll[k] * i_q_pll[k])
    Qg[k] = 1.5 * (v_d_pll[k] * i_q_pll[k] - v_q_pll[k] * i_d_pll[k])

    # Low-pass filter on power
    Pg_fil[k] = Pg_fil[k-1] + (dt / Tf_PQ) * (Pg[k] - Pg_fil[k-1])
    Qg_fil[k] = Qg_fil[k-1] + (dt / Tf_PQ) * (Qg[k] - Qg_fil[k-1])

    # Active power PI -> d-axis current reference
    I_dP[k] = I_dP[k-1] + ki_PQ * (Pref[k] - Pg_fil[k]) * dt
    i_d_ref[k] = kp_PQ * (Pref[k] - Pg_fil[k]) + I_dP[k]

    # Reactive power PI -> q-axis current reference
    I_qQ[k] = I_qQ[k-1] + ki_PQ * (Qref[k] - Qg_fil[k]) * dt
    i_q_ref[k] = kp_PQ * (Qref[k] - Qg_fil[k]) + I_qQ[k]

    # ========================================================
    # 5) CURRENT CONTROLLER (CC)
    # ========================================================
    # Filter measured dq currents
    i_d_fil[k] = i_d_fil[k-1] + (dt / Tf_ig) * (i_d_pll[k] - i_d_fil[k-1])
    i_q_fil[k] = i_q_fil[k-1] + (dt / Tf_ig) * (i_q_pll[k] - i_q_fil[k-1])

    # d-axis current PI
    V_d_cc[k] = V_d_cc[k-1] + ki_ig * (i_d_ref[k] - i_d_fil[k]) * dt
    eps_d[k] = kp_ig * (i_d_ref[k] - i_d_fil[k]) + V_d_cc[k]

    # q-axis current PI
    V_q_cc[k] = V_q_cc[k-1] + ki_ig * (i_q_ref[k] - i_q_fil[k]) * dt
    eps_q[k] = kp_ig * (i_q_ref[k] - i_q_fil[k]) + V_q_cc[k]

    # Simplified case: no filter inductor
    e_inv_d[k] = eps_d[k]
    e_inv_q[k] = eps_q[k]

    e_inv_max = np.sqrt(2) * 230.0

    e_inv_d[k] = np.clip(eps_d[k], -e_inv_max, e_inv_max)
    e_inv_q[k] = np.clip(eps_q[k], -e_inv_max, e_inv_max)

    # ========================================================
    # 6) INVERTER VOLTAGE COMMAND IN abc FRAME
    # ========================================================
    e_inv_a[k], e_inv_b[k], e_inv_c[k] = inverse_park_transform(e_inv_d[k], e_inv_q[k], theta_hat[k])

# ============================================================
# Per-second RMS
# ============================================================

i_g_a_rms = rms_per_second(i_g_a, samples_per_second)
i_g_b_rms = rms_per_second(i_g_b, samples_per_second)
i_g_c_rms = rms_per_second(i_g_c, samples_per_second)

# ============================================================
# Console summary
# ============================================================

print("Simulation complete.")
print(f"Total simulation time: {mission_profile_size} s")
print(f"Time step dt: {dt:.6e} s")
print(f"Number of samples: {N}")

print("\nRMS current per second:")
print("i_g_a RMS:", i_g_a_rms)
print("i_g_b RMS:", i_g_b_rms)
print("i_g_c RMS:", i_g_c_rms)

print("\nFinal PLL values:")
print("theta_PLL[-1]   =", theta_PLL[-1])
print("theta_hat[-1]   =", theta_hat[-1])
print("delta_omega[-1] =", delta_omega[-1])
print("omega_hat[-1]   =", omega_hat[-1])
print("v_q_pll[-1]     =", v_q_pll[-1])
print("v_q_fil[-1]     =", v_q_fil[-1])

print("\nFinal Power Controller values:")
print("Pg[-1]      =", Pg[-1])
print("Qg[-1]      =", Qg[-1])
print("Pg_fil[-1]  =", Pg_fil[-1])
print("Qg_fil[-1]  =", Qg_fil[-1])
print("i_d_ref[-1] =", i_d_ref[-1])
print("i_q_ref[-1] =", i_q_ref[-1])

# ============================================================
# Plot everything for full 2-second simulation
# ============================================================

# 1. Three-phase voltages
plot_three_phase_full(
    t, 2, v_slack_a, v_slack_b, v_slack_c,
    title="Slack/Grid Voltages (abc) - Full 2 s",
    ylabel="Voltage (V)",
    filename="Figures/01_slack_voltages_abc_2s.png",
    labels=("v_slack_a", "v_slack_b", "v_slack_c"))

plot_three_phase_full(
    t, 2, e_inv_a, e_inv_b, e_inv_c,
    title="Inverter Voltages (abc) - Full 2 s",
    ylabel="Voltage (V)",
    filename="Figures/02_inverter_voltages_abc_2s.png",
    labels=("e_inv_a", "e_inv_b", "e_inv_c"))

plot_three_phase_full(
    t, 2, v_g_a, v_g_b, v_g_c,
    title="PCC Voltages (abc) - Full 2 s",
    ylabel="Voltage (V)",
    filename="Figures/03_pcc_voltages_abc_2s.png",
    labels=("v_g_a", "v_g_b", "v_g_c"))

# 2. Three-phase currents
plot_three_phase_full(
    t, 2, i_g_a, i_g_b, i_g_c,
    title="Grid Currents (abc) - Full 2 s",
    ylabel="Current (A)",
    filename="Figures/04_grid_currents_abc_2s.png",
    labels=("i_g_a", "i_g_b", "i_g_c"))

# 4. PLL internal signals
plot_two_signals_full(
    t, 2, v_q_pll, v_q_fil,
    title="PLL q-axis Voltage and Filtered q-axis Voltage - Full 2 s",
    ylabel="Voltage (V)",
    filename="Figures/05_pll_vq_and_filtered_2s.png",
    labels=("v_q_pll", "v_q_fil"))

plot_two_signals_full(
    t, 2, delta_PLL, I_PLL,
    title="PLL PI States - Full 2 s",
    ylabel="Signal",
    filename="Figures/06_pll_pi_states_2s.png",
    labels=("delta_PLL", "I_PLL"))

plot_two_signals_full(
    t, 2, delta_omega, omega_hat,
    title="PLL Frequency Correction and Estimated Frequency - Full 2 s",
    ylabel="rad/s",
    filename="Figures/07_pll_frequency_2s.png",
    labels=("delta_omega", "omega_hat"))

plot_two_signals_full(
    t, 2, theta_PLL, theta_hat,
    title="PLL Angle States - Full 2 s",
    ylabel="Angle (rad)",
    filename="Figures/8_pll_angle_states_2s.png",
    labels=("theta_PLL", "theta_hat"))

# 5. Closed-loop dq with PLL
plot_two_signals_full(
    t, 2, v_d_pll, v_q_pll,
    title="Closed-Loop dq Voltages with PLL - Full 2 s",
    ylabel="Voltage (V)",
    filename="Figures/9_closedloop_dq_voltages_2s.png",
    labels=("v_d_pll", "v_q_pll"))

plot_two_signals_full(
    t, 2, i_d_pll, i_q_pll,
    title="Closed-Loop dq Currents with PLL - Full 2 s",
    ylabel="Current (A)",
    filename="Figures/10_closedloop_dq_currents_2s.png",
    labels=("i_d_pll", "i_q_pll"))

# 6. Power controller signals
plot_two_signals_full(
    t, 2, Pg, Pg_fil,
    title="Active Power and Filtered Active Power - Full 2 s",
    ylabel="Power (W)",
    filename="Figures/14_active_power_and_filtered_2s.png",
    labels=("Pg", "Pg_fil"))

plot_two_signals_full(
    t, 2, Qg, Qg_fil,
    title="Reactive Power and Filtered Reactive Power - Full 2 s",
    ylabel="Power (var)",
    filename="Figures/15_reactive_power_and_filtered_2s.png",
    labels=("Qg", "Qg_fil"))

plot_two_signals_full(
    t, 2, i_d_ref, i_q_ref,
    title="Power Controller Current References - Full 2 s",
    ylabel="Current Reference",
    filename="Figures/16_power_controller_current_refs_2s.png",
    labels=("i_d_ref", "i_q_ref"))

plot_two_signals_full(
    t, 2, Pref, Qref,
    title="Dynamic Power References - Full 2 s",
    ylabel="Reference",
    filename="Figures/17_dynamic_power_references_2s.png",
    labels=("Pref", "Qref"))


# 6. RMS current per second
seconds_axis = np.arange(1, len(i_g_a_rms) + 1)
plt.figure(figsize=(12, 5))
plt.plot(seconds_axis, i_g_a_rms, marker='o', label='i_g_a RMS')
plt.plot(seconds_axis, i_g_b_rms, marker='o', label='i_g_b RMS')
plt.plot(seconds_axis, i_g_c_rms, marker='o', label='i_g_c RMS')
plt.xlabel("Second")
plt.ylabel("RMS Current (A)")
plt.title("Grid Current RMS Per Second")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Figures/13_grid_current_rms_per_second.png", dpi=200)
plt.close()

'''