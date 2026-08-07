import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 15,
    "font.family": "Times New Roman",
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
})

# ----------------------------------------------------------------------
# Design values
# ----------------------------------------------------------------------

L1 = 115e-6
L2 = 6.54e-6
C0 = 0.0001671444
R3 = 0.01057967

fsw = 10000.0

LT = L1 + L2
fr0 = np.sqrt(LT / (L1 * L2 * C0)) / (2.0 * np.pi)

C_ratios = [1.0,0.95,0.90,0.85]
colors = ["black", "red", "blue", "green"]
styles = ["-", "-", "-", "-"]

# ----------------------------------------------------------------------
# Frequency axis in Hz
# ----------------------------------------------------------------------
f = np.logspace(2, 5, 5000)
s = 1j * 2.0 * np.pi * f

# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
plt.figure(figsize=(6.4, 4.8*0.75))

gains_at_fsw = []

for C_ratio, color, style in zip(C_ratios, colors, styles):

    C = C_ratio * C0

    numerator = 1.0 + s * R3 * C
    denominator = s**3 * L1 * L2 * C + s**2 * R3 * C * LT + s * LT
    G = numerator / denominator

    gain_db = 20.0 * np.log10(np.abs(G))
    gains_at_fsw.append(np.interp(fsw, f, gain_db))

    label = "$C/C_0 = %.2f$" % C_ratio

    plt.semilogx(f, gain_db, style, color=color, linewidth=1.5, label=label)

# ----------------------------------------------------------------------
# Switching frequency marker
# ----------------------------------------------------------------------
g_nominal = gains_at_fsw[0]
g_degraded = gains_at_fsw[-1]
delta_db = g_degraded - g_nominal
delta_percent = (10.0 ** (delta_db / 20.0) - 1.0) * 100.0

plt.axvline(fsw, color="grey", linewidth=1.5, linestyle="--")
plt.text(fsw * 0.95, -32.5, "$f_{sw}$", color="grey",horizontalalignment="right")

plt.axvline(fsw / 2.0, color="grey", linewidth=1.5, linestyle="--")
plt.text(fsw / 2.0 * 1.225, -32.5, "$f_{sw}/2$", color="grey",horizontalalignment="right")

plt.annotate("",xy=(fsw, g_degraded),xytext=(fsw, g_nominal))

plt.text(fsw * 1.05,(g_nominal + g_degraded) / 2.0,"$+%.1f$ dB" % delta_db,color="red",verticalalignment="center")

plt.xlabel("Frequency (kHz)")
plt.ylabel("$|I_g / V_i|$  (dB)")
plt.xlim(1000*2.5, 10000*1.5)
plt.xticks([3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 14000],
           ["3", "4", "5", "6", "7", "8", "9", "10", "12", "14"])
plt.ylim(-35, 17.5)
#plt.grid(True, which="both", alpha=0.25, linewidth=0.6)
plt.legend(frameon=False, loc="lower left")
plt.tight_layout()

plt.savefig("lcl_drift_bode.png", dpi=300, bbox_inches="tight")
plt.savefig("lcl_drift_bode.pdf", bbox_inches="tight")

# ----------------------------------------------------------------------
# Numbers for the caption
# ----------------------------------------------------------------------
print("f_r0                     : %8.1f Hz" % fr0)
print("f_sw / f_r0              : %8.2f" % (fsw / fr0))
print(f"gain at f_sw, C/C0 = {C_ratios[0]} : %8.2f dB" % g_nominal)
print(f"gain at f_sw, C/C0 = {C_ratios[-1]} : %8.2f dB" % g_degraded)
print("increase                 : %8.2f dB" % delta_db)
print("equivalent current rise  : %8.1f %%" % delta_percent)

for C_ratio in C_ratios:
    print("C/C0 = %.3f  ->  f_r = %8.1f Hz" % (C_ratio, fr0 / np.sqrt(C_ratio)))