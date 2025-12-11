import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------
# parameters
# -------------------------------------
ESR_eff = 3.6e-3
minimum_insulation_resistance = 10000 / (100e-6)
Thermal_resistance = 1/0.2
T_amb = 298.15

# -------------------------------------
# sweep values
# -------------------------------------
V_range = np.linspace(0, 800, 50)
I_range = np.linspace(0, 40, 50)

V_list = []
I_list = []
T_list = []

# -------------------------------------
# compute T_core for all (V, I)
# -------------------------------------
for V in V_range:
    for I in I_range:

        P_ripple = I**2 * ESR_eff
        I_leak = V / minimum_insulation_resistance
        P_leak = I_leak * V
        P_losses = P_ripple + P_leak

        T_core = T_amb + 1.5 * Thermal_resistance * P_losses
        T_core_C = T_core - 273.15   # convert to °C

        V_list.append(V)
        I_list.append(I)
        T_list.append(T_core_C)

# -------------------------------------
# scatter plot
# -------------------------------------
plt.figure(figsize=(8,6))
sc = plt.scatter(I_list, V_list, c=T_list, s=20)

plt.xlabel("I_per_cap (A)")
plt.ylabel("V_per_cap (V)")
plt.title("Scatter Plot of T_core for All V–I Combinations")
plt.colorbar(sc, label="T_core (°C)")

plt.show()
