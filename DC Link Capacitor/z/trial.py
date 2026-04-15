import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({"font.size": 15, "font.family": "Times New Roman", "axes.labelsize": 15, "axes.titlesize": 15,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15})

# Time axis: 1 year with hourly resolution
hours = 365 * 24
t = np.linspace(0, hours, hours)

np.random.seed(0)

dc_link = np.cumsum(np.random.normal(loc=(0.5/hours)*(t/hours)**0.2, scale=0.00001, size=hours))
semiconductor = np.cumsum(np.random.normal(loc=(0.4/hours)*(t/hours)**0.4, scale=0.00001, size=hours))
lcl_filter = np.cumsum(np.random.normal(loc=(0.6/hours)*(t/hours)**0.3, scale=0.00001, size=hours))

# Normalize
dc_link = dc_link * (0.8 / dc_link[-1])
semiconductor = semiconductor * (0.3 / semiconductor[-1])
lcl_filter = lcl_filter * (0.7 / lcl_filter[-1])


# System level damage (average)
system_damage = (dc_link + semiconductor + lcl_filter) / 3



# Plot
plt.figure(figsize=(6.4, 4.8))
plt.plot(t, dc_link, label='DC-link Capacitor')
plt.plot(t, semiconductor, label='Semiconductor')
plt.plot(t, lcl_filter, label='LCL Filter')
plt.plot(t, system_damage, label='System Damage (Average)', linewidth=2)

# Red dotted failure line at Damage = 1
plt.axhline(y=1, linestyle='--', color='red')

# Labels
plt.xlabel('Time (hours)')
plt.ylabel('Damage Accumulation')
plt.ylim(0, 1.1)
plt.xlim(0, hours)
plt.legend()
plt.grid(True)


plt.savefig("Damage_accumulation.png")