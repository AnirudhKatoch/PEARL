# 01 - Full run: 90 days at 1 s

The normal PEARL v3 run on a synthetic 90-day PV profile (clouds, reactive support, daily ambient cycle), 2 devices in parallel, the original 2 K/W heat sink. Figures are in PEARL_output/Simulation_1/Figures, data in PEARL_output/Simulation_1/Data. Total time 22.3 s.

**Lifetime per device**

| device | lifetime_years | B10_years | share_of_damage_from_50Hz | Tj_max_C | Tj_mean_C | max_50Hz_dTj_K | mission_cycles |
|---|---|---|---|---|---|---|---|
| IGBT | 15.65 | 9.264 | 0.2872 | 136.2 | 43.95 | 20.93 | 10337 |
| Diode | 56.37 | 32.65 | 1.057e-06 | 117.2 | 41.09 | 8.157 | 10270 |

**Reliability (Monte Carlo + Weibull); Inverter = 12 devices in series**

| level | B1_years | B10_years | B50_years | weibull_shape |
|---|---|---|---|---|
| IGBT | 4.58 | 9.264 | 16.29 | 3.336 |
| Diode | 15.58 | 32.65 | 59.08 | 3.177 |
| Switch | 4.599 | 9.281 | 16.29 | 3.347 |
| Inverter | 2.189 | 4.417 | 7.755 | 3.347 |

**Where the time goes [s]**

| step | seconds |
|---|---|
| 0_inputs | 0.8899 |
| 1_bins | 0.7132 |
| 2_losses_ripple_cycles | 2.203 |
| 3_mean_temperature | 1.713 |
| 4_5_damage_reliability | 11.38 |
| 6_save | 0.9292 |
| 7_plots | 4.465 |
| total | 22.3 |

