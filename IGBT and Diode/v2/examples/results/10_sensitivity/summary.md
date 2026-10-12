# 10 - Sensitivity

Each row changes one input from the baseline. Temperature-dependent losses (kappa) and the heat sink move the lifetime the most; small changes in Zth_jc also matter because the lifetime model is steep.

**Lifetime factors**

| case | IGBT_life_y | Diode_life_y | IGBT_factor | Diode_factor | IGBT_Tj_max_C | IGBT_dTj50_max_K |
|---|---|---|---|---|---|---|
| baseline (2 devices, 2 K/W, 10 kHz, 600 V) | 16.25 | 59.91 | 1 | 1 | 136.2 | 20.93 |
| losses +0.1 %/K (kappa 0.001) | 7.204 | 33.97 | 0.4434 | 0.5669 | 147.7 | 23.19 |
| losses +0.2 %/K (kappa 0.002) | 2.721 | 18.36 | 0.1675 | 0.3064 | 161.8 | 25.97 |
| losses +0.3 %/K (kappa 0.003) | 0.8864 | 9.419 | 0.05456 | 0.1572 | 179 | 29.43 |
| heat sink 1.14 K/W (2 m/s) | 60.47 | 699.8 | 3.722 | 11.68 | 102.1 | 20.93 |
| heat sink 0.72 K/W (10U, 4 m/s) | 122.2 | 4872 | 7.521 | 81.32 | 85.52 | 20.93 |
| 3 devices in parallel | 400.1 | 939 | 24.63 | 15.67 | 95.41 | 12.88 |
| Zth_jc +10 % | 9.291 | 55.26 | 0.5718 | 0.9224 | 139 | 22.98 |
| Zth_jc -10 % | 24.64 | 64.99 | 1.516 | 1.085 | 133.3 | 18.87 |
| DC link 580 V | 17.3 | 70.04 | 1.065 | 1.169 | 134.9 | 20.87 |
| DC link 560 V (too low for 230 V AC) (not possible: Invalid input: AC phase RMS voltage exceeds the theoretical ) | nan | nan | nan | nan | nan | nan |
| switching 5 kHz | 82.04 | 240.5 | 5.05 | 4.014 | 114.2 | 16.95 |
| switching 20 kHz | 0.6744 | 7.097 | 0.04151 | 0.1185 | 180.1 | 28.87 |

