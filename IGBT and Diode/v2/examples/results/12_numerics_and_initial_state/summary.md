# 12 - Numerical settings and initial state

The numerical settings (points per period, binning, slow gate) change the lifetime by a few percent at most. The start condition matters for short repeated profiles: 'ambient' counts a warm-up cycle every repetition.

**Settings (30 days)**

| setting | IGBT_life_y | Diode_life_y | IGBT_vs_default_% | Diode_vs_default_% | operating_bins | mission_cycles_IGBT | time_s |
|---|---|---|---|---|---|---|---|
| Np = 20 | 16.43 | 61.27 | 1.124 | 2.263 | 168026 | 3307 | 3.086 |
| Np = 40 (default) | 16.25 | 59.91 | 0 | 0 | 168026 | 3307 | 3.737 |
| Np = 100 | 16.19 | 58.91 | -0.3707 | -1.666 | 168026 | 3307 | 5.665 |
| no operating-point binning | 16.25 | 59.93 | 0.0265 | 0.03968 | 1585273 | 3295 | 14.44 |
| coarse binning (0.1 A, 5 mrad) | 16.25 | 59.92 | -0.002132 | 0.01865 | 25436 | 3327 | 2.982 |
| slow_gate = 0 | 16.24 | 59.91 | -0.02078 | -8.89e-08 | 168026 | 7881 | 4.054 |
| slow_gate = 0.5 K | 16.25 | 59.91 | 0.0004928 | 2.276e-09 | 168026 | 3152 | 3.679 |
| init = ambient (cold start) | 16.25 | 59.91 | 0 | 0 | 168026 | 3307 | 3.575 |
| init = periodic | 16.24 | 59.85 | -0.03814 | -0.1022 | 168026 | 3304 | 4.744 |

**1 h constant operation, three start conditions**

| pf | init | IGBT_life_y | Diode_life_y | IGBT_damage_share_from_slow_cycles |
|---|---|---|---|---|
| 1 | ambient | 0.5095 | 8.47 | 0.1249 |
| 1 | steady | 0.5751 | 8.906e+05 | 1.414e-05 |
| 1 | periodic | 0.5751 | 8.906e+05 | 0 |
| 0.7 | ambient | 0.5527 | 2.628 | 0.1324 |
| 0.7 | steady | 0.6292 | 6.413 | 2.625e-05 |
| 0.7 | periodic | 0.6292 | 6.413 | 0 |
| 0.4 | ambient | 1.087 | 0.07091 | 0.2372 |
| 0.4 | steady | 1.407 | 0.07208 | 0 |
| 0.4 | periodic | 1.407 | 0.07208 | 0 |

