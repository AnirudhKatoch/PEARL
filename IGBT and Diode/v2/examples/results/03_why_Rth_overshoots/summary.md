# 03 - Why P(t)*Rth overshoots

|Zth| falls with frequency. Rth (0 Hz) is right for the mean temperature; the 50 Hz ripple sees |Zth| at 50, 100, 150 Hz..., where the heat sink contributes almost nothing. Using Rth for the ripple therefore overshoots by hundreds of percent. Columns '..._%' are errors relative to the exact periodic solution.

**Thermal impedance [K/W]**

| frequency_Hz | IGBT_Tj_per_W_IGBT | Diode_Tj_per_W_Diode | case_per_W_total |
|---|---|---|---|
| 0 | 2.453 | 3.053 | 2.003 |
| 0.001 | 2.398 | 2.991 | 1.957 |
| 0.01 | 1.1 | 1.591 | 0.8327 |
| 0.1 | 0.4677 | 1.062 | 0.09151 |
| 1 | 0.4287 | 0.9977 | 0.009679 |
| 10 | 0.2675 | 0.7556 | 0.003267 |
| 50 | 0.1685 | 0.5411 | 0.003008 |
| 100 | 0.1439 | 0.4306 | 0.002676 |
| 150 | 0.1296 | 0.3727 | 0.002307 |

**Error of each shortcut [%] (min..max over 12 operating points)**

| device | naive_P(t)*Rth_ja_% min | P(t)*Rth_jc_% min | 1st_harmonic_% min | harmonics_1-5_% min | rectangular_pulse_formula_% min | naive_P(t)*Rth_ja_% max | P(t)*Rth_jc_% max | 1st_harmonic_% max | harmonics_1-5_% max | rectangular_pulse_formula_% max |
|---|---|---|---|---|---|---|---|---|---|---|
| Diode | 491 | 103.3 | -29.31 | -4.521 | -46.26 | 566.8 | 129.3 | -12.25 | 0.3494 | -22.25 |
| IGBT | 1237 | 145.1 | -19.01 | -0.8192 | -36.57 | 1378 | 171 | -8.739 | 0.1071 | -25.63 |

