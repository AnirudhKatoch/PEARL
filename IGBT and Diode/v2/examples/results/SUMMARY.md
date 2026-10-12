# PEARL v3 example studies - all results


## 01 - Full run: 90 days at 1 s

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


Files: results/01_full_run_90_days/

## 02 - Validation against a brute-force reference

20 minutes of broken clouds. Reference = the same losses simulated in time (no shortcut) followed by rainflow on every point. At 400 points per period (0.05 ms) the two agree within about 2 %. At 20 points (1 ms, the old PEARL time step) the brute force itself overestimates the 50 Hz swing, which shows up as a lower ratio.

**Damage ratio PEARL v3 / brute force**

| design | points_per_period | device | ratio | time_bruteforce_s | time_PEARL_v3_s |
|---|---|---|---|---|---|
| 2 devices, 2 K/W sink | 20 | IGBT | 0.965 | 0.63 | 0.0251 |
| 2 devices, 2 K/W sink | 20 | Diode | 0.9648 | 0.63 | 0.0251 |
| 2 devices, 2 K/W sink | 100 | IGBT | 0.9851 | 1.423 | 0.03555 |
| 2 devices, 2 K/W sink | 100 | Diode | 0.9863 | 1.423 | 0.03555 |
| 2 devices, 2 K/W sink | 400 | IGBT | 0.9896 | 5.062 | 0.07746 |
| 2 devices, 2 K/W sink | 400 | Diode | 0.9905 | 5.062 | 0.07746 |
| 1 device, 0.72 K/W sink | 20 | IGBT | 0.9461 | 0.5384 | 0.02398 |
| 1 device, 0.72 K/W sink | 20 | Diode | 0.9053 | 0.5384 | 0.02398 |
| 1 device, 0.72 K/W sink | 100 | IGBT | 0.9878 | 1.486 | 0.03696 |
| 1 device, 0.72 K/W sink | 100 | Diode | 0.9671 | 1.486 | 0.03696 |
| 1 device, 0.72 K/W sink | 400 | IGBT | 0.9991 | 4.753 | 0.0739 |
| 1 device, 0.72 K/W sink | 400 | Diode | 0.9782 | 4.753 | 0.0739 |


Files: results/02_validation_vs_bruteforce/

## 03 - Why P(t)*Rth overshoots

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


Files: results/03_why_Rth_overshoots/

## 04 - Loss model

v3 computes losses at every instant from the real phase current and the duty cycle; its period average matches the textbook average formulas (exactly for sine PWM with m <= 1; this operating point needs m = 1.08, so v3 uses SVPWM and differs slightly from the sine formula). The v1 per-time-step formulas give about half the IGBT loss.

**Mean losses at 0, 90 and 180 degrees [W]**

| phi_deg | IGBT_v3_W | IGBT_textbook_W | IGBT_v1_legacy_W | Diode_v3_W | Diode_textbook_W | Diode_v1_legacy_W |
|---|---|---|---|---|---|---|
| 0 | 95.75 | 94.02 | 48.23 | 10.68 | 12.55 | 0.6199 |
| 90 | 65.43 | 65.43 | 16.47 | 44.04 | 44.04 | 11.61 |
| 180 | 35.1 | 36.83 | 4.303 | 77.41 | 75.54 | 6.326 |


Files: results/04_loss_model/

## 05 - From PEARL v1 physics to PEARL v3

Every row adds one correction to the row above. The big jumps are the loss model (v1 losses were about half), the Foster thermal model (larger 50 Hz swings) and the 40 ms t_on floor.

**Lifetimes**

| step | IGBT_life_years | Diode_life_years | IGBT_B10_years | IGBT_share_50Hz | IGBT_dTj50_max_K | IGBT_Tj_max_C |
|---|---|---|---|---|---|---|
| 1 v1 physics: legacy losses, ladder, raw t_on, cold start, diode k=0.5 | 2817 | 6156 | 1669 | 1.293e-06 | 7.633 | 76.56 |
| 2 + physically consistent losses | 34.25 | 51.67 | 19.7 | 0.001156 | 14.27 | 131.6 |
| 3 + datasheet Foster Zth_jc | 20.84 | 46.09 | 12.09 | 0.09771 | 20.93 | 136.2 |
| 4 + t_on >= 40 ms (AN 21-001) | 16.25 | 46.09 | 9.586 | 0.2964 | 20.93 | 136.2 |
| 5 + steady-state start (v2) | 16.25 | 46.09 | 9.586 | 0.2964 | 20.93 | 136.2 |
| 6 + freewheeling-diode factor 0.65 (v3 default) | 16.25 | 59.91 | 9.586 | 0.2964 | 20.93 | 136.2 |


Files: results/05_v1_to_v3_steps/

## 06 - Lifetime models

The thermal part is computed once; every model then counts damage on the same cycles. The models agree on large, slow cycles but disagree strongly on small, short 50 Hz cycles (left plot), which is why the lifetimes differ by orders of magnitude. USER_CURVES_EXAMPLE uses invented numbers to show the format. Bayerer uses placeholder bond-wire data.

**Lifetime by model (30 days)**

| model | device | lifetime_years | B10_years | share_50Hz |
|---|---|---|---|---|
| semikron2024 | IGBT | 16.25 | 9.586 | 0.2964 |
| semikron2024 | Diode | 59.91 | 34.8 | 1.117e-06 |
| semikron2024_no_baseplate | IGBT | 12.12 | 7.072 | 0.1706 |
| semikron2024_no_baseplate | Diode | 28.79 | 16.7 | 5.384e-07 |
| semikron2024_sintered | IGBT | 31.65 | 18.56 | 0.1351 |
| semikron2024_sintered | Diode | 63.47 | 36.96 | 1.198e-06 |
| lesit | IGBT | 0.03402 | 0.02037 | 0.9964 |
| lesit | Diode | 7.005 | 4.623 | 0.9012 |
| bayerer2008 | IGBT | 0.6561 | 0.4443 | 0.981 |
| bayerer2008 | Diode | 30.66 | 21.62 | 0.7097 |
| my_module_PC_curves | IGBT | 13.86 | 11.07 | 0.5732 |
| my_module_PC_curves | Diode | 165.2 | 129 | 0.02132 |

**Checks of the user-curve and fitting options**

| check | value |
|---|---|
| Semikron formula entered as 3 digitised curves: IGBT lifetime ratio | 1.005 |
| same, diode | 0.9989 |
| fit_cma on 12 noisy points: n (true 5.2) | 5.228 |
| fit_cma: Ea in 1e-19 J (true 1.0) | 0.9998 |


Files: results/06_lifetime_models/

## 07 - Power-factor sweep (rated S, 2 devices, continuous operation, no cold start)

Delivering power loads the IGBT; charging (P < 0) and low power factor load the diode. Solid = Q delivered, dashed = Q absorbed (nearly identical). Continuous operation at full rated S is a stress test, not a mission.

**Delivering, Q delivered**

| pf | IGBT_loss_W | Diode_loss_W | IGBT_dTj50_K | Diode_dTj50_K | IGBT_life_y | Diode_life_y |
|---|---|---|---|---|---|---|
| 1 | 39.41 | 4.73 | 23.73 | 8.542 | 0.5751 | 8.906e+05 |
| 0.9 | 38.42 | 5.8 | 24.06 | 11.76 | 0.4996 | 4679 |
| 0.8 | 37.37 | 6.928 | 23.91 | 15.26 | 0.5333 | 97.02 |
| 0.7 | 36.29 | 8.091 | 23.54 | 18.71 | 0.6292 | 6.413 |
| 0.6 | 35.2 | 9.277 | 23.24 | 21.99 | 0.7235 | 0.9369 |
| 0.5 | 34.09 | 10.48 | 22.61 | 25.21 | 0.9742 | 0.2177 |
| 0.4 | 32.92 | 11.75 | 21.88 | 28.26 | 1.407 | 0.07208 |
| 0.3 | 31.75 | 13.02 | 21.13 | 31.67 | 2.087 | 0.02664 |
| 0.2 | 30.6 | 14.27 | 20.24 | 34.58 | 3.454 | 0.01317 |
| 0.1 | 29.39 | 15.58 | 19.26 | 37.27 | 6.271 | 0.007523 |
| 0 | 28.21 | 16.86 | 18.35 | 40.04 | 11.46 | 0.004547 |

**Charging, Q delivered**

| pf | IGBT_loss_W | Diode_loss_W | IGBT_dTj50_K | Diode_dTj50_K | IGBT_life_y | Diode_life_y |
|---|---|---|---|---|---|---|
| 1 | 17 | 29 | 9.612 | 55.81 | 1.673e+05 | 0.0005814 |
| 0.9 | 17.99 | 27.93 | 9.954 | 54.91 | 9.397e+04 | 0.0006382 |
| 0.8 | 19.04 | 26.81 | 9.891 | 54.45 | 1.042e+05 | 0.0006735 |
| 0.7 | 20.11 | 25.64 | 10.5 | 53.35 | 3.908e+04 | 0.0007592 |
| 0.6 | 21.21 | 24.46 | 11.84 | 52.01 | 5855 | 0.0008819 |
| 0.5 | 22.32 | 23.26 | 13.02 | 50.61 | 1359 | 0.001036 |
| 0.4 | 23.49 | 21.98 | 14.13 | 48.8 | 403.7 | 0.001288 |
| 0.3 | 24.66 | 20.72 | 15.26 | 46.79 | 135 | 0.001661 |
| 0.2 | 25.81 | 19.47 | 16.36 | 44.59 | 51.44 | 0.002239 |
| 0.1 | 27.02 | 18.15 | 17.4 | 42.49 | 22.79 | 0.00306 |
| 0 | 28.21 | 16.86 | 18.35 | 40.04 | 11.46 | 0.004547 |


Files: results/07_power_factor_sweep/

## 08 - Mission-profile resolution

The same 30 days, block-averaged before the simulation. Averaging removes short cloud dips, so fewer and smaller slow cycles are seen and the predicted lifetime grows. Use 1-s data when you have it.

**Results**

| data_every_s | IGBT_life_y | Diode_life_y | IGBT_share_50Hz | IGBT_mission_cycles | IGBT_Tj_max_C | run_time_s |
|---|---|---|---|---|---|---|
| 1 | 16.25 | 59.91 | 0.2964 | 3307 | 136.2 | 3.94 |
| 5 | 17.29 | 63.66 | 0.314 | 3303 | 136.2 | 1.112 |
| 10 | 18.45 | 68.34 | 0.3324 | 3247 | 136.2 | 0.6447 |
| 30 | 22.39 | 85.73 | 0.3895 | 2793 | 136.2 | 0.3032 |
| 60 | 26.91 | 107.2 | 0.4469 | 2219 | 136.1 | 0.1955 |
| 300 | 40.23 | 163.3 | 0.549 | 762 | 136.1 | 0.08086 |
| 600 | 44.87 | 181.3 | 0.5668 | 417 | 136 | 0.05893 |


Files: results/08_data_resolution/

## 09 - Time scales and capacitances

Thermal time constants of the network [s]: 7.5e-06, 4.4e-05, 0.0001, 0.00022, 0.00072, 0.00098, 0.0023, 0.0083, 0.015, 0.074, 0.11, 35. At a 1-s step the chip modes (<= 0.1 s) are already settled, so dropping their capacitance changes nothing; the heat sink (about 34 s here) must keep its capacitance: without it the mean temperature jumps with every cloud edge (errors up to about 60 K here) and the predicted lifetime roughly halves.

**Variants (7 days)**

| variant | IGBT_life_y | Diode_life_y | max_|Tj_mean - exact|_K | IGBT_mission_cycles | IGBT_share_50Hz |
|---|---|---|---|---|---|
| exact (all capacitances) | 14.62 | 54.22 | 0 | 785 | 0.2996 |
| chip quasi-static (tau < 1 s) | 14.62 | 54.23 | 0.09355 | 785 | 0.2996 |
| only heat sink dynamic (tau < 10 s) | 14.62 | 54.23 | 0.09355 | 785 | 0.2996 |
| pure Rth, no capacitance | 7.147 | 19.15 | 59.56 | 815 | 0.1535 |


Files: results/09_timescales_capacitance/

## 10 - Sensitivity

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


Files: results/10_sensitivity/

## 11 - Monte Carlo and reliability

Each sample shifts the cycle swings, the mean temperatures and the lifetime-model coefficient and slope a little, then recounts the damage of every stored cycle. B10 = time when 10 % have failed. The inverter fails when any of its 12 devices fails, so its B10 is shorter than one switch's.

**B-lives**

| uncertainty | level | B1_y | B10_y | B50_y | weibull_shape |
|---|---|---|---|---|---|
| low (+-2 % at 3 sigma) | IGBT | 9.915 | 13.16 | 16.52 | 8.294 |
| low (+-2 % at 3 sigma) | Diode | 35.6 | 48.03 | 61.05 | 7.852 |
| low (+-2 % at 3 sigma) | Switch | 9.915 | 13.16 | 16.52 | 8.294 |
| low (+-2 % at 3 sigma) | Inverter | 7.348 | 9.755 | 12.24 | 8.294 |
| default (+-5 % at 3 sigma) | IGBT | 4.767 | 9.629 | 16.92 | 3.343 |
| default (+-5 % at 3 sigma) | Diode | 16.55 | 34.69 | 62.79 | 3.175 |
| default (+-5 % at 3 sigma) | Switch | 4.786 | 9.645 | 16.92 | 3.353 |
| default (+-5 % at 3 sigma) | Inverter | 2.281 | 4.597 | 8.062 | 3.353 |
| high (+-15 % at 3 sigma) | IGBT | 0.4206 | 3.405 | 18.21 | 1.123 |
| high (+-15 % at 3 sigma) | Diode | 1.314 | 11.86 | 69.2 | 1.068 |
| high (+-15 % at 3 sigma) | Switch | 0.5901 | 3.722 | 16.3 | 1.276 |
| high (+-15 % at 3 sigma) | Inverter | 0.08415 | 0.5308 | 2.324 | 1.276 |


Files: results/11_monte_carlo_reliability/

## 12 - Numerical settings and initial state

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


Files: results/12_numerics_and_initial_state/

## 13 - Speed

v1 time is extrapolated from a measured 10-s run of the old BDF solver (thermal step only, without the 1-ms rainflow). v3 time includes losses, thermal, damage and a 5000-sample Monte Carlo, no plots.

**Timing**

| days | seconds_of_profile | PEARL_v3_s | PEARL_v1_estimate_h | speed_up | operating_bins |
|---|---|---|---|---|---|
| 1 | 8.64e+04 | 1.179 | 3.456 | 1.055e+04 | 1.332e+04 |
| 7 | 6.048e+05 | 2.713 | 24.19 | 3.21e+04 | 4.32e+04 |
| 30 | 2.592e+06 | 6.726 | 103.7 | 5.549e+04 | 1.68e+05 |
| 90 | 7.776e+06 | 17.05 | 311 | 6.569e+04 | 3.376e+05 |


Files: results/13_speed_benchmark/
