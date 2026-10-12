# Parameters and outputs

## Parameters (Input_parameters.py; any of them can be passed to mother_function as keyword)

### Method
| Name | Default | Meaning |
|---|---|---|
| profile_dt | 1.0 s | time step of P, Q, T_env |
| loss_model | "physical" | "legacy" = PEARL v1 formulas (comparison only) |
| junction_model | "foster" | "ladder" = PEARL v1 thermal network (comparison only) |
| ton_rule, ton_min | "semikron", 0.04 s | t_on floor of the v1-style Semikron path ("raw" switches it off) |
| lifetime_model | "semikron2024" | name, dict, user curves, or list (see LIFETIME_MODELS.md) |
| Np | 40 | samples per 50 Hz period |
| op_resolution | (0.02 A, 1e-3 rad) | operating-point bin size; None = no binning |
| init | "steady" | start state: "steady", "ambient" (cold start), "periodic" (profile repeats) |
| slow_gate | 0.05 K | ignore mission-profile reversals smaller than this |
| quasi_static_tau | 0 s | thermal modes faster than this use Rth only |
| kappa_IGBT, kappa_Diode, kappa_T_ref, coupling_iterations | 0, 0, 298.15 K, 4 | electro-thermal coupling |
| temperature_limit_action | "warn" | "raise" stops when Tj exceeds the limit |
| matrix_bins | (0.5 K, 2 degC) | rainflow-matrix bins for the figures |
| save_per_second, results_dir, Plotting_flag, plot_day | False, "Results", True, None | output control |
| MC_samples, MC_samples_other_models, MC_sigma, MC_seed | 5000, 2000, {dT: 5%/3, Tjm_K: 2, A: 5%/3, alpha: 5%/3}, 0 | Monte Carlo |

### Lifetime model (Semikron default set) and limits
A0 2.9e9, A1 60, T0_K 40, lambda_K 17, alpha -4.3, Ea_J 4.5e-20, kB 1.38e-23, C 1, gamma -0.75,
k_thickness {IGBT 1, Diode 0.65}. Limits: max_V_CE 600 V, max_IGBT_current 50 A, max_Diode_current 30 A,
max temperatures 448.15 K, IGBT_max_life / Diode_max_life 15 y (only for the "capped" values).

### Thermal (IKW50N60H3 + Kryonaut + Cool Innovations 3-121211U)
r_I, tau_I, r_D, tau_D (datasheet Foster), r_paste 0.0032 K/W, tau_paste 1 ms, r_sink 2 K/W,
sink_mass 19 g, sink_specific_heat 900 J/(kg K).

### Device losses (25 degC datasheet values)
f_sw 10 kHz, t_on 60 ns, t_off 259 ns, I_ref 30 A, V_ref 400 V, Q_rr 0.88 uC, Err_D (None -> Q_rr x V_ref),
I_I_points / V_I_points and I_D_points / V_D_points (linear fit -> R and V0).

### Inverter
Vs_rms 230 V, V_dc_value 600 V, M 1, inverter_phases 3, modulation_scheme "svm",
single_phase_inverter_topology "full", N_parallel 1, f 50 Hz.

## Outputs

### `results` (returned by mother_function)
| Key | Content |
|---|---|
| `IGBT`, `Diode` | lifetime_years, damage_total / damage_50Hz / damage_mission_cycles, share_50Hz, Tj_max_C, Tj_mean_C, dTj_50Hz_max_K, n_mission_cycles, time_above_limit, rainflow_matrix |
| `reliability` | Weibull shape, scale, B1, B10, B50 for IGBT, Diode, Switch, Inverter |
| `lifetime_models` | the same per lifetime model when several are selected |
| `per_second` | Tj_mean_I/D, Tj_max_I/D, dTj_50Hz_I/D, damage_50Hz_I/D, T_case, T_sink, P_mean_I/D |
| `hourly` | t_start_h, Tj_mean/max per device, damage_50Hz / damage_mission per device, P_kW, T_amb_C |
| `monte_carlo_years` | lifetime samples per IGBT, Diode, Switch |
| `params`, `network`, `bins` | the inputs, the thermal network, the operating-point bins |
| `timing_s`, `settings`, `folders` | run information |

### Files in `Results/Simulation_N/`
| File | Content |
|---|---|
| Data/summary.json | settings, lifetimes, damage split, reliability, all lifetime models, timing |
| Data/hourly.parquet | the hourly table |
| Data/rainflow_matrix_*.npz | dT_edges_K, Tjm_edges_C, counts_50Hz, damage_50Hz, counts_mission, damage_mission |
| Data/monte_carlo_lifetime_years.parquet | Monte Carlo samples |
| Data/lifetime_models.csv | lifetime, B10, 50 Hz share per model and device |
| Data/per_second.parquet | only with save_per_second=True |
| Figures/01_mission_profile.png ... 08_lifetime_models.png | profile, temperatures, one period, thermal impedance, rainflow matrices, damage, reliability, model comparison |
