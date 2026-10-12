# Example studies

Run one study: right-click it > Run (from PyCharm) or `python 07_power_factor_sweep.py` inside this folder.
Run all: `python run_all_examples.py` (about 10 minutes); it rewrites `results/SUMMARY.md`.
Every study writes figures (.png), tables (.csv) and `summary.md` (what it shows + numbers) to `results/<study>/`.
Pre-computed results are included.

| Study | Question it answers |
|---|---|
| 01_full_run_90_days | the standard run: lifetime, B10, all eight figures, timing |
| 02_validation_vs_bruteforce | is the fast method right? (vs 0.05 ms time simulation + full rainflow) |
| 03_why_Rth_overshoots | why P(t) x Rth gives absurd 50 Hz swings; which shortcuts work |
| 04_loss_model | instantaneous losses vs textbook averages vs PEARL v1; P > 0 and P < 0 |
| 05_v1_to_v3_steps | how much each correction changes the lifetime |
| 06_lifetime_models | built-in models, user curves, fitting; how far they disagree and why |
| 07_power_factor_sweep | rated S, pf 0..1, delivering and charging, Q delivered and absorbed |
| 08_data_resolution | 1 s vs 5 s ... 10 min mission-profile data |
| 09_timescales_capacitance | which thermal capacitances matter at a 1-s step |
| 10_sensitivity | temperature-dependent losses, heat sink, devices in parallel, Zth, V_dc, f_sw |
| 11_monte_carlo_reliability | uncertainty, Weibull, B10 of device, switch and inverter |
| 12_numerics_and_initial_state | convergence of the numerical settings; cold start vs steady start |
| 13_speed_benchmark | run time vs profile length, compared with PEARL v1 |

The mission profile in all studies is synthetic (main.synthetic_pv_profile): clear, broken-cloud and overcast
days, reactive support, a daily ambient cycle. Replace it with measured data for real conclusions.
