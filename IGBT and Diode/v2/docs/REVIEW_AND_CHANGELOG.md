# Review and change log

## PEARL v1 -> v2 (method and physics)

| # | Change | Why | Effect (30-90 day profile) |
|---|---|---|---|
| 1 | No time simulation at 1 ms: mean at 1 s (exact exponential steps) + 50 Hz ripple from Zth(jw) | 1 ms BDF needed ~311 h for 3 months and hundreds of GB | 17 s, 2 MB |
| 2 | Physically consistent instantaneous losses | v1 applied period-averaged formulas at every instant (losses about halved) | IGBT life 2,800 y -> 34 y |
| 3 | Datasheet Foster Zth_jc used as junction rise above case | Foster values in a Cauer chain respond too slowly (50 Hz swing about 30 % too small) | 34 y -> 21 y |
| 4 | t_on >= 40 ms for the Semikron model | AN 21-001: no further Nf increase below 40 ms; fitted on 0.04-60 s | 21 y -> 16 y |
| 5 | Steady-state start (no warm-up cycle per repetition) | cold start counted every hour dominated short case studies | removes the artificial pf = 0.4 peak |
| 6 | Modulation amplitude from Vs and V_dc, SVPWM zero sequence | 230 V AC from 600 V DC needs m_a = 1.084 | correct duty cycle |
| 7 | Rainflow superposition rule for slow cycles, vectorised periodic rainflow | consistency with full-signal rainflow | agreement within ~2 % |
| 8 | Monte Carlo on all stored cycles, Weibull, switch and inverter B10 | v1 used one equivalent cycle | system-level reliability |
| 9 | Selectable lifetime models, user curves, curve fitting | no single model fits every device | model spread made visible |

## PEARL v2 -> v3 (review findings)

| # | Finding | Fix |
|---|---|---|
| 1 | Battery charging (P < 0) was treated like discharging: v1 power flow lost the sign of P and returned a negative RMS current | phi = atan2(Q, P), Is = abs(Is); charging now loads the diode (example 07, test) |
| 2 | Diode chip-thickness factor 0.5 is AN 21-001's class for rectifier diodes; freewheeling (CAL-type) diodes are 0.65 | default 0.65 (diode lifetime about 30 % longer) |
| 3 | Ambient in degC silently accepted | error if all T_env < 150 |
| 4 | No way to load measured data | `main.py` reads a CSV (P_W, Q_var, T_env_C); example file in data/ |
| 5 | Results per model not exported | Data/lifetime_models.csv |
| 6 | No automatic checks | tests/test_pearl.py (10 tests) |
| 7 | No reproducible study set | examples/ (13 studies with pre-computed results) |

Checked and confirmed during the review: loss averages vs textbook (exact), ripple vs time simulation (< 2 %),
exact step average vs fine simulation (< 0.01 K), periodic rainflow vs the rainflow package (< 2 %), damage vs
brute force (0.98-1.00), binning (< 0.1 %), fit and user curves (exact recovery within 1-5 %), reactive-power
sign convention (Q > 0 delivered, consistent with v1).

## Known limits (not fixed, by nature or for lack of data)

1. **Lifetime model extrapolation.** 50 Hz cycles (5-30 K, a few ms) are below the test range of every model;
   models disagree by orders of magnitude there (example 06). Report the model and the 50 Hz damage share.
2. **Parameter set vs device.** The Semikron sets are for modules; the IKW50N60H3 is a discrete TO-247 part.
3. **Loss data at 25 degC.** Real losses at 100-150 degC are 20-40 % higher; set kappa (example 10: very sensitive).
4. **Err_D = Q_rr x V_ref** is an upper bound for the recovery energy; use the datasheet E_rec when available.
5. **Linear thermal model** (R, C independent of temperature); **Foster data** assume an ideal case reference.
6. **Ripple settles instantly** after a change of operating point (about 1-2 % damage error).
7. **Symmetric operation** assumed for all switches; unbalanced grids are not modelled.
8. **Only chip-solder and bond-wire fatigue.** Baseplate-solder thermal cycling (modules), capacitors, gate
   drivers etc. are not included.
9. **Memory** grows with profile length (about 2.5 GB for 3 months); split year-long profiles.

## Ideas for later versions

Temperature-dependent losses from datasheet curves (instead of one kappa); baseplate-solder thermal cycling from
the case temperature; wear-out feedback (Rth and V_CE rise with age); calibration against power-cycling tests or
measured junction temperatures; lifetime-aware control of reactive power using the fast evaluation.
