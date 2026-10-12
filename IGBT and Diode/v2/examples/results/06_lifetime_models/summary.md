# 06 - Lifetime models

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

