# 13 - Speed

v1 time is extrapolated from a measured 10-s run of the old BDF solver (thermal step only, without the 1-ms rainflow). v3 time includes losses, thermal, damage and a 5000-sample Monte Carlo, no plots.

**Timing**

| days | seconds_of_profile | PEARL_v3_s | PEARL_v1_estimate_h | speed_up | operating_bins |
|---|---|---|---|---|---|
| 1 | 8.64e+04 | 1.179 | 3.456 | 1.055e+04 | 1.332e+04 |
| 7 | 6.048e+05 | 2.713 | 24.19 | 3.21e+04 | 4.32e+04 |
| 30 | 2.592e+06 | 6.726 | 103.7 | 5.549e+04 | 1.68e+05 |
| 90 | 7.776e+06 | 17.05 | 311 | 6.569e+04 | 3.376e+05 |

