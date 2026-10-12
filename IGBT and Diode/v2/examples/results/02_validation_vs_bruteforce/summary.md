# 02 - Validation against a brute-force reference

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

