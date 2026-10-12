# 09 - Time scales and capacitances

Thermal time constants of the network [s]: 7.5e-06, 4.4e-05, 0.0001, 0.00022, 0.00072, 0.00098, 0.0023, 0.0083, 0.015, 0.074, 0.11, 35. At a 1-s step the chip modes (<= 0.1 s) are already settled, so dropping their capacitance changes nothing; the heat sink (about 34 s here) must keep its capacitance: without it the mean temperature jumps with every cloud edge (errors up to about 60 K here) and the predicted lifetime roughly halves.

**Variants (7 days)**

| variant | IGBT_life_y | Diode_life_y | max_|Tj_mean - exact|_K | IGBT_mission_cycles | IGBT_share_50Hz |
|---|---|---|---|---|---|
| exact (all capacitances) | 14.62 | 54.22 | 0 | 785 | 0.2996 |
| chip quasi-static (tau < 1 s) | 14.62 | 54.23 | 0.09355 | 785 | 0.2996 |
| only heat sink dynamic (tau < 10 s) | 14.62 | 54.23 | 0.09355 | 785 | 0.2996 |
| pure Rth, no capacitance | 7.147 | 19.15 | 59.56 | 815 | 0.1535 |

