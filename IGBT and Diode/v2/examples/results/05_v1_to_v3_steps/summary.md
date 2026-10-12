# 05 - From PEARL v1 physics to PEARL v3

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

