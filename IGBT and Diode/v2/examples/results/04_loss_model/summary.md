# 04 - Loss model

v3 computes losses at every instant from the real phase current and the duty cycle; its period average matches the textbook average formulas (exactly for sine PWM with m <= 1; this operating point needs m = 1.08, so v3 uses SVPWM and differs slightly from the sine formula). The v1 per-time-step formulas give about half the IGBT loss.

**Mean losses at 0, 90 and 180 degrees [W]**

| phi_deg | IGBT_v3_W | IGBT_textbook_W | IGBT_v1_legacy_W | Diode_v3_W | Diode_textbook_W | Diode_v1_legacy_W |
|---|---|---|---|---|---|---|
| 0 | 95.75 | 94.02 | 48.23 | 10.68 | 12.55 | 0.6199 |
| 90 | 65.43 | 65.43 | 16.47 | 44.04 | 44.04 | 11.61 |
| 180 | 35.1 | 36.83 | 4.303 | 77.41 | 75.54 | 6.326 |

