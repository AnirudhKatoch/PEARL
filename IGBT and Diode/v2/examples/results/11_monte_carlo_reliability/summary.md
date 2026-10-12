# 11 - Monte Carlo and reliability

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

