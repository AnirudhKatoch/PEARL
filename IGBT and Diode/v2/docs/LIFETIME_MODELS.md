# Lifetime models in PEARL v3

Set `lifetime_model` in `main.py` (SETTINGS) or `Input_parameters.py`. Every model returns Nf(dTj, Tjm, t_on)
(Tj_min = Tjm - dTj/2, Tj_max = Tjm + dTj/2 are derived) and applies its own t_on validity limits.

| Name | Formula | Parameters (defaults) | t_on limits |
|---|---|---|---|
| `"semikron2024"` (default) | A0 A1^b dTj^(alpha-b) exp(Ea/(kB Tjm)) (C+t_on^g)/(C+2^g) k_th, b = exp(-(dTj-T0)/lambda) | from Input_parameters (baseplate set); also `parameter_set="no_baseplate"` or `"sintered"`; k_th per device: IGBT 1.0, diode 0.65 | t_on >= 0.04 s |
| `"lesit"` | A dTj^alpha exp(Q/(R Tjm)) | A = 640, alpha = -5, Q = 78 kJ/mol | none (no t_on term) |
| `"bayerer2008"` | K dTj^b1 exp(b2/Tj_min) t_on^b3 I^b4 V^b5 D^b6 | K 9.30e14, b1 -4.416, b2 1285, b3 -0.463, b4 -0.716, b5 -0.761, b6 -0.5; I = current per bond wire [A], V = blocking voltage/100, D = wire diameter [um] (I, V, D placeholders: set them) | 1 .. 15 s |
| `"cma"` | A dTj^(-n) exp(Ea/(kB T)) [x optional time factor] | A, n, Ea_J, temperature ("Tjm", "Tj_max", "Tj_min") | optional |
| `"curves"` | your digitised curves | see below | optional |

Examples:

```python
lifetime_model = "lesit"
lifetime_model = {"model": "bayerer2008", "I_per_wire_A": 8.0, "V_blocking_V": 600, "D_wire_um": 400}
lifetime_model = {"model": "semikron2024", "parameter_set": "no_baseplate", "name": "semikron_no_bp"}
lifetime_model = ["semikron2024", "lesit", my_curves]      # several: the first is the main result
```

## Your own curves (manufacturer power-cycling curves)

Read points off the plot (dTj and Nf, per temperature) and fill a dictionary
(template: `Lifetime_models.USER_CURVES_EXAMPLE`):

```python
my_curves = {
    "model": "curves", "name": "my_module",
    "temperature": "Tj_max",               # what labels each curve: "Tj_max", "Tjm" or "Tj_min"
    "curves": {
        "IGBT": {125: {"dTj": [30, 40, 60, 80, 100], "Nf": [6e7, 1.2e7, 1.4e6, 3e5, 9e4]},
                 150: {"dTj": [30, 40, 60, 80, 100], "Nf": [3e7, 6e6, 7e5, 1.5e5, 4.5e4]}},
        "Diode": "IGBT",                    # reuse the IGBT curves, or give its own
    },
    "t_on_s": 2.0,                          # t_on the curves were measured at
    "ton_correction": {"C": 1.0, "gamma": -0.75},   # how Nf changes with t_on; None = no change
    "t_on_min": 0.04,                       # lower limit for the t_on correction
    "Ea_J": None,                           # only needed if you give ONE temperature curve
    "extrapolate_dT": "power",              # outside the dTj points: "power" (end slope) or "clamp"
    "extrapolate_T": "arrhenius",           # outside the temperatures: "arrhenius" or "clamp"
    "failure_probability": 0.05,            # what the curves represent (kept for your report)
}
```

Interpolation: log(Nf) linear in log(dTj) along each curve, linear in 1/T between curves.
Check (example 06): the Semikron formula entered as three digitised curves reproduces its lifetime within 1 %.

## Only points, no curve family

```python
from Lifetime_models import fit_cma
spec = fit_cma([(dTj_K, T_degC, Nf), ...], temperature="Tj_max")   # least squares, returns a "cma" spec
lifetime_model = spec
```

## Which model to use

1. Manufacturer curves or model for your device family: use them (`"curves"` or the matching built-in set).
2. Nothing for your device: closest published model, plus two others to show the spread.
3. Calibrate against power-cycling tests or field data when absolute numbers matter.

Watch the definitions: temperature input (Tjm vs Tj_min vs Tj_max), t_on definition and range, failure
probability of the curves (AN 21-001: 15 %). The 50 Hz cycles (5 to 30 K, a few ms) lie outside the test range
of all these models; this is where they disagree most (example 06).
