# PEARL v3 - IGBT and diode lifetime from mission profiles

PEARL (Power Electronics Age and Reliability) estimates the wear-out lifetime of the IGBT and
anti-parallel diode of an inverter from a mission profile (active power, reactive power and ambient
temperature, typically one value per second).

PEARL v3 computes 3 months of 1-s mission profile in about 17 s (v1 needed an estimated 311 h), writes
about 2 MB of results (v1: hundreds of GB), and agrees with a brute-force 0.05 ms time simulation within
about 2 % in damage.

## Quick start (PyCharm)

1. Unzip, then in PyCharm: **File > Open...** and select the `PEARL_v3` folder.
2. **Settings > Project > Python Interpreter > Add Interpreter > Virtualenv** (Python 3.10, 3.11 or 3.12).
3. Open `requirements.txt` and click "Install requirements" (or run `pip install -r requirements.txt` in the terminal).
4. Right-click `main.py` > **Run**. A 90-day synthetic PV profile is simulated; results appear in `Results/Simulation_1`.
5. Optional: run `tests/test_pearl.py` (10 checks, about 5 s) and `examples/run_all_examples.py` (13 studies, about 10 min).

Full instructions: [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md).

## Repository

```
PEARL_v3/
  main.py                    start here: choose the profile (CSV or synthetic) and the settings
  mother_function.py         the pipeline: operating points -> losses -> temperatures -> damage -> reliability
  Input_parameters.py        every parameter and setting (all can be overridden from main.py)
  Calculation_functions.py   the calculation library (v1 functions kept + v2/v3 functions)
  Electrical_model.py        losses over one 20 ms period per operating point
  Thermal_model.py           thermal network: Rth, Zth(jw), exact stepping, 50 Hz ripple
  Lifetime_models.py         selectable lifetime models + user-defined curves + curve fitting
  Plotting_results.py        the eight standard figures
  requirements.txt
  data/example_profile_1day.csv     the CSV format for your own mission profile
  docs/                      how to run, methodology, parameters and outputs, lifetime models, review
  examples/                  13 studies of everything the model does, with pre-computed results
  tests/test_pearl.py        10 automatic checks against independent references
```

## How it works (one paragraph)

The junction temperature is split into a mean part and a 50 Hz ripple. The mean uses the thermal
resistance and the network's time constants, stepped exactly once per second. The ripple comes from the
loss waveform of one 20 ms period, multiplied harmonic by harmonic with the thermal impedance Zth at that
frequency (Rth would overshoot by hundreds of percent, because the heat sink cannot follow 50 Hz). The
50 Hz cycles are then known directly; rainflow is only needed on the 1-s mean temperature. Damage comes
from a selectable lifetime model and Miner's rule; Monte Carlo gives B10 for device, switch and inverter.
Details: [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Headline results of the example studies (synthetic profile, 2 devices in parallel, 2 K/W heat sink)

| Result | Value | Study |
|---|---|---|
| IGBT / diode lifetime, 90 days of PV profile | 15.7 / 56 years (B10: 9.3 / 33 years) | 01 |
| Agreement with brute-force simulation | damage ratio 0.98 to 1.00 | 02 |
| Error of P(t) x Rth for the 50 Hz swing | +500 to +1,400 % | 03 |
| Same profile with PEARL v1 physics | 2,800 years (IGBT) | 05 |
| Spread between lifetime models | 0.03 to 32 years (IGBT) | 06 |
| 10-min instead of 1-s data | lifetime 2.8 times too long | 08 |
| Losses +0.3 %/K with temperature | IGBT lifetime x 0.05 | 10 |
| Run time, 90 days incl. Monte Carlo | 17 s | 13 |

All study results: [examples/results/SUMMARY.md](examples/results/SUMMARY.md).

## Read before using the numbers

Absolute lifetimes depend strongly on the loss data, the thermal data and above all the lifetime model.
Use PEARL to compare designs and operating strategies, report the lifetime model with every number, and
see [docs/REVIEW_AND_CHANGELOG.md](docs/REVIEW_AND_CHANGELOG.md) for the known limits.
