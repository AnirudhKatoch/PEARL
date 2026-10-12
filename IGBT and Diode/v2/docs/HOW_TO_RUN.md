# How to run PEARL v3

## 1. What you need

- Python 3.10, 3.11 or 3.12 (developed and tested with 3.12).
- The packages in `requirements.txt`: numpy, scipy, pandas, pyarrow, numexpr, rainflow, matplotlib, pytest.
- RAM: about 2.5 GB for 3 months of 1-s data (about 8 to 9 GB for a full year: split long profiles, see section 7).
- Disk: a few MB per run (plus about 190 MB if you switch on the per-second table).

## 2. Set up in PyCharm

1. **File > Open...**, select the `PEARL_v3` folder (the folder that contains `main.py`), open it as a new project.
2. **File > Settings > Project: PEARL_v3 > Python Interpreter > Add Interpreter > Add Local Interpreter >
   Virtualenv Environment > New**, base interpreter Python 3.10 to 3.12, OK.
3. Install the packages: open `requirements.txt`, click **Install requirements** in the yellow bar, or in the
   PyCharm terminal: `pip install -r requirements.txt`.
4. Check: right-click `tests/test_pearl.py` > **Run**. All 10 tests print `PASS` (about 5 s).

All files sit in the repository root and import each other directly, so the project root must be the working
directory and on the path. PyCharm does this automatically when you open the folder as the project. The
example scripts add the root to the path themselves.

Without PyCharm: `python -m venv .venv`, activate it, `pip install -r requirements.txt`, `python main.py`.

## 3. First run

Right-click `main.py` > **Run**. With `PROFILE_CSV = None` a synthetic 90-day PV profile is simulated
(about 20 s including figures). The console prints the lifetime, the share of damage from 50 Hz cycles,
the maximum junction temperature and B10 for IGBT, diode, switch and inverter. Results go to
`Results/Simulation_N/` (N counts up with every run):

- `Data/summary.json`: all key numbers and settings
- `Data/hourly.parquet`: hourly table (temperatures, damage, power)
- `Data/rainflow_matrix_IGBT.npz`, `..._Diode.npz`: cycle counts and damage per (dTj, Tjm) cell
- `Data/monte_carlo_lifetime_years.parquet`: the Monte Carlo samples
- `Data/lifetime_models.csv`: one row per lifetime model and device
- `Figures/01...08_*.png`

## 4. Your own mission profile

Make a CSV with one row per time step and these columns (see `data/example_profile_1day.csv`):

| Column | Unit | Meaning |
|---|---|---|
| `P_W` | W | active power of the whole inverter, + = delivered to the grid, - = absorbed (battery charging) |
| `Q_var` | var | reactive power, + = delivered by the inverter, - = absorbed |
| `T_env_C` | degC | ambient temperature at the heat sink |

In `main.py` set `PROFILE_CSV = Path(__file__).parent / "data" / "your_file.csv"`. If the rows are not 1 s
apart, set `profile_dt` in `SETTINGS` (for example `profile_dt=60.0` for minute data). Gaps must be filled
before (PEARL assumes equally spaced rows).

## 5. The settings you will change most

All settings live in `Input_parameters.py` and can be passed in `SETTINGS` in `main.py` without editing that
file. The full list is in [PARAMETERS_AND_OUTPUTS.md](PARAMETERS_AND_OUTPUTS.md).

| Setting | Why you would change it |
|---|---|
| `N_parallel` | devices in parallel per switch; with 1 the 50 A part carries 100 A peak at 48.8 kVA |
| `r_sink`, `sink_mass` | another heat sink or air speed |
| `lifetime_model` | another lifetime model, your own curves, or a list of several |
| `kappa_IGBT`, `kappa_Diode` | losses that rise with temperature (fit from datasheet values at 25 and 125/150 degC) |
| `profile_dt` | time step of your data |
| `init` | `"steady"` (default), `"ambient"` (real cold start), `"periodic"` (short profile repeated over life) |
| device data (`r_I`, `tau_I`, `I_I_points`, ...) | another IGBT module; take Foster r/tau and loss data from its datasheet |

## 6. Examples and tests

- `examples/run_all_examples.py` runs all 13 studies, each in its own process (about 10 minutes), and writes
  `examples/results/SUMMARY.md`. Each study can also be run alone (right-click > Run). Pre-computed results are
  already in `examples/results/`.
- `tests/test_pearl.py`: run directly or with `pytest tests`.

## 7. Things to keep in mind

1. **Units.** Power in W and var for the whole inverter (PEARL divides by `N_parallel`), temperatures in the CSV
   in degC (PEARL converts to K; passing degC directly to `mother_function` raises an error on purpose).
2. **Impossible operating points stop the run** with a clear message: AC voltage too high for the DC voltage,
   current above the package limit, DC voltage above the IGBT rating.
3. **Junction temperature above 175 degC gives a warning.** The lifetime is then meaningless: the design fails
   thermally, not by wear-out.
4. **Absolute lifetimes are uncertain** (lifetime model choice alone: factor 10 to 500). Compare designs and
   strategies; always state the lifetime model; run several models for important conclusions.
5. **Loss data at 25 degC underestimate losses** at operating temperature; set `kappa` once you have the data.
6. **Resolution.** 1-s data captures cloud cycles; 10-min data predicts about 3 times longer life.
7. **Long profiles.** For a year at 1 s, run months separately and add their damage (Miner's rule is additive):
   lifetime_years = total_seconds / sum(damage_total) / 31,536,000.
8. **Short repeated profiles** (for example one hour repeated over life): use `init="periodic"`.

## 8. Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` | interpreter without the packages, or project root not the working directory | install `requirements.txt` into the project interpreter; open the `PEARL_v3` folder as the project |
| `T_env must be in kelvin` | ambient passed in degC to `mother_function` | add 273.15 (the CSV loader does this) |
| `AC phase RMS voltage exceeds the theoretical` / `Modulation amplitude ... exceeds` | V_dc too low for Vs | raise `V_dc_value` or lower `Vs_rms`, or use `modulation_scheme="svm"` for three-phase |
| package current limit error | too much current per device | raise `N_parallel` |
| warning: junction temperature above 175 degC | design too hot | bigger heat sink, more devices, lower switching frequency |
| `MemoryError` | profile too long | split the profile (section 7) |
