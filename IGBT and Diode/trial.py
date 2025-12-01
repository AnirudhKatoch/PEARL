import numpy as np
from Calculation_functions import Calculation_functions_class

# ------------------------------------
# LESIT parameters (your values)
# ------------------------------------
A0 = 2.9e9          # Technology Coefficient
A1 = 60             # Factor of Low ΔTj Extension
T0_K = 40           # Initial Temperature for Low ΔTj Extension [K]
lambda_K = 17       # Drop Constant of Low ΔTj Extension [K]
alpha = -4.3        # Coffin-Manson Exponent
Ea_J = 4.50e-20     # Activation Energy [J]
kB_J_per_K = 1.38e-23  # Boltzmann Constant [J/K]
C = 1               # Time Coefficient
gamma = -0.75       # Time Exponent
k_thickness = {"IGBT": 1, "Diode": 0.5}

calc = Calculation_functions_class()

# ------------------------------------
# Ranges
# ------------------------------------
# Resolution (you can increase if needed)
N_T   = 30   # number of samples in Tmean
N_dT  = 30   # number of samples in deltaT
N_ton = 30   # number of samples in ton

Tmean_C = np.linspace(25.0, 100.0, N_T)       # [°C]
Tmean_K = Tmean_C + 273.15                    # [K]

deltaT  = np.linspace(20.0, 60.0, N_dT)       # [K]
# If you want to enforce LESIT valid region:
# deltaT = np.clip(deltaT, 30.0, None)

ton_val = np.linspace(0.005, 1.0, N_ton)      # [s]

# Create 3D meshgrid
Tmean_grid_K, deltaT_grid, ton_grid = np.meshgrid(
    Tmean_K, deltaT, ton_val, indexing="ij"
)

# Flatten for vectorized evaluation
Tmean_flat  = Tmean_grid_K.ravel()
deltaT_flat = deltaT_grid.ravel()
ton_flat    = ton_grid.ravel()

# ------------------------------------
# Compute Nf for all combinations
# ------------------------------------
Nf_flat = calc.cycles_to_failure_lesit(
    deltaT=deltaT_flat,
    Tmean=Tmean_flat,
    thermal_cycle_period=ton_flat,
    A0=A0,
    A1=A1,
    T0_K=T0_K,
    lambda_K=lambda_K,
    alpha=alpha,
    Ea_J=Ea_J,
    kB_J_per_K=kB_J_per_K,
    C=C,
    gamma=gamma,
    k_thickness=k_thickness["IGBT"],
)

# ------------------------------------
# Range of Nf
# ------------------------------------
Nf_min = Nf_flat.min()
Nf_max = Nf_flat.max()

print("Nf range over given parameter space:")
print(f"  Nf_min = {Nf_min:.3e}")
print(f"  Nf_max = {Nf_max:.3e}")

# Optional: log10 range
print("log10(Nf) range:")
print(f"  log10(Nf_min) = {np.log10(Nf_min):.2f}")
print(f"  log10(Nf_max) = {np.log10(Nf_max):.2f}")
