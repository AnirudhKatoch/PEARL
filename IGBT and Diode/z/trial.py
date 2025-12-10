
import numpy as np
import numexpr as ne
from pathlib import Path
import rainflow
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed to register 3D projection

def cycles_to_failure_lesit(deltaT,  # ΔT_j   : array or scalar
                            Tmean,  # T_jm   : array or scalar (K)
                            thermal_cycle_period,  # : array or scalar (s)
                            A0,
                            A1,
                            T0_K,
                            lambda_K,  # T0_K, λ
                            alpha,
                            Ea_J,
                            kB_J_per_K,  # activation energy, Boltzmann
                            C,
                            gamma,
                            k_thickness):  # k_thickness for IGBT or diode

    # Make sure inputs are float64 and contiguous (good for numexpr)
    deltaT = np.ascontiguousarray(deltaT, dtype=np.float64)
    Tmean = np.ascontiguousarray(Tmean, dtype=np.float64)
    thermal_cycle_period = np.ascontiguousarray(thermal_cycle_period, dtype=np.float64)

    if np.any(Tmean <= 0):
        raise ValueError("Tmean contains 0 K or negative values, which is not physically possible.")

    # Arrhenius temperature factor: exp(Ea / (kB * Tmean))
    c_arrhenius = ne.evaluate("exp(Ea_J / (kB_J_per_K * Tmean))",
                              local_dict=dict(Ea_J=Ea_J, kB_J_per_K=kB_J_per_K, Tmean=Tmean))

    # exp_low = exp( - (ΔT - T0_K) / λ )
    exp_low = ne.evaluate("exp(-(deltaT - T0_K) / lambda_K)",
                          local_dict=dict(deltaT=deltaT, T0_K=T0_K, lambda_K=lambda_K))

    Nf = ne.evaluate(
        "A0 * (A1 ** exp_low) * "
        "(deltaT ** (alpha - exp_low)) * "
        "c_arrhenius * "
        "((C + thermal_cycle_period**gamma) / (C + 2.0**gamma)) * "
        "k_thickness",
        local_dict=dict(A0=A0, A1=A1, alpha=alpha, C=C,
                        gamma=gamma, k_thickness=k_thickness, deltaT=deltaT, exp_low=exp_low,
                        c_arrhenius=c_arrhenius, thermal_cycle_period=thermal_cycle_period))
    return Nf


A0 = 2.9e9  # Technology Coefficient
A1 = 60  # Factor of Low ΔTj Extension
T0_K = 40  # Initial Temperature for Low ΔTj Extension [K]
lambda_K = 17  # Drop Constant of Low ΔTj Extension [K]
alpha = -4.3  # Coffin-Manson Exponent
Ea_J = 4.50e-20  # Activation Energy [J]
kB_J_per_K = 1.38e-23  # Boltzmann Constant [J/K]
C = 1  # Time Coefficient
gamma = -0.75  # Time Exponent
k_thickness = 1

deltaT = np.linspace(0, 10, 100)
Tmean = np.linspace(273.15+0, 273.15+50, 100)
thermal_cycle_period = np.linspace(0.005, 1, 100)

# ---- compute Nf ----
Nf = cycles_to_failure_lesit(
    deltaT=deltaT,
    Tmean=Tmean,
    thermal_cycle_period=thermal_cycle_period,
    A0=A0,
    A1=A1,
    T0_K=T0_K,
    lambda_K=lambda_K,
    alpha=alpha,
    Ea_J=Ea_J,
    kB_J_per_K=kB_J_per_K,
    C=C,
    gamma=gamma,
    k_thickness=k_thickness
)

# ---- 3D scatter: z = thermal cycle period,
#      color = Nf ----
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
from matplotlib.colors import LogNorm
sc = ax.scatter(
    deltaT,
    Tmean,
    thermal_cycle_period,
    c=Nf,
    norm=LogNorm(),     # <<< log scale
    cmap='viridis',
    s=25
)


ax.set_xlabel(r'$\Delta T$')
ax.set_ylabel(r'$T_{\mathrm{mean}}$ [K]')
ax.set_zlabel('Thermal cycle period [s]')

cbar = plt.colorbar(sc, label=r'$N_f$')


plt.show()
