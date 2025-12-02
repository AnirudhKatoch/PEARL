import numexpr as ne
from scipy.optimize import brentq

def residual_deltaT_MC(deltaT, Nf_target, Tmean, thermal_cycle_period,
                    A0, A1, T0_K, lambda_K, alpha, Ea_J, kB_J_per_K,
                    C, gamma, k_thickness):

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

    return Nf - Nf_target


Nf_target = 1e6      # cycles to failure you want to match
Tmean = 350          # in Kelvin
thermal_cycle_period = 0.01  # fixed per paper

# Example parameters
A0 = 3.43e14
A1 = 1.94
T0_K = 350
lambda_K = 70
alpha = -4.9
Ea_J = 0.066 * 1.602e-19  # eV -> J
kB_J_per_K = 1.38e-23
C = 1.434
gamma = -1.208
k_thickness = 1.0

deltaT_solution = brentq(residual_deltaT_MC,1.0, 150.0, args=(Nf_target, Tmean, thermal_cycle_period,A0, A1, T0_K, lambda_K, alpha, Ea_J, kB_J_per_K,C, gamma, k_thickness))

print("Equivalent ΔT* =", deltaT_solution)
