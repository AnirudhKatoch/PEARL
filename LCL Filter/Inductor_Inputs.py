import numpy as np

# [A] Core datasheet    → part number 4216L1R-B
# [B] Material datasheet → Metglas Inc. 2605SA1

#------------------------------------------------------------------------------#
# System parameters for inverter side Inductor L1
#------------------------------------------------------------------------------#
L1          = 115e-6                        # [H]   target inductance
I_RMS_L1    = 1000                          # [A]   RMS current
I_peak_L1   = np.sqrt(2) * I_RMS_L1        # [A]   peak current
f           = 50                            # [Hz]  fundamental frequency
fsw         = 10000                         # [Hz]  switching frequency

#------------------------------------------------------------------------------#
# Material: Metglas 2605SA1
#------------------------------------------------------------------------------#

kw_L1        = 0.00336922369454695  # [W/kg]  from [A] Table 5 sine row
a_L1         = 1.30103359460677     # [-]     from [A] Table 5 sine row
b_L1         = 2.13595976775746     # [-]     from [A] Table 5 sine row
rho_mass_L1  = 7180                 # [kg/m³] from [B] page 1 physical table
k_L1         = kw_L1 * rho_mass_L1 # [W/m³]  derived from [A] + [B]
Bsat_L1      = 1.56                 # [T]     from [B] page 1 electromagnetic table
B_max_L1     = 0.7 * Bsat_L1       # [T]     70% safety margin
mu_r_L1      = 3000                 # [-]     from [A] Fig 12a at 10kHz

#------------------------------------------------------------------------------#
# Core geometry
#------------------------------------------------------------------------------#

mu_0    = 4 * np.pi * 1e-7  # [H/m]  permeability of free space

Ae_L1   = 1230e-6           # [m²]  from [A] Table 2
le_L1   = 0.583             # [m]   from [A] Table 2
Ve_L1   = Ae_L1 * le_L1    # [m³]  effective volume


def calculate_turns(L, I_peak, B_max, Ae):
    """
    Calculate the minimum number of turns required for an inductor.

    Physical origin
    ---------------
    Start from the definition of inductance:
        L = N * Phi / I = N * B * Ae / I

    Rearranging for B:
        B = L * I / (N * Ae)

    At peak current I = I_peak, we want B to stay below B_max:
        B_peak = L * I_peak / (N * Ae) <= B_max

    Solving for the minimum N:
        N >= L * I_peak / (B_max * Ae)

    We round UP to the nearest integer because:
        - N must be a whole number of turns
        - Rounding down would give B_peak > B_max (saturation risk)
        - Rounding up keeps B_peak safely below B_max

    Reference
    ---------
    Kazimierczuk, M.K., "High-Frequency Magnetic Components",
    2nd Ed., Wiley-IEEE Press, 2014, Chapter 2.

    Parameters
    ----------
    L      : float  target inductance            [H]
    I_peak : float  peak current                 [A]
    B_max  : float  maximum allowed flux density [T]
    Ae     : float  effective cross section area [m²]

    Returns
    -------
    N : int
        Minimum number of turns required.
        Unit: [-]
    """
    N = int(np.ceil((L * I_peak) / (B_max * Ae)))
    return N
N_L1 = calculate_turns(L=L1, I_peak=I_peak_L1, B_max=B_max_L1, Ae = Ae_L1)

def calculate_air_gap(mu_0, N, Ae, L, le, mu_r):
    """
    Calculate the required air gap length for a gapped inductor core.

    Physical origin
    ---------------
    From Ampere's law around the magnetic circuit:
        N * I = H_core * le  +  H_gap * lg
    Where:
        H_core = B / (mu_0 * mu_r)
        H_gap  = B / mu_0
    Using B = L * I / (N * Ae) and solving for lg:
        lg = (mu_0 * N² * Ae / L) - (le / mu_r)

    Reference
    ---------
    Kazimierczuk, M.K., "High-Frequency Magnetic Components",
    2nd Ed., Wiley-IEEE Press, 2014, Chapter 2.

    Parameters
    ----------
    mu_0 : float  permeability of free space  [H/m]
    N    : int    number of turns             [-]
    Ae   : float  effective cross section     [m²]
    L    : float  target inductance           [H]
    le   : float  magnetic path length        [m]
    mu_r : float  relative permeability       [-]

    Returns
    -------
    lg : float
        Required air gap length.
        Unit: [m]
    """
    lg = (mu_0 * N**2 * Ae / L) - (le / mu_r)
    return lg
lg_L1 = calculate_air_gap(mu_0=mu_0, N=N_L1, Ae=Ae_L1, L=L1, le=le_L1, mu_r=mu_r_L1)

def calculate_B_peak(mu_0, N, I_peak, lg, le, mu_r):
    """
    Calculate the peak flux density in the inductor core.

    Physical origin
    ---------------
    From Ampere's law:

        N * I = B * le / (mu_0 * mu_r)  +  B * lg / mu_0
        N * I = B * (le/mu_r + lg) / mu_0

    Solving for B at I = I_peak:

        B_peak = mu_0 * N * I_peak / (lg + le/mu_r)

    Reference
    ---------
    Kazimierczuk, M.K., "High-Frequency Magnetic Components",
    2nd Ed., Wiley-IEEE Press, 2014, Chapter 2.

    Parameters
    ----------
    mu_0   : float  permeability of free space  [H/m]
    N      : int    number of turns             [-]
    I_peak : float  peak current                [A]
    lg     : float  air gap length              [m]
    le     : float  magnetic path length        [m]
    mu_r   : float  relative permeability       [-]

    Returns
    -------
    B_peak : float
        Peak flux density in the core.
        Unit: [T]
    """
    B_peak = (mu_0 * N * I_peak) / (lg + le / mu_r)
    return B_peak
B_peak_L1 = calculate_B_peak(mu_0=mu_0, N=N_L1, I_peak=I_peak_L1, lg=lg_L1, le=le_L1, mu_r=mu_r_L1)

def safety_checks(B_peak, B_max,Bsat,lg,le):
    # ── Check 1: B_peak must be below B_max ──────────────────────────────────────
    if B_peak >= B_max:
        raise ValueError(
            f"\nSAFETY CHECK FAILED: Flux density exceeds maximum operating limit.\n"
            f"B_peak = {B_peak:.4f} T\n"
            f"B_max  = {B_max:.4f} T\n"
            f"\nRecommendation:\n"
            f"  Increase N by 1 and recalculate lg, or\n"
            f"  Increase Ae to reduce required N, or\n"
            f"  Reduce I_peak by using more parallel inductor units.")

    # ── Check 2: B_peak must be below Bsat ───────────────────────────────────────
    if B_peak >= Bsat:
        raise ValueError(
            f"\nSAFETY CHECK FAILED: Flux density exceeds saturation limit.\n"
            f"B_peak = {B_peak:.4f} T\n"
            f"Bsat   = {Bsat:.4f} T\n"
            f"\nRecommendation:\n"
            f"  Core will saturate and inductance will collapse.\n"
            f"  Increase N or increase Ae immediately.")

    # ── Check 3: lg must be positive ─────────────────────────────────────────────
    if lg <= 0:
        raise ValueError(
            f"\nSAFETY CHECK FAILED: Air gap is zero or negative.\n"
            f"lg = {lg * 1000:.2f} mm\n"
            f"\nRecommendation:\n"
            f"  Core is too large for the required inductance.\n"
            f"  Reduce Ae or reduce N.")


    # ── Check 4: lg must be less than le ─────────────────────────────────────────
    if lg >= le:
        raise ValueError(
            f"\nSAFETY CHECK FAILED: Air gap is larger than magnetic path length.\n"
            f"lg = {lg * 1000:.2f} mm\n"
            f"le = {le * 1000:.2f} mm\n"
            f"\nRecommendation:\n"
            f"  This core is far too small for this current level.\n"
            f"  Increase Ae significantly, or\n"
            f"  Use multiple cores in parallel, or\n"
            f"  Use a custom larger core.")

    # ── Check 5: lg/le ratio warning ─────────────────────────────────────────────
    lg_le_ratio = lg / le
    if lg_le_ratio > 0.10:
        #print(
        #    f"\nWARNING CHECK 5: Air gap ratio lg/le = {lg_le_ratio * 100:.1f}%\n"
        #    f"  Recommended maximum is 10%.\n"
        #    f"  Large air gap causes fringing flux which increases losses.\n"
        #    f"  Recommendation: Increase Ae to reduce required air gap.")
        None
safety_checks(B_peak=B_peak_L1, B_max=B_max_L1, Bsat=Bsat_L1, lg=lg_L1, le=le_L1)


def calculate_l_turn(Ae):
    """
    Estimate mean length of one turn assuming square core cross section.

    Parameters
    ----------
    Ae : float  effective cross sectional area  [m²]

    Returns
    -------
    l_turn : float  mean length of one turn  [m]
    """
    side   = np.sqrt(Ae)
    l_turn = 4 * side
    return l_turn


def calculate_Rdc(rho_copper, N, l_turn, A_wire):
    """
    Calculate DC winding resistance.

    Parameters
    ----------
    rho_copper : float  copper resistivity  [Ω·m]
    N          : int    number of turns     [-]
    l_turn     : float  mean turn length    [m]
    A_wire     : float  wire cross section  [m²]

    Returns
    -------
    Rdc : float  DC winding resistance  [Ω]
    """
    Rdc = (rho_copper * N * l_turn) / A_wire
    return Rdc

# Check all this values

rho_copper = 1.72e-8                                              # [Ω·m]
J_max      = 4e6                                                  # [A/m²]
A_wire_L1  = I_RMS_L1 / J_max                                    # [m²]
l_turn_L1  = calculate_l_turn(Ae=Ae_L1)                          # [m]
Rdc_L1     = calculate_Rdc(rho_copper=rho_copper, N=N_L1,
                            l_turn=l_turn_L1, A_wire=A_wire_L1)  # [Ω]
