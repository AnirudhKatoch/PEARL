import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

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
# Core Material Parameters
#------------------------------------------------------------------------------#

kw_L1        = 0.00336922369454695  # [W/kg]  from [A] Table 5 sine row # We might need to use Sawtooth/Trapezoidal 50% duty, if there is too much error
a_L1         = 1.30103359460677     # [-]     from [A] Table 5 sine row # We might need to use Sawtooth/Trapezoidal 50% duty, if there is too much error
b_L1         = 2.13595976775746     # [-]     from [A] Table 5 sine row # We might need to use Sawtooth/Trapezoidal 50% duty, if there is too much error
rho_mass_L1  = 7180                 # [kg/m³] from [B] page 1 physical table
k_L1         = kw_L1 * rho_mass_L1  # [W/m³]  derived from [A] + [B]
Bsat_L1      = 1.56                 # [T]     from [B] page 1 electromagnetic table
B_max_L1     = 0.7 * Bsat_L1        # [T]     70% safety margin
mu_r_L1      = 3000                 # [-]     from [A] Fig 12a at 10kHz

#------------------------------------------------------------------------------#
# Core Geometry Parameters
#------------------------------------------------------------------------------#

# Core dimensions from datasheet Table 1 [m]
A_core_L1 = 180e-3  # [m]  Overall width of the core; outer horizontal dimension; from [A] Table 1
B_core_L1 = 240e-3  # [m]  Overall height of the core; outer vertical dimension; from [A] Table 1
D_core_L1 = 30e-3   # [m]  Depth of the core (cast width); the dimension going into the page; from [A] Table 1
F_core_L1 = 80e-3   # [m]  Width of the core window; the inner horizontal opening through which the winding passes; from [A] Table 1
G_core_L1 = 140e-3  # [m]  Height of the core window; the inner vertical opening through which the winding passes; from [A] Table 1

# Surface area of rectangular toroidal core
A_surface_L1 = ((2 * (A_core_L1 + B_core_L1)  ) * D_core_L1 + (2 * (F_core_L1 + G_core_L1) ) * D_core_L1 + 2 * (A_core_L1*B_core_L1 - F_core_L1*G_core_L1   )) # [m²]  Two end faces (front and back); total exposed core surface area available for convective cooling

mu_0    = 4 * np.pi * 1e-7  # [H/m]  Permeability of free space (physical constant)
Ae_L1   = 1230e-6           # [m²]  Effective cross-sectional area of the core; the area through which the magnetic flux passes; from [A] Table 2
le_L1   = 0.583             # [m]   Effective magnetic path length; the average distance the flux travels around the core loop; from [A] Table 2
Ve_L1   = Ae_L1 * le_L1     # [m³]  Effective core volume; used to compute total core losses from volumetric loss density

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
N_L1 = calculate_turns(L=L1, I_peak=I_peak_L1, B_max=B_max_L1, Ae = Ae_L1)                         # [-] Minimum number of turns required.

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
lg_L1 = calculate_air_gap(mu_0=mu_0, N=N_L1, Ae=Ae_L1, L=L1, le=le_L1, mu_r=mu_r_L1)                # [m] Required air gap length.

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
B_peak_L1 = calculate_B_peak(mu_0=mu_0, N=N_L1, I_peak=I_peak_L1, lg=lg_L1, le=le_L1, mu_r=mu_r_L1) # [T] Peak flux density in the core.

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

#------------------------------------------------------------------------------#
# Winding Parameters
#------------------------------------------------------------------------------#

rho        = 1.72e-8          # [ohm·m]  Electrical resistivity of copper at 20°C (material constant)
J_max      = 4e6              # [A/m²] Maximum allowed current density in the copper winding; 4 A/mm² is typical for forced-air cooling; sets wire cross-section:
A_wire_L1  = I_RMS_L1 / J_max # [m²]  Minimum wire cross-sectional area required to keep current density within J_max;

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
l_turn_L1  = calculate_l_turn(Ae=Ae_L1)  # [m]   Mean length of one copper turn around the core;

def calculate_Rdc(rho, N, l_turn, A_wire):
    """
    Calculate DC winding resistance.

    Parameters
    ----------
    rho : float  copper resistivity  [ohm·m]
    N          : int    number of turns     [-]
    l_turn     : float  mean turn length    [m]
    A_wire     : float  wire cross section  [m²]

    Returns
    -------
    Rdc : float  DC winding resistance  [ohm]
    """
    Rdc = (rho * N * l_turn) / A_wire
    return Rdc
Rdc_L1 = calculate_Rdc(rho=rho, N=N_L1, l_turn=l_turn_L1, A_wire=A_wire_L1)  # [ohm] float  DC winding resistance

#------------------------------------------------------------------------------#
# Power Losses Calculations
#------------------------------------------------------------------------------#

df = pd.read_parquet('Figures/L1_signals.parquet')
t  = df['t'].to_numpy(); V_L1 = df['V_L1'].to_numpy(); I_L1 = df['I_L1'].to_numpy()

def compute_I_L1_peak_per_harmonic_for_inductor(I_L1, f, resolution_per_cycle, Profile_size):

    """
    Decompose the inductor current I_L1 into peak amplitudes per harmonic order for each second of the mission profile.

    Unlike the capacitor function which targets specific harmonics, this function includes ALL harmonics from order 1
    up to the Nyquist limit. This is required for inductor core loss calculation via Steinmetz's equation (Eq. 8), where
    the f^alpha term amplifies contributions from higher harmonics, making cherry-picking insufficient. The paper
    (Martin-Arroyo et al., ICREPQ 2022) showed that using only 511 harmonics gives 73.6% error, while 5000 harmonics gives 0.6% error.

    Note: Returns PEAK amplitudes (not RMS) because the Steinmetz equation requires peak flux density B_j = (mu_0 * N * I_j_peak) / (lg + le/mu_r).

    Parameters
    ----------
    I_L1 : np.ndarray
        Time-domain inductor current signal [A]
    f : float
        Fundamental frequency [Hz]
    resolution_per_cycle : int
        Number of discrete simulation samples per fundamental cycle
    Profile_size : int
        Number of seconds in the mission profile

    Returns
    -------
    I_L1_peak_harmonics : np.ndarray
        Shape: (Profile_size, N//2)
        I_L1_peak_harmonics[i, j] = peak current amplitude [A] at harmonic order (j+1) during mission-profile second i
    harmonic_orders : np.ndarray
        Shape: (N//2,)
        Harmonic order indices [1, 2, 3, ..., N//2]; multiply by f to get frequency in Hz
    harmonic_freqs : np.ndarray
        Shape: (N//2,)
        Frequency [Hz] corresponding to each harmonic order; harmonic_freqs = harmonic_orders * f
    """

    samples_per_second = int(resolution_per_cycle * f)  # Number of samples in 1 second
    N = samples_per_second

    # ----------------------------------------#
    # All harmonic orders from 1 to Nyquist
    # ----------------------------------------#
    # Include every harmonic the simulation can resolve.
    # Maximum resolvable harmonic order = N//2 (Nyquist limit)
    # Maximum resolvable frequency      = N//2 * f  [Hz]

    harmonic_orders = np.arange(1, N // 2 + 1)           # [1, 2, 3, ..., N//2]
    harmonic_freqs  = harmonic_orders * f                 # [Hz]

    # ----------------------------------------#
    # Reshape into (Profile_size, N) matrix
    # ----------------------------------------#
    # Each row = one second of time-domain data
    I_matrix = I_L1.reshape(Profile_size, N)             # Shape: (Profile_size, N)

    # ----------------------------------------#
    # FFT for all seconds simultaneously
    # ----------------------------------------#
    fft_vals = np.fft.rfft(I_matrix, axis=1)             # Shape: (Profile_size, N//2 + 1)
    # ----------------------------------------#
    # Extract peak amplitudes at all harmonics
    # ----------------------------------------#
    # bin index j corresponds to harmonic order j (1-indexed), so slice from index 1 onward
    # fft_vals[:, 0]      = DC component  → discard
    # fft_vals[:, 1]      = fundamental (harmonic order 1)
    # fft_vals[:, N//2]   = Nyquist component (harmonic order N//2)

    fft_harmonic_vals  = fft_vals[:, 1 : N // 2 + 1]    # Shape: (Profile_size, N//2); drop DC bin

    I_L1_peak_harmonics = (2 * np.abs(fft_harmonic_vals)) / N  # Peak amplitude [A]
    # Factor of 2: rfft only returns positive frequencies; negative-frequency mirror has equal amplitude
    # Divide by N: numpy FFT is unnormalised; dividing by N converts to physical amplitude

    return I_L1_peak_harmonics, harmonic_orders, harmonic_freqs
I_L1_peak_harmonics, harmonic_orders_L1, harmonic_freqs_L1  = compute_I_L1_peak_per_harmonic_for_inductor(I_L1=I_L1, f=50, resolution_per_cycle=3000, Profile_size=2)

def calculate_inductor_core_losses(I_peak_harmonics, harmonic_freqs, mu_0, N, lg, le, mu_r, k, a, b, Ve):
    """
    Compute the total core loss in the inductor L1 for each second of the mission profile using Steinmetz's equation
     applied to all harmonics of the inductor current (Eq. 8, Martin-Arroyo et al., ICREPQ 2022).

    Parameters
    ----------
    I_peak_harmonics : np.ndarray, shape (Profile_size, N//2)
        Peak current amplitude [A] at each harmonic order for each second
        of the mission profile. Row i = second i, column j = harmonic j+1.
    harmonic_freqs : np.ndarray, shape (N//2,)
        Frequency [Hz] of each harmonic order.
    mu_0 : float
        Permeability of free space [H/m]. Physical constant = 4π × 10⁻⁷.
    N : int
        Number of turns in the winding [-].
    lg : float
        Air gap length [m].
    le : float
        Effective magnetic path length of the core [m].
    mu_r : float
        Relative permeability of the core material [-].
    k : float
        Steinmetz loss coefficient [W/m³].
    a : float
        Steinmetz frequency exponent alpha [-].
    b : float
        Steinmetz flux density exponent beta [-].
    Ve : float
        Effective core volume [m³].

    Returns
    -------
    P_c : np.ndarray, shape (Profile_size,)
        Total core loss [W] for each second of the mission profile.
    P_c_matrix : np.ndarray, shape (Profile_size, N//2)
        Core loss contribution [W] per second per harmonic.
    """

    # ----------------------------------------#
    # Step 1 — Peak flux density per harmonic
    # ----------------------------------------#
    # Ampere's law for a gapped core:
    #   B_j = (mu_0 * N * I_j_peak) / (lg + le/mu_r)
    # Shape: (Profile_size, N//2)
    B_j = (mu_0 * N * I_peak_harmonics) / (lg + le / mu_r)

    # ----------------------------------------#
    # Step 2 — Core loss per harmonic
    # ----------------------------------------#
    # Steinmetz equation per harmonic:
    #   P_c_j = k * f_j^alpha * B_j^beta * Ve
    # harmonic_freqs shape (N//2,) broadcasts across (Profile_size, N//2)
    # Shape: (Profile_size, N//2)
    P_c_matrix = k * (harmonic_freqs ** a) * (B_j ** b) * Ve

    # ----------------------------------------#
    # Step 3 — Sum over all harmonics
    # ----------------------------------------#
    # Shape: (Profile_size,)
    P_c = np.sum(P_c_matrix, axis=1)

    return P_c, P_c_matrix
Power_loss_L1, _ = calculate_inductor_core_losses(I_peak_harmonics=I_L1_peak_harmonics, harmonic_freqs=harmonic_freqs_L1, mu_0=mu_0, N=N_L1, lg=lg_L1, le=le_L1, mu_r=mu_r_L1, k=k_L1, a=a_L1, b=b_L1, Ve=Ve_L1)

#------------------------------------------------------------------------------#
# Temperature Calculations
#------------------------------------------------------------------------------#

Profile_size = 2
T_amb = np.full(Profile_size, 273+25)

def calculate_inductor_thermal_resistance(method, R_th_user=None, A_surface=None, heat_transfer_coefficient=None, Ve_L1=None):

    """
    Calculate or supply the thermal resistance of the inductor core to ambient.

    Three methods are supported:

    Method 1 — "user":
        User provides R_th directly from a datasheet or measurement.

    Method 2 — "surface_area":
        R_th is computed from the core surface area and convective heat
        transfer coefficient:
            R_th = 1 / (h * A_surface)
        Reference: Incropera, F.P., DeWitt, D.P., Bergman, T.L., Lavine, A.S.,
                   "Fundamentals of Heat and Mass Transfer",
                   7th Ed., Wiley, 2011, Table 1.1

    Method 3 — "empirical":
        R_th is estimated from core volume using the Kazimierczuk empirical
        formula for naturally cooled magnetic cores:
            R_th = 14.5 / Ve^0.37   [K/W], Ve in cm³
        Reference: Kazimierczuk, M.K., "High-Frequency Magnetic Components",
                   2nd Ed., Wiley-IEEE Press, 2014, Chapter 1, Eq. (1.186)

    Parameters
    ----------
    method : str
        Calculation method. One of: "user", "surface_area", "empirical".

    R_th_user : float, optional
        User-supplied thermal resistance [K/W].
        Required when method = "user".

    A_surface : float, optional
        Total exposed surface area of the core [m²].
        Required when method = "surface_area".

    heat_transfer_coefficient : float, optional
        Convective heat transfer coefficient [W/(m²·K)].
        Required when method = "surface_area".
        Typical values:
            10  W/(m²·K) — natural convection, still air
            50  W/(m²·K) — moderate forced air cooling
            250 W/(m²·K) — high-velocity forced air
        Source: Incropera et al., Table 1.1

    Ve_L1 : float, optional
        Effective core volume [m³].
        Required when method = "empirical".

    Returns
    -------
    R_th : float
        Thermal resistance from core to ambient [K/W].
    """

    if method == "user":
        if R_th_user is None:
            raise ValueError("method='user' requires R_th_user to be provided.")
        R_th = R_th_user

    elif method == "surface_area":
        if A_surface is None or heat_transfer_coefficient is None:
            raise ValueError("method='surface_area' requires A_surface and heat_transfer_coefficient.")
        R_th = 1 / (heat_transfer_coefficient * A_surface)

    elif method == "empirical":
        if Ve_L1 is None:
            raise ValueError("method='empirical' requires Ve_cm3 to be provided.")
        Ve_cm3 = Ve_L1 * 1e6  # [cm³]
        R_th = 14.5 / (Ve_cm3 ** 0.37)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: 'user', 'surface_area', 'empirical'.")

    return R_th
R_th_L1 = calculate_inductor_thermal_resistance(method="surface_area", A_surface=A_surface_L1, heat_transfer_coefficient=10) # Thermal resistance from core to ambient [K/W].

T_core_L1 = T_amb + R_th_L1* Power_loss_L1

print(T_core_L1)