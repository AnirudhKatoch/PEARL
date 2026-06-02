import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

#------------------------------------------------------------------------------#
# Simulation parameters
#------------------------------------------------------------------------------#

Profile_size = 2
resolution_per_cycle = 3000

# [A] Core datasheet    → part number 4216L1R-B
# [B] Material datasheet → Metglas Inc. 2605SA1

#------------------------------------------------------------------------------#
# System parameters for inverter side Inductor L1
#------------------------------------------------------------------------------#

L1                = 115e-6                        # [H]   target inductance
I_RMS_rated_L1    = 1000                          # [A]   RMS current
I_peak_L1         = np.sqrt(2) * I_RMS_rated_L1   # [A]   peak current
f                 = 50                            # [Hz]  fundamental frequency

df = pd.read_parquet('Figures/L1_signals.parquet')
t  = df['t'].to_numpy()
V_L1 = df['V_L1'].to_numpy()
I_L1 = df['I_L1'].to_numpy()

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
mu_r_L1      = 3000                 # [-]     from [A] Fig 12a at 10kHz # Assumed constant for simplicity


#------------------------------------------------------------------------------#
# Core Geometry Parameters
#------------------------------------------------------------------------------#

# Core dimensions from datasheet Table 1 [m]
A_core_L1 = 180e-3 * 1.55  # [m]  Overall width of the core; outer horizontal dimension; from [A] Table 1
B_core_L1 = 240e-3 * 1.55 # [m]  Overall height of the core; outer vertical dimension; from [A] Table 1
D_core_L1 = 30e-3  * 1.55 # [m]  Depth of the core (cast width); the dimension going into the page; from [A] Table 1
E_core_L1 = 50e-3  * 1.55 # [m]  Thickness of the build; from [A] Table 1
F_core_L1 = 80e-3  * 1.55 # [m]  Width of the core window; the inner horizontal opening through which the winding passes; from [A] Table 1
G_core_L1 = 140e-3 * 1.55 # [m]  Height of the core window; the inner vertical opening through which the winding passes; from [A] Table 1

# Surface area of rectangular toroidal core
A_surface_L1 = ((2 * (A_core_L1 + B_core_L1)  ) * D_core_L1 + (2 * (F_core_L1 + G_core_L1) ) * D_core_L1 + 2 * (A_core_L1*B_core_L1 - F_core_L1*G_core_L1   )) # [m²]  Two end faces (front and back); total exposed core surface area available for convective cooling
    # Calculate Ae_L1 and le_L1 with geometry value and not directly
# Make a function to either input Ae_L1 and le_L1 ypurself or calculate it through the gemorety

mu_0    = 4 * np.pi * 1e-7  # [H/m]  Permeability of free space (physical constant)
kf_L1  = 0.82               # [-] Core Stacking factor

def calculate_Ae(method, Ae_user=None, kf=None, D_core=None, E_core=None):
    """
    Calculate or supply the effective cross-sectional area of the inductor core.

    Two methods are supported:

    Method 1 — "user":
        User provides Ae directly from the datasheet Table 2.
        Use this when the manufacturer has already accounted for the stacking
        factor and published the effective area directly.

    Method 2 — "geometry":
        Ae is computed from core geometry and stacking factor:
            Ae = kf * D * E
        where D is the core depth, E is the build (thickness), and kf is the
        stacking factor accounting for gaps between lamination layers.
        Use this when scaling an existing core or estimating a custom core size.
        Reference: Kazimierczuk, M.K., "High-Frequency Magnetic Components",
                   2nd Ed., Wiley-IEEE Press, 2014, Chapter 1.

    Parameters
    ----------
    method : str
        Calculation method. One of: "user", "geometry".

    Ae_user : float, optional
        User-supplied effective cross-sectional area [m²].
        Required when method = "user".

    kf : float, optional
        Core stacking factor [-]; ratio of magnetic material to total cross-section.
        Accounts for gaps between lamination layers.
        For Metglas 2605SA1: kf = 0.82 from [A] Table 2.
        Required when method = "geometry".

    D_core : float, optional
        Depth of the core (cast width) [m]; one side of the cross-section.
        From datasheet Table 1.
        Required when method = "geometry".

    E_core : float, optional
        Build (thickness) of the core [m]; other side of the cross-section.
        From datasheet Table 1.
        Required when method = "geometry".

    Returns
    -------
    Ae : float
        Effective cross-sectional area of the core [m²].
    """

    if method == "user":
        if Ae_user is None:
            raise ValueError("method='user' requires Ae_user to be provided.")
        Ae = Ae_user

    elif method == "geometry":
        if kf is None or D_core is None or E_core is None:
            raise ValueError("method='geometry' requires kf, D_core, and E_core.")
        Ae = kf * D_core * E_core  # [m²]  Ae = kf × D × E

    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: 'user', 'geometry'.")

    return Ae
Ae_L1 = calculate_Ae(method = "geometry", kf=kf_L1, D_core=D_core_L1, E_core=E_core_L1) # Effective cross-sectional area of the core [m²].


def calculate_le(method, le_user=None, A_core=None, B_core=None, F_core=None, G_core=None):
    """
    Calculate or supply the effective magnetic path length of the inductor core.

    Two methods are supported:

    Method 1 — "user":
        User provides le directly from the datasheet Table 2.
        Use this when the manufacturer has published the effective magnetic
        path length directly. Preferred when available — more accurate than
        the geometry estimate because it accounts for corner rounding.

    Method 2 — "geometry":
        le is estimated from the core outer and window dimensions as the
        perimeter of the centreline rectangle through the core material:

            leg_w = (A - F) / 2          horizontal leg half-width
            leg_h = (B - G) / 2          vertical leg half-height
            le    = (A + F) + (B + G)    centreline perimeter

        Physical origin: flux travels through the midpoint of each leg,
        not along the outer or inner edge. The centreline lies halfway
        between the outer dimension and the window dimension on each side.
        This formula gives a close approximation; the small residual error
        (~10%) relative to the datasheet value comes from corner rounding
        of the wound core, which shortens the actual path slightly.

        Reference: Kazimierczuk, M.K., "High-Frequency Magnetic Components",
                   2nd Ed., Wiley-IEEE Press, 2014, Chapter 1.

    Parameters
    ----------
    method : str
        Calculation method. One of: "user", "geometry".

    le_user : float, optional
        User-supplied effective magnetic path length [m].
        Required when method = "user".

    A_core : float, optional
        Overall width of the core [m]; outer horizontal dimension.
        From datasheet Table 1. Required when method = "geometry".

    B_core : float, optional
        Overall height of the core [m]; outer vertical dimension.
        From datasheet Table 1. Required when method = "geometry".

    F_core : float, optional
        Width of the core window [m]; inner horizontal opening.
        From datasheet Table 1. Required when method = "geometry".

    G_core : float, optional
        Height of the core window [m]; inner vertical opening.
        From datasheet Table 1. Required when method = "geometry".

    Returns
    -------
    le : float
        Effective magnetic path length [m].
    """

    if method == "user":
        if le_user is None:
            raise ValueError("method='user' requires le_user to be provided.")
        le = le_user

    elif method == "geometry":
        if any(v is None for v in [A_core, B_core, F_core, G_core]):
            raise ValueError("method='geometry' requires A_core, B_core, F_core, and G_core.")

        leg_w = (A_core - F_core) / 2                # [m]  horizontal leg half-width
        leg_h = (B_core - G_core) / 2                # [m]  vertical leg half-height
        le    = 2 * (F_core + leg_w) + 2 * (G_core + leg_h)  # [m]  centreline perimeter
        # Simplified: le = (A_core + F_core) + (B_core + G_core)

    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: 'user', 'geometry'.")

    return le
le_L1 = calculate_le(method="geometry", A_core=A_core_L1, B_core=B_core_L1, F_core=F_core_L1, G_core=G_core_L1)  # [m]   Effective magnetic path length; the average distance the flux travels around the core loop

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
        raise ValueError(
            f"\nWARNING CHECK 5: Air gap ratio lg/le = {lg_le_ratio * 100:.1f}%\n"
            f"  Recommended maximum is 10%.\n"
            f"  Large air gap causes fringing flux which increases losses.\n"
            f"  Recommendation: Increase Ae to reduce required air gap.")
        None
safety_checks(B_peak=B_peak_L1, B_max=B_max_L1, Bsat=Bsat_L1, lg=lg_L1, le=le_L1)

#------------------------------------------------------------------------------#
# Winding Parameters
#------------------------------------------------------------------------------#

# Wire: Elektrisola Amidester 200 (A200), Theic-modified Polyesterimide
# Insulation standard: IEC 60317-8 / NEMA MW 74
# Thermal class: 200°C (temperature index 210°C at 20,000 h per IEC 60172)
# Source: Elektrisola Enamelled Copper Wire — Manufacturing Programme and Technical Data Page 3 (product table) and Page 4 (dimensional table) https://www.mintex.si/wp-content/uploads/2018/06/CuL-englisch.pdf

rho                = 1.709e-8        # [Ω·m]   Copper resistivity at 20°C
alpha_cu           = 0.00393         # [1/°C]  Temperature coefficient of resistivity for copper
J_max              = 4e6             # [A/m²]  Maximum allowed current density = 4 A/mm²

A_wire_L1_minimum  = I_RMS_rated_L1 / J_max   # [m²]  Minimum total copper cross-section required
d_wire_L1_minimum  = np.sqrt((4 * A_wire_L1_minimum) / np.pi)  # [m]  Equivalent minimum round wire diameter

# Winding loss model: conductor type not specified; Rac ≈ Rdc (skin and proximity effects neglected — conservative underestimate)
d_strand_wire_L1   = 0.500e-3        # [m]    Strand diameter — largest available in Elektrisola datasheet  Source: Elektrisola datasheet, page 4, nominal diameter column
A_strand_wire_L1   = 0.196350e-6     # [m²]   Bare wire cross-sectional area of one strand Source: Elektrisola datasheet, page 4, section column (mm²),

N_parallel_wire_L1 = int(np.ceil(A_wire_L1_minimum / A_strand_wire_L1))  # [-]  Number of parallel strands required
A_wire_actual_L1   = N_parallel_wire_L1 * A_strand_wire_L1  # [m²]  Actual total copper area after rounding up

# Check whether the winding physically fits inside the core window.
def check_window_fill(N_turns, N_parallel, A_wire_bare, F_core, G_core, kf_window_max):
    """
    Check whether the winding physically fits inside the core window
    using the standard window utilization factor ku defined in Kazimierczuk.

    Definition (Kazimierczuk, Chapter 10):
        ku = (N_turns * N_parallel * A_bare) / A_window

    The limits below use bare copper area — insulation, air gaps, and
    imperfect packing are already absorbed into the empirical limit values:
        ku ≤ 0.3  — hand-wound toroid, random lay
        ku ≤ 0.4  — machine-wound, random lay
        ku ≤ 0.6  — organised / orthocyclic winding

    Reference: Kazimierczuk, M.K., "High-Frequency Magnetic Components", 2nd Ed., Wiley-IEEE Press, 2014, Chapter 10.

    Parameters
    ----------
    N_turns      : int    number of winding turns              [-]
    N_parallel   : int    number of parallel strands per turn  [-]
    A_wire_bare  : float  bare copper area of one strand       [m²]
                          Use A_strand_wire_L1 (not outer area)
    F_core       : float  core window width  (inner)           [m]
    G_core       : float  core window height (inner)           [m]
    kf_window_max: float  maximum allowed ku                   [-]

    Returns
    -------
    ku : float  actual window utilization factor [-]
    """

    A_window       = F_core * G_core                           # [m²] window area
    A_copper_total = N_turns * N_parallel * A_wire_bare        # [m²] total bare copper area

    ku = A_copper_total / A_window                             # [-]  utilization factor

    if ku > kf_window_max:
        raise ValueError(
            f"\nSAFETY CHECK FAILED: Winding does not fit inside core window.\n"
            f"\n  Core window area        : {A_window * 1e6:.1f} mm²"
            f"  (F={F_core*1e3:.1f} mm × G={G_core*1e3:.1f} mm)\n"
            f"  Total bare copper area  : {A_copper_total * 1e6:.1f} mm²"
            f"  ({N_turns} turns × {N_parallel} strands"
            f" × {A_wire_bare * 1e6:.4f} mm² per strand)\n"
            f"  Actual ku               : {ku:.3f}\n"
            f"  Maximum allowed ku      : {kf_window_max:.3f}\n"
            f"\nRecommendations:\n"
            f"  (1) Increase core window: scale F_core and G_core up, or\n"
            f"  (2) Reduce N_turns: increase Ae to allow fewer turns, or\n"
            f"  (3) Switch to copper foil or busbar winding\n"
            f"      (standard practice at MW current levels), or\n"
            f"  (4) Use organised winding and set kf_window_max=0.6.")

    return ku
ku_L1 = check_window_fill(N_turns = N_L1, N_parallel = N_parallel_wire_L1, A_wire_bare = A_strand_wire_L1, F_core = F_core_L1, G_core = G_core_L1, kf_window_max= 0.5)

def calculate_l_turn(D_core, E_core):
    """
    Estimate mean length of one turn for a rectangular toroidal core.

    The copper winding wraps around one leg of the core.
    The mean turn path goes around the perimeter of the core cross-section,
    which is D (depth) × E (build/thickness).

    Physical origin:
        The winding sits at the midpoint of the core leg cross-section.
        Mean turn = perimeter of the D × E rectangle.

    Parameters
    ----------
    D_core : float  core depth (cast width)     [m]   from datasheet Table 1
    E_core : float  core build (thickness)       [m]   from datasheet Table 1

    Returns
    -------
    l_turn : float  mean length of one turn      [m]
    """
    l_turn = 2 * (D_core + E_core)   # [m]  perimeter of D × E rectangle
    return l_turn
l_turn_L1 = calculate_l_turn(D_core=D_core_L1, E_core=E_core_L1)

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
Rdc_L1 = calculate_Rdc(rho=rho, N=N_L1, l_turn=l_turn_L1, A_wire=A_wire_actual_L1)  # [ohm] float  DC winding resistance # Assumed  no Skin or Proximity Effect

#------------------------------------------------------------------------------#
# Power Losses Calculations
#------------------------------------------------------------------------------#

def compute_I_L1_peak_per_harmonic_for_inductor(I_L1, f, resolution_per_cycle, Profile_size):
    """
    Decompose the inductor current I_L1 into peak amplitudes at each harmonic of the fundamental frequency, for each
     second of the mission profile.

    Unlike the capacitor function which targets specific harmonics, this function includes ALL harmonics from order 1
    up to the Nyquist limit. This is required for inductor core loss calculation via Steinmetz's equation (Eq. 8),
    where the f^alpha term amplifies contributions from higher harmonics, making cherry-picking insufficient. The paper
    (Martin-Arroyo et al., ICREPQ 2022) showed that using only 511 harmonics gives 73.6% error, while 5000 harmonics
    gives 0.6% error.

    Note: Returns PEAK amplitudes (not RMS) because the Steinmetz equation requires peak flux density:
     B_j = (mu_0 * N * I_j_peak) / (lg + le/mu_r).

    FFT bin spacing
    ---------------
    The signal is 1 second long with N = resolution_per_cycle * f samples.
    The FFT frequency resolution is therefore:
        df = sampling_rate / N = N / N = 1 Hz per bin
    So FFT bin k corresponds to exactly k Hz.
    Harmonic order j of the fundamental (f Hz) sits at frequency j*f Hz, which is FFT bin j*f. Only these bins carry
    real signal energy. All other bins between harmonics contain only numerical noise and must be discarded — including
    them would corrupt the Steinmetz sum.

    Parameters
    ----------
    I_L1 : np.ndarray
        Time-domain inductor current signal [A].
        Length = Profile_size * resolution_per_cycle * f
    f : float
        Fundamental frequency [Hz]
    resolution_per_cycle : int
        Number of discrete simulation samples per fundamental cycle. Controls the Nyquist limit: higher values resolve more harmonics.
    Profile_size : int
        Number of seconds in the mission profile.

    Returns
    -------
    I_L1_peak_harmonics : np.ndarray, shape (Profile_size, max_harmonic_order)
        Peak current amplitude [A] at each harmonic of f, for each mission-profile second.
        I_L1_peak_harmonics[i, j] = peak amplitude at harmonic order (j+1) during second i.
    harmonic_orders : np.ndarray, shape (max_harmonic_order,)
        Harmonic order indices [1, 2, 3, ..., max_harmonic_order]. Multiply by f to get frequency in Hz.
    harmonic_freqs : np.ndarray, shape (max_harmonic_order,)
        Physical frequency [Hz] of each harmonic order.
        harmonic_freqs[j] = (j+1) * f
        Spans from f to f_Nyquist = sampling_rate / 2.
    """

    samples_per_second = int(resolution_per_cycle * f)  # [samples/s] total samples in one second; also the FFT length N
    N                  = samples_per_second

    # ----------------------------------------#
    # Harmonic orders and frequencies
    # ----------------------------------------#
    # FFT bin spacing = sampling_rate / N = N / N = 1 Hz per bin
    # → FFT bin k = k Hz exactly
    # → Harmonic order j of fundamental f sits at bin j*f
    #
    # Maximum resolvable frequency = Nyquist = N / 2 Hz
    # Maximum resolvable harmonic order = Nyquist / f = (N/2) / f
    #
    # Example: resolution_per_cycle=3000, f=50
    #   N               = 150,000 samples/s
    #   Nyquist         = 75,000 Hz
    #   max_harmonic    = 75,000 / 50 = 1,500
    #   harmonic_freqs  = [50, 100, 150, ..., 75,000] Hz

    max_harmonic_order = int((N // 2) // f)                      # highest harmonic order resolvable within Nyquist limit

    harmonic_orders    = np.arange(1, max_harmonic_order + 1)    # [1, 2, 3, ..., max_harmonic_order]
    harmonic_freqs     = harmonic_orders * f                     # [f, 2f, 3f, ..., max_harmonic_order * f] [Hz]

    # ----------------------------------------#
    # Reshape into (Profile_size, N) matrix
    # ----------------------------------------#
    # Each row = one second of time-domain data
    I_matrix = I_L1.reshape(Profile_size, N)                     # Shape: (Profile_size, N)

    # ----------------------------------------#
    # FFT for all seconds simultaneously
    # ----------------------------------------#
    # rfft returns N//2 + 1 complex bins for a real input of length N
    # Bin k corresponds to frequency k * (sampling_rate / N) = k * 1 Hz = k Hz
    fft_vals = np.fft.rfft(I_matrix, axis=1)                     # Shape: (Profile_size, N//2 + 1)

    # ----------------------------------------#
    # Extract bins at harmonic frequencies only
    # ----------------------------------------#
    # Harmonic order j sits at frequency j*f Hz = FFT bin j*f
    # We must extract ONLY these bins and discard all others.
    # Extracting all bins would include inter-harmonic noise bins,
    # which have no physical meaning but would inflate the Steinmetz sum.
    #
    # bin_indices[j] = harmonic_orders[j] * f = the FFT bin number for harmonic j
    # Example: harmonic 1 → bin 50 (50 Hz), harmonic 200 → bin 10000 (10 kHz)

    bin_indices       = (harmonic_orders * int(f)).astype(int)   # FFT bin index for each harmonic order
    fft_harmonic_vals = fft_vals[:, bin_indices]                 # Shape: (Profile_size, max_harmonic_order)

    # ----------------------------------------#
    # Convert FFT output to peak amplitudes
    # ----------------------------------------#
    # Factor of 2: rfft only returns positive-frequency bins; the negative-
    #   frequency mirror carries equal energy, so multiply by 2 to recover
    #   the full single-sided peak amplitude.
    # Divide by N: numpy's rfft is unnormalised (sum of inputs, not average),
    #   so dividing by N converts raw FFT magnitude to physical amplitude [A].
    # DC bin (bin 0) is excluded — it has no factor-of-2 correction and
    #   carries no harmonic information relevant to core losses.

    I_L1_peak_harmonics = (2 * np.abs(fft_harmonic_vals)) / N   # Shape: (Profile_size, max_harmonic_order); Peak amplitude [A]

    return I_L1_peak_harmonics, harmonic_orders, harmonic_freqs
I_L1_peak_harmonics, harmonic_orders_L1, harmonic_freqs_L1  = compute_I_L1_peak_per_harmonic_for_inductor(I_L1=I_L1, f=f, resolution_per_cycle=resolution_per_cycle, Profile_size=Profile_size)

# [W] Inductor core loss;
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
P_c_L1, _ = calculate_inductor_core_losses(I_peak_harmonics=I_L1_peak_harmonics, harmonic_freqs=harmonic_freqs_L1, mu_0=mu_0, N=N_L1, lg=lg_L1, le=le_L1, mu_r=mu_r_L1, k=k_L1, a=a_L1, b=b_L1, Ve=Ve_L1)

samples_per_second   = int(resolution_per_cycle * f)
I_matrix  = I_L1.reshape(Profile_size, samples_per_second)              # Shape: (Profile_size, samples_per_second)
I_RMS_L1  = np.sqrt(np.mean(I_matrix ** 2, axis=1))

# [W] DC copper winding losses; P_w = Rdc * I_RMS²
P_w_L1    = Rdc_L1 * I_RMS_L1 ** 2

P_total_L1 = P_c_L1 + P_w_L1   # [W] Total inductor losses

#------------------------------------------------------------------------------#
# Temperature Calculations
#------------------------------------------------------------------------------#

# Single-node model: core and winding assumed at same temperature (consistent with Martín-Arroyo et al.)

T_amb = np.full(Profile_size, 273+25)

def calculate_inductor_thermal_resistance(method, R_th_user=None, A_surface=None, heat_transfer_coefficient=None, Ve_m3 =None):

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
            500 W/(m²·K) — Liquid Cooling
        Source: Incropera et al., Table 1.1

    Ve_m3  : float, optional
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
        if Ve_m3  is None:
            raise ValueError("method='empirical' requires Ve_cm3 to be provided.")
        Ve_cm3 = Ve_m3  * 1e6  # [cm³]
        R_th = 14.5 / (Ve_cm3 ** 0.37)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: 'user', 'surface_area', 'empirical'.")

    return R_th
R_th_L1 = calculate_inductor_thermal_resistance(method="surface_area", A_surface=A_surface_L1, heat_transfer_coefficient=10) # Thermal resistance from core to ambient [K/W].
T_inductor_L1   = T_amb + R_th_L1 * P_total_L1

#------------------------------------------------------------------------------#
# Lifetime Calculations
#------------------------------------------------------------------------------#

# Temperature

# Wire: Elektrisola Amidester 200 (A200), Theic-modified Polyesterimide
# Source: Elektrisola Enamelled Copper Wire — Manufacturing Programme and
#         Technical Data, Page 3, product table, thermal values row
#         https://www.mintex.si/wp-content/uploads/2018/06/CuL-englisch.pdf

T_insulation_rated = 273 + 210         # [K]  Temperature index of Amidester 200 insulation.
                                       #  Definition: the continuous operating temperature at which the insulation reaches its reference lifetime of 20,000 hours.
                                       #  Value 210°C taken directly from Elektrisola datasheet page 3, thermal values table, column "Amidester 200", row
L_insulation_rated = 20000             # [h] Reference lifetime at T_insulation_rated.
Ea_insulation      = 1.1 * 1.602e-19   # [J] Activation energy of the thermal degradation reaction in the polyesterimide insulation.
                                       #  Source: Emery, F.T., "Arrhenius model for insulation aging", IEEE Electrical Insulation
                                       #  Magazine, widely adopted for Class 200 polyesterimide systems.
kb_insulation      = 1.381e-23         # [J/K] Boltzmann constant — physical constant relating thermal energy to temperature.

# Voltage

# Voltage endurance coefficient for polyesterimide insulation
# Source: Montanari, G.C., "Aging and lifetime of electrical insulation: challenges and perspectives", IEEE Electrical Insulation Magazine, Vol. 9, No. 5, 1993.
# Simoni, L., "General equation of the decline in the electric strength for combined thermal-electrical stresses", IEEE Trans. on Electrical Insulation, Vol. EI-19, No. 1, 1984.

# Typical range for Class 200 polyesterimide: n = 9 to 12 Conservative value n = 9 used here.
n_endurance = 9                  # [-] voltage endurance coefficient

# Insulation breakdown voltage from Elektrisola datasheet page 4 Wire: 0.500mm, Grade 1, cylinder test, minimum value
# Source: Elektrisola datasheet page 4, "Minimum breakdown voltage" column
V_bd_insulation = 2400           # [V] minimum breakdown voltage

def compute_V_turn_peak(V_L1, N_turns, resolution_per_cycle, f, Profile_size):

    """
    Compute the peak turn-to-turn voltage for each second of the mission profile.

    The voltage across one turn is approximately:
        V_turn = V_L1 / N_turns

    The insulation between two adjacent turns sees this voltage.
    Peak value is used for the electrical stress assessment.

    Parameters
    ----------
    V_L1                 : np.ndarray  instantaneous inductor voltage [V]
    N_turns              : int         number of winding turns        [-]
    resolution_per_cycle : int     samples per fundamental cycle  [-]
    f                    : float       fundamental frequency          [Hz]
    Profile_size         : int         mission profile length         [s]

    Returns
    -------
    V_turn_peak : np.ndarray, shape (Profile_size,)
        Peak turn-to-turn voltage for each profile second [V].
    """

    samples_per_second = int(resolution_per_cycle * f)
    V_matrix    = V_L1.reshape(Profile_size, samples_per_second)  # (Profile_size, N)
    V_turn_peak = np.max(np.abs(V_matrix), axis=1) / N_turns      # [V] peak per second
    return V_turn_peak
V_turn_peak_L1 = compute_V_turn_peak(V_L1 = V_L1, N_turns = N_L1, resolution_per_cycle = resolution_per_cycle, f = f, Profile_size = Profile_size)

def check_voltage_stress(V_turn_peak, V_bd):

    """
    Check that the peak turn-to-turn voltage does not approach the insulation breakdown voltage.

    Two thresholds are applied:
        V_turn / V_bd >= 1.0  → hard failure  (insulation will fail immediately)

    Parameters
    ----------
    V_turn_peak : np.ndarray  peak turn-to-turn voltage per second  [V]
    V_bd        : float       insulation breakdown voltage          [V]

    Returns
    -------
    V_stress_ratio : np.ndarray  V_turn_peak / V_bd per second [-]

    Raises
    ------
    ValueError
        If any V_stress_ratio >= 1.0
    """

    V_stress_ratio = V_turn_peak / V_bd

    if np.any(V_stress_ratio >= 1.0):
        raise ValueError(
            f"\nSAFETY CHECK FAILED: Turn-to-turn voltage exceeds breakdown voltage.\n"
            f"  Peak V_turn  = {np.max(V_turn_peak):.2f} V\n"
            f"  V_bd         = {V_bd:.2f} V\n"
            f"  Stress ratio = {np.max(V_stress_ratio):.4f}\n"
            f"\nRecommendations:\n"
            f"  (1) Increase N to reduce voltage per turn, or\n"
            f"  (2) Use Grade 2 wire (V_bd = 4600 V) or Grade 3 (V_bd = 7000 V).")

    return V_stress_ratio
V_stress_ratio_L1 = check_voltage_stress(V_turn_peak = V_turn_peak_L1, V_bd = V_bd_insulation)

# Arrhenius lifetime formula
# Standard: IEC 60216 — "Electrical insulating materials — Thermal endurance properties" — defines the application of the Arrhenius model to predict insulation lifetime from temperature.
# Reference: Montanari, G.C., "Aging and lifetime of electrical insulation: challenges and perspectives", IEEE Electrical Insulation Magazine, Vol. 9, No. 5, 1993.

def calculate_inductor_lifetime(T_operating, T_rated, L_rated, Ea, kb):

    """
    Calculate winding insulation lifetime using the Arrhenius thermal aging model.

    Parameters
    ----------
    T_operating : np.ndarray or float
        Operating temperature of the inductor [K].
        Use T_inductor_L1 from the thermal model.
    T_rated : float
        Rated temperature index of the insulation [K].
        Source: Elektrisola datasheet page 3 — 210°C = 483 K for Amidester 200.
    L_rated : float
        Reference lifetime at T_rated [h].
        Source: IEC 60172 — 20,000 h reference point.
    Ea : float
        Activation energy of insulation degradation [J].
    kb : float
        Boltzmann constant [J/K]

    Returns
    -------
    L : np.ndarray or float
        Predicted insulation lifetime [Years] at each operating temperature.
    """

    L = L_rated * np.exp((Ea / kb) * (1/T_operating - 1/T_rated))
    L = L/(365*24)
    return L
L_inductor_L1 = calculate_inductor_lifetime(T_operating = T_inductor_L1, T_rated = T_insulation_rated, L_rated = L_insulation_rated, Ea = Ea_insulation, kb = kb_insulation )   # [Years]  Predicted winding insulation lifetime at each second of the mission profile

def apply_miners_rule(L_per_second):

    """

    Apply Miner's cumulative damage rule to compute expected total lifetime.

    D_cycle  = sum( dt / L_i )          — damage per mission-profile cycle
    L_total  = Profile_duration / D     — expected lifetime [years]

    Reference: Miner, M.A., Journal of Applied Mechanics, 1945. IEC 60216-1.

    Parameters
    ----------
    L_per_second : np.ndarray  lifetime at each profile second [years]

    Returns
    -------
    L_total : float  expected total lifetime [years]

    """

    dt_profile_years = 1 / (365 * 24 * 3600)
    d_i              = dt_profile_years / L_per_second        # [-] damage per step
    D_cycle          = np.sum(d_i)                            # [-] total damage per cycle
    Profile_duration = len(L_per_second) * dt_profile_years   # [years]
    L_total          = Profile_duration / D_cycle             # [years]

    return L_total
L_total_L1 = apply_miners_rule(L_per_second = L_inductor_L1)
