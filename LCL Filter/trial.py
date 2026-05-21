@staticmethod
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