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