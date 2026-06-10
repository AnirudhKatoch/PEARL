import numpy as np


def LCL_filter_design_function(Vg_ll_RMS,
                               S_rated,
                               I_rated,
                               fsw,
                               omega_sw,
                               fo,
                               Udc_rated,
                               M_rated,
                               inverter_phases,
                               modulation_scheme,
                               print_values,
                               current_ripple_limit=0.30,
                               delta=0.20,
                               num_C_values=100):

    """

    Source: Han, Y., Yang, M., Li, H., Yang, P., Xu, L., Coelho, E.A.A. and Guerrero, J.M. (2019)
    'Modeling and stability analysis of LCL-type grid-connected inverters: a comprehensive overview',
    IEEE Access, 7, pp. 114975–115001. doi: 10.1109/ACCESS.2019.2935806.

    Design the passive components of an LCL filter for a grid-connected inverter.

    This function follows the common LCL-filter design procedure used for
    grid-connected PWM inverters. The design is based on the rated inverter
    power, grid voltage, switching frequency, current ripple limit, harmonic
    attenuation ratio, total inductance constraint, and resonance-frequency
    constraint.

    The function calculates:
    1. Maximum allowable filter capacitance C_max.
    2. Maximum allowable total inductance L_T_max.
    3. Minimum inverter-side inductance L1_min based on switching-current ripple.
    4. Candidate capacitor values C.
    5. Candidate grid-side inductance values L2 from the harmonic attenuation ratio.
    6. Candidate resonance frequencies fr.
    7. A valid LCL design satisfying:
       - L1 + L2 <= L_T_max
       - 10*fo < fr < 0.5*fsw
       - L2 > 0
       - C > 0
    8. The capacitor-series passive damping resistor R3 using the PD-3 method.

    Parameters
    ----------
    Vg_ll_RMS : float
        RMS value of the fundamental line-to-line grid voltage [V].
    S_rated : float
        Rated apparent power of the inverter [VA].
    I_rated : float
        Rated current of the inverter [A].
    fsw : float
        Inverter switching frequency [Hz].
    omega_sw : float
        Switching angular frequency [rad/s]
    fo : float
        Fundamental grid frequency [Hz].
    Udc_rated : float
        Rated DC-link voltage of the inverter [V].
    M_rated : float
        Rated modulation index [-].
    inverter_phases : int
        Number of inverter phases.
    modulation_scheme : str
        PWM modulation scheme. Supported values: (a) "spwm" : sinusoidal pulse-width modulation, (b) "svm"  : space-vector modulation for the L1 sizing equation only
    current_ripple_limit : float, optional
        Maximum allowed inverter-side switching current ripple as a fraction of rated current [-]. Typical values are 0.20 to 0.30. Default is 0.30, meaning 30% of rated current.
    delta : float, optional
        Harmonic attenuation ratio [-]. A typical initial value is 0.20. Smaller values give stronger attenuation but usually require a larger grid-side inductor L2. Default is 0.20.
    num_C_values : int, optional
        Number of capacitor candidate values tested between a small value and C_max. Increasing this value improves the
        chance of finding a valid design but increases computation slightly. Default is 100.

    Returns
    -------
    L1 : float
        Selected inverter-side filter inductance [H].
    L2 : float
        Selected grid-side filter inductance [H].
    C : float
        Selected filter capacitance [F].
    R3 : float
        Selected passive damping resistor for the PD-3 method [Ohm]. This resistor is placed in series with the filter capacitor branch.
    """

    # Choosing Capactitor
    C_max = 0.05 * S_rated / (2 * np.pi * fo * (Vg_ll_RMS ** 2))    # [F] Capacitor's max capacitance value

    # Choosing total inductance which is L1 + L2
    L_T_max = 0.10 * (Vg_ll_RMS ** 2) / (2 * np.pi * fo * S_rated)  # [H] Total inductance value of the two inductors

    # Now for choosing L1
    if inverter_phases == 1:
        if modulation_scheme == "spwm":
            r = 2 # Single-phase bipolar SPWM,  Use r = 2 for bipolar SPWM
            L1_min = Udc_rated / (current_ripple_limit * I_rated * r * fsw)
        else:
            raise ValueError("For single-phase inverter, only SPWM is currently supported.")
    elif inverter_phases == 3:
        if modulation_scheme == "spwm" or modulation_scheme == "svm":
            L1_min = ((np.sqrt(3) / 12) * (Udc_rated / (current_ripple_limit * I_rated * fsw)) * M_rated)
        else:
            raise ValueError("modulation_scheme must be 'spwm' or 'svm'.")
    else:
        raise ValueError("inverter_phases must be 1 or 3.")

    # ----------------------------------------#
    # Now choosing L2
    # ----------------------------------------#

    # Number of capacitor divisions
    C_candidates = C_max * (np.arange(1, num_C_values + 1) / num_C_values)

    # Possible L2 values
    aL_candidates = np.abs(((1 / delta) - 1)/(1 - L1_min * C_candidates * omega_sw**2))

    L2_candidates = aL_candidates * L1_min
    L_total_candidates = L1_min + L2_candidates



    # Possible Resonance frequency values
    fr_candidates = np.sqrt((L1_min + L2_candidates) /(L1_min * L2_candidates * C_candidates)) / (2 * np.pi)
    fr_min = 10 * fo
    fr_max = 0.5 * fsw

    # ----------------------------------------#
    # Candidate validity checks
    # ----------------------------------------#

    valid_L_total = np.where(
        (aL_candidates > 0) &
        (L2_candidates > 0) &
        (L_total_candidates <= L_T_max))[0]

    valid_indices = np.where(
        (aL_candidates > 0) &
        (L2_candidates > 0) &
        (L_total_candidates <= L_T_max) &
        (fr_candidates > fr_min) &
        (fr_candidates < fr_max))[0]

    # ----------------------------------------#
    # Check validity of LCL design candidates
    # ----------------------------------------#

    if len(valid_indices) > 0:
        pass

    else:
        # ----------------------------------------#
        # Check total inductance constraint
        # ----------------------------------------#

        if len(valid_L_total) == 0:
            raise ValueError(
                "\nNo valid LCL design candidate satisfies all constraints.\n"
                "\nProblem: No candidate satisfies the total inductance constraint.\n"
                "Condition failed: L1 + L2 <= L_T_max\n"
                f"Minimum L_total candidate = {np.min(L_total_candidates):.6e} H\n"
                f"Allowed L_T_max           = {L_T_max:.6e} H\n"
                "\nPossible fixes:\n"
                "- Increase Vg_ll_RMS if the rated output power remains the same.\n"
                "- Reduce S_rated if the grid voltage remains the same.\n"
                "- Increase delta, because larger delta reduces L2.\n"
                "- Reduce L1_min by allowing more current ripple, but this increases inverter-side current ripple.")

        # ----------------------------------------#
        # Check resonance only among candidates that satisfy total inductance
        # ----------------------------------------#

        fr_candidates_valid_L = fr_candidates[valid_L_total]

        valid_fr_given_L_total = np.where(
            (fr_candidates_valid_L > fr_min) &
            (fr_candidates_valid_L < fr_max))[0]

        if len(valid_fr_given_L_total) == 0:

            if np.all(fr_candidates_valid_L > fr_max):
                raise ValueError(
                    "\nNo valid LCL design candidate satisfies all constraints.\n"
                    "\nProblem: Candidates satisfy L_total, but none satisfy the resonance-frequency constraint.\n"
                    "Condition failed: 10*fo < fr < 0.5*fsw\n"
                    f"Allowed fr range = {fr_min:.2f} Hz to {fr_max:.2f} Hz\n"
                    f"Minimum feasible fr = {np.min(fr_candidates_valid_L):.2f} Hz\n"
                    f"Maximum feasible fr = {np.max(fr_candidates_valid_L):.2f} Hz\n"
                    "\nAll feasible resonance frequencies are above the maximum allowed value.\n"
                    "This means fr is too close to the switching-frequency region.\n"
                    "\nRecommended fixes:\n"
                    "- Reduce delta. This increases L2, which lowers fr.\n"
                    "- Use a larger C value, but only if L1 + L2 remains <= L_T_max.\n"
                    "- Allow a larger total inductance limit if acceptable.\n"
                    "- You will need a larger grid-side inductor L2, but the design can work if L_total remains within the limit.")

            elif np.all(fr_candidates_valid_L < fr_min):
                raise ValueError(
                    "\nNo valid LCL design candidate satisfies all constraints.\n"
                    "\nProblem: Candidates satisfy L_total, but none satisfy the resonance-frequency constraint.\n"
                    "Condition failed: 10*fo < fr < 0.5*fsw\n"
                    f"Allowed fr range = {fr_min:.2f} Hz to {fr_max:.2f} Hz\n"
                    f"Minimum feasible fr = {np.min(fr_candidates_valid_L):.2f} Hz\n"
                    f"Maximum feasible fr = {np.max(fr_candidates_valid_L):.2f} Hz\n"
                    "\nAll feasible resonance frequencies are below the minimum allowed value.\n"
                    "This means fr is too close to the fundamental-frequency/control-bandwidth region.\n"
                    "\nRecommended fixes:\n"
                    "- Increase delta. This reduces L2, which increases fr.\n"
                    "- Use a smaller C value.\n"
                    "- Reduce the allowed total inductance.\n"
                    "- Check that C is not too close to C_max.")

            else:
                raise ValueError(
                    "\nNo valid LCL design candidate satisfies all constraints.\n"
                    "\nProblem: Some feasible resonance frequencies are below the allowed band and some are above it,\n"
                    "but none fall inside the allowed resonance-frequency range.\n"
                    "Condition failed: 10*fo < fr < 0.5*fsw\n"
                    f"Allowed fr range = {fr_min:.2f} Hz to {fr_max:.2f} Hz\n"
                    f"Minimum feasible fr = {np.min(fr_candidates_valid_L):.2f} Hz\n"
                    f"Maximum feasible fr = {np.max(fr_candidates_valid_L):.2f} Hz\n"
                    "\nRecommended fixes:\n"
                    "- Increase num_C_values.\n"
                    "- Search over delta as well as C.\n"
                    "- Avoid using np.abs() on aL; instead require aL > 0.")

        else:
            # This case should normally not happen if valid_indices was empty,
            # but keep it as a safety check.
            raise ValueError(
                "\nUnexpected design-selection issue.\n"
                "Some candidates satisfy both L_total and fr constraints locally,\n"
                "but valid_indices is still empty.\n"
                "Check whether additional constraints such as aL > 0 or L2 > 0 are being applied inconsistently.")

    # ----------------------------------------#
    # Select final C and L2 from valid candidates
    # ----------------------------------------#

    C_target = C_max / 2  # Desired capacitor target: closest valid value to C_max/2

    if len(valid_indices) == 0:
        raise ValueError("No valid LCL design candidate available for selecting C and L2.")

    # Extract only valid candidates
    valid_C_values = C_candidates[valid_indices]

    # Find valid C closest to C_target
    best_local_index = np.argmin(np.abs(valid_C_values - C_target))

    # Convert local valid-array index back to original candidate-array index
    best_global_index = valid_indices[best_local_index]

    # Select final values
    C = C_candidates[best_global_index]
    L1 = L1_min
    L2 = L2_candidates[best_global_index]
    L_total = L_total_candidates[best_global_index]
    fr = fr_candidates[best_global_index]
    aL = aL_candidates[best_global_index]


    # ----------------------------------------#
    # Final safety checks
    # ----------------------------------------#

    if L2 <= 0:
        raise ValueError("Invalid L2 selected. L2 must be positive.")

    if (L1 + L2) > L_T_max:
        raise ValueError(
            "Selected LCL design is invalid: L1 + L2 exceeds L_T_max.\n"
            f"L1 + L2 = {L1 + L2:.6e} H\n"
            f"L_T_max = {L_T_max:.6e} H")

    if not (fr_min < fr < fr_max):
        raise ValueError(
            "Selected LCL design is invalid: resonance frequency is outside the allowed range.\n"
            f"fr = {fr:.2f} Hz\n"
            f"Allowed range = {fr_min:.2f} Hz to {fr_max:.2f} Hz")

    # Damping resistor in series with the filter capacitor branch

    omega_r = 2 * np.pi * fr

    R3_min = ((1 / (6 * np.pi)) * (L2 * fsw / (L1 * fr)) * (1 / (C * omega_r)))
    R3_max = 1 / (omega_sw * C)
    R3 = np.sqrt(R3_min * R3_max)

    if R3_min >= R3_max:
        raise ValueError(
            "No valid PD-3 damping resistor range.\n"
            f"R3_min = {R3_min:.6f} ohm\n"
            f"R3_max = {R3_max:.6f} ohm")

    if print_values == True:
        print("Valid LCL design selected:")
        print(f"C_target = {C_target:.6e} F")
        print(f"C        = {C:.6e} F")
        print(f"L1       = {L1:.6e} H")
        print(f"L2       = {L2:.6e} H")
        print(f"R3       = {R3:.6f} ohm")
        print(f"aL       = {aL:.6f}")
        print(f"L_total  = {L_total:.6e} H")
        print(f"L_T_max  = {L_T_max:.6e} H")
        print(f"fr       = {fr:.2f} Hz")
        print(f"fr range = {fr_min:.2f} Hz to {fr_max:.2f} Hz")

    return L1, L2, C, R3

