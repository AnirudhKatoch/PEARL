# Methodology of PEARL v3

PEARL never simulates the switching or the 50 Hz waveform in time. It uses three facts:
(1) the thermal network is linear, (2) within one mission-profile second the losses repeat the same 20 ms
period, and (3) the chip, the heat sink and the mission profile live on very different time scales.

## 1. Operating points (mother_function step 1)

For every time step k: `Is` (RMS phase current per device = S / (3 Vs N_parallel) for three-phase),
`phi = atan2(Q, P)` (angle by which the phase voltage leads the phase current; P < 0 gives |phi| > 90 deg,
i.e. charging/rectifier operation), `V_dc`, and the modulation amplitude

    m_a = sqrt(2) Vs / (V_dc / 2)     (three-phase and half bridge)      m_a = sqrt(2) Vs / V_dc   (full bridge)

Seconds whose (Is, phi, V_dc, m_a) agree within the bin size (0.02 A, 1 mrad, 0.5 V, 1e-4) share one
"operating-point bin"; every bin is computed once.

## 2. Losses over one period (Electrical_model, Calculation_functions.loss_waveforms_physical)

Upper IGBT and upper diode of one leg, switching-period averaged, with phase current i(t) = sqrt(2) Is sin(wt)
and duty cycle d(t) of the upper switch (SVPWM: min/max zero-sequence added for three-phase):

    IGBT  (i > 0):  p_I(t) = f_sw * 0.5 * V_dc * i * (t_on + t_off)       + d(t) * (V0_I * i + R_I * i^2)
    Diode (i < 0):  p_D(t) = f_sw * Err * (|i| / I_ref) * (V_dc / V_ref)  + d(t) * (V0_D * |i| + R_D * i^2)

Their period averages equal the textbook average-loss formulas (checked in tests and example 04).

## 3. Thermal network (Thermal_model.ThermalNetwork)

Foster junction-to-case networks (datasheet r_i, tau_i) for IGBT and diode, each adding its rise to the case
temperature; case and heat sink form a chain driven by P_I + P_D and the ambient. The network is written as
independent first-order modes (generalised eigenproblem G v = mu C v), giving

    dz_m/dt = lambda_m z_m + B_m u,     y = W_out z,     u = [P_I, P_D, T_amb]

- Frequency response (thermal impedance): `H(jw) = sum_m W_out[:,m] B[m,:] / (jw - lambda_m)`. H(0) = Rth.
- Exact step of length dt for inputs constant within the step: `z[k+1] = a z[k] + g w[k]`, `a = exp(lambda dt)`,
  `g = (a - 1) / lambda`. The exact average over the step is used as the temperature of that second.

## 4. 50 Hz ripple (Thermal_model.ripple_from_losses)

    Tj(t) = T_amb + Rth P_mean + sum_{n>=1} Re{ 2 P_n Zth(j n w0) exp(j n w0 t) }

The ripple of each bin = inverse FFT of (FFT of the loss period x H at the harmonic frequencies), mean removed.
Np = 40 samples per period is converged (about 1 % vs 4000 samples).

## 5. 50 Hz cycles (Calculation_functions.periodic_cycles)

Rainflow counting of one closed period, vectorised over all bins: repeatedly remove the neighbouring pair of
turning points with the smallest range (each pair = one full cycle; the last pair = the max/min cycle).
Every second then has f0 identical cycles with dTj = range (scaled by the electro-thermal factor),
Tjm = Tj_mean(second) + cycle offset, and t_on = heating time of the cycle.

## 6. Mission-profile cycles (Calculation_functions.slow_cycle_damage)

Rainflow (ASTM E1049) on the 1-s mean junction temperature (turning points found with numpy, optional
hysteresis gate of 0.05 K). A slow cycle in the real signal runs from a ripple trough at its cold end to a
ripple peak at its hot end and absorbs one 50 Hz cycle there (rainflow superposition), so

    damage(slow) = max(0, 1/Nf(merged cycle) - 1/Nf(absorbed 50 Hz cycle)) * count

## 7. Damage and lifetime

Each cycle: damage 1/Nf(dTj, Tjm, t_on) from the selected lifetime model (see LIFETIME_MODELS.md).
Miner: D = sum of all damages in the mission; lifetime = mission time / D.

## 8. Electro-thermal coupling (optional)

Losses scale with (1 + kappa (Tj - T_ref)). Fixed-point iteration (2 to 4 passes) over the mean temperature;
the ripple of each second is scaled by the same factor (exact when conduction and switching losses share kappa).

## 9. Monte Carlo and system reliability

All cycles are stored compactly as cells (0.25 K x 1 K x 1/4 decade of t_on, count-weighted means; the absorbed
50 Hz cycles as negative counts). Each sample draws one set of factors for the whole mission: dTj x (1+e_dT),
Tjm + e_T, coefficient x (1+e_A), slope exponent x (1+e_alpha), and recounts the damage; a calibration factor
removes the small cell-rounding error. Two-parameter Weibull fit -> B1, B10, B50. Switch = min(IGBT, diode)
per sample. Inverter = n identical switches in series (6 x N_parallel for three-phase):
same Weibull shape, scale x n^(-1/shape).

## 10. Assumptions

Linear thermal network; ripple settled immediately after a change of operating point (error about 1 to 2 %);
all switches of the inverter see the same stress (symmetric operation); losses from 25 degC data unless kappa
is set; Foster data describe junction-to-case with the case as reference. See REVIEW_AND_CHANGELOG.md.
