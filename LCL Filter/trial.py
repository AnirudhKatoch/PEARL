@staticmethod
def compute_THD(t, Signal, Signal_ref, f, dt, resolution_per_cycle, save_path, printing, n_cycles=1, max_harmonic=None,
                plot=True, ):
    """
    Compute the THD of the grid-side current Signal over the last n_cycles
    fundamental periods, and report tracking metrics vs the reference Signal_ref.

    The analysis window is selected by integer sample count (not by a
    float-time mask), so every harmonic lands exactly on an FFT bin and
    single-bin extraction is leakage-free.

    Parameters
    ----------
    t : array                  Time vector [s]
    Signal : array               Simulated grid-side current [A]
    Signal_ref : array             Reference grid current [A]
    f : float                  Fundamental frequency [Hz]
    dt : float                 Simulation time step [s]
    resolution_per_cycle : int Samples per fundamental cycle [-]
    save_path : str            Path for the saved comparison figure
    n_cycles : int, optional   Number of trailing cycles to analyse. Default 1.
    max_harmonic : int, optional
        Highest harmonic order included in THD. Default None = up to Nyquist.
    plot : bool, optional      Save the waveform/error plot. Default True.

    Returns
    -------
    THD_percent : float        THD of Signal [%]
    """

    Signal = np.asarray(Signal)
    Signal_ref = np.asarray(Signal_ref)
    t = np.asarray(t)

    # --- exact-integer-cycle window (last n_cycles periods) ---
    spc = int(round(resolution_per_cycle))  # samples per cycle
    win = n_cycles * spc
    if win > len(Signal):
        raise ValueError(
            f"Window of {win} samples exceeds signal length {len(Signal)}.")

    last = slice(-win, None)
    Signal_w = Signal[last]
    Signal_ref_w = Signal_ref[last]
    t_w = t[last]

    # ── 1. RMS ────────────────────────────────────────────────
    Signal_RMS = np.sqrt(np.mean(Signal_w ** 2))
    Signal_ref_RMS = np.sqrt(np.mean(Signal_ref_w ** 2))

    # ── 2. Tracking error ─────────────────────────────────────
    error = Signal_w - Signal_ref_w
    error_RMS = np.sqrt(np.mean(error ** 2))
    error_peak = np.max(np.abs(error))
    NRMSE = error_RMS / Signal_ref_RMS * 100

    # ── 3. FFT (DC removed) ───────────────────────────────────
    N = len(Signal_w)
    freqs = np.fft.rfftfreq(N, d=dt)

    fft_L2 = np.fft.rfft(Signal_w - np.mean(Signal_w))
    fft_ref = np.fft.rfft(Signal_ref_w - np.mean(Signal_ref_w))

    # fundamental sits at bin = n_cycles (n_cycles periods in the window)
    idx_f = n_cycles

    # RMS amplitude of fundamental: (2|X|/N)/sqrt(2) = sqrt(2)|X|/N
    amp_L2 = np.sqrt(2) * np.abs(fft_L2[idx_f]) / N
    amp_ref = np.sqrt(2) * np.abs(fft_ref[idx_f]) / N

    phase_L2 = np.angle(fft_L2[idx_f], deg=True)
    phase_ref = np.angle(fft_ref[idx_f], deg=True)
    phase_err = (phase_L2 - phase_ref + 180) % 360 - 180  # wrap to ±180

    # ── 4. THD of Signal (harmonics 2..max up to Nyquist) ───────
    # harmonic h sits exactly on bin h*n_cycles
    nyq_order = (len(fft_L2) - 1) // n_cycles  # highest resolvable order
    top = nyq_order if max_harmonic is None else min(max_harmonic, nyq_order)
    h_orders = np.arange(2, top + 1)
    h_bins = h_orders * n_cycles

    P_harmonics = np.sum(np.abs(fft_L2[h_bins]) ** 2)
    P_fundamental = np.abs(fft_L2[idx_f]) ** 2
    THD = np.sqrt(P_harmonics / P_fundamental)
    THD_percent = THD * 100

    # ── Print summary (unchanged format) ──────────────────────
    if printing == True:
        print("=" * 46)
        print(f"  Signal_ref RMS          : {Signal_ref_RMS:>10.4f}  A")
        print(f"  Signal   RMS          : {Signal_RMS:>10.4f}  A")
        print("-" * 46)
        print(f"  Tracking error RMS  : {error_RMS:>10.4f}  A")
        print(f"  Tracking error peak : {error_peak:>10.4f}  A")
        print(f"  NRMSE               : {NRMSE:>10.4f}  %")
        print("-" * 46)
        print(f"  Fundamental amp ref : {amp_ref:>10.4f}  A (RMS)")
        print(f"  Fundamental amp L2  : {amp_L2:>10.4f}  A (RMS)")
        print(f"  Phase ref           : {phase_ref:>10.4f}  deg")
        print(f"  Phase Signal          : {phase_L2:>10.4f}  deg")
        print(f"  Phase error         : {phase_err:>10.4f}  deg")
        print("-" * 46)
        print(f"  THD of Signal         : {THD_percent:>10.4f}  %")
        print("=" * 46)

    # ── Plot ──────────────────────────────────────────────────
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(t_w, Signal_ref_w, label='Signal_ref', linewidth=1.5)
        axes[0].plot(t_w, Signal_w, label='Signal', linewidth=1.0, linestyle='--')
        axes[0].set_ylabel("Current [A]")
        axes[0].legend()
        axes[0].set_title("Waveform comparison")

        axes[1].plot(t_w, error, color='red', linewidth=1.0,
                     label='Error (Signal − Signal_ref)')
        axes[1].axhline(0, color='k', linewidth=0.5)
        axes[1].set_ylabel("Error [A]")
        axes[1].set_xlabel("Time [s]")
        axes[1].legend()
        axes[1].set_title(
            f"Tracking error  |  NRMSE = {NRMSE:.3f}%  |  THD = {THD_percent:.3f}%")

        plt.xlim(t_w[0], t_w[-1])
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    return THD_percent