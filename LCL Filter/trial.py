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
            f"  Reduce I_peak by using more parallel inductor units."
        )
    else:
        print(f"CHECK 1 PASSED: B_peak = {B_peak:.4f} T < B_max = {B_max:.4f} T")

    # ── Check 2: B_peak must be below Bsat ───────────────────────────────────────
    if B_peak >= Bsat:
        raise ValueError(
            f"\nSAFETY CHECK FAILED: Flux density exceeds saturation limit.\n"
            f"B_peak = {B_peak:.4f} T\n"
            f"Bsat   = {Bsat:.4f} T\n"
            f"\nRecommendation:\n"
            f"  Core will saturate and inductance will collapse.\n"
            f"  Increase N or increase Ae immediately."
        )
    else:
        print(f"CHECK 2 PASSED: B_peak = {B_peak:.4f} T < Bsat  = {Bsat:.4f} T")

    # ── Check 3: lg must be positive ─────────────────────────────────────────────
    if lg <= 0:
        raise ValueError(
            f"\nSAFETY CHECK FAILED: Air gap is zero or negative.\n"
            f"lg = {lg * 1000:.2f} mm\n"
            f"\nRecommendation:\n"
            f"  Core is too large for the required inductance.\n"
            f"  Reduce Ae or reduce N."
        )
    else:
        print(f"CHECK 3 PASSED: lg = {lg * 1000:.2f} mm > 0")

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
            f"  Use a custom larger core."
        )
    else:
        print(f"CHECK 4 PASSED: lg = {lg * 1000:.2f} mm < le = {le * 1000:.2f} mm")

    # ── Check 5: lg/le ratio warning ─────────────────────────────────────────────
    lg_le_ratio = lg / le
    if lg_le_ratio > 0.10:
        print(
            f"\nWARNING CHECK 5: Air gap ratio lg/le = {lg_le_ratio * 100:.1f}%\n"
            f"  Recommended maximum is 10%.\n"
            f"  Large air gap causes fringing flux which increases losses.\n"
            f"  Recommendation: Increase Ae to reduce required air gap."
        )
    else:
        print(f"CHECK 5 PASSED: lg/le = {lg_le_ratio * 100:.1f}% < 10%")